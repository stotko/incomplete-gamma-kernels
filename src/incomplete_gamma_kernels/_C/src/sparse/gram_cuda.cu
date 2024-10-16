#include <ATen/Dispatch.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/ThrustAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <cub/device/device_select.cuh>
#include <cuda/functional>
#include <glm/geometric.hpp>
#include <glm/vec3.hpp>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/zip_function.h>
#include <torch/types.h>

#include <igk/batch_transform_reduce.cuh>
#include <igk/dispatch.h>
#include <igk/indexing.h>
#include <igk/kernels.h>

namespace igk
{

struct LopGramKernel
{
    LopGramKernel(const int64_t N, const bool use_h)
      : N_(N)
      , use_h_(use_h)
    {
    }

    inline C10_HOST_DEVICE float
    operator()(const glm::vec3& p, const glm::vec3& q, const float variance, const float h) const
    {
        return use_h_ ? (1.f / (static_cast<float>(N_) * (h * h * h))) * lop_kernel(p, q, variance, h)
                      : lop_kernel(p, q, variance);
    }

private:
    int64_t N_;
    bool use_h_;
};

template <>
struct dispatch_traits<LopGramKernel>
{
    static constexpr std::string_view kind = "lop";
};

} // namespace igk

namespace igk::sparse
{

torch::Tensor
gram_cuda(const torch::Tensor& p,
          const std::string& kernel,
          const float variance,
          const float r_truncation,
          const bool use_h,
          const float h)
{
    TORCH_CHECK_EQ(p.dim(), 2);
    TORCH_CHECK_EQ(p.size(1), 3);
    TORCH_CHECK_GT(variance, 0.f);
    TORCH_CHECK_GT(r_truncation, 0.f);
    TORCH_CHECK_GT(h, 0.f);

    at::cuda::CUDAGuard device_guard{ p.device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    at::cuda::ThrustAllocator allocator;
    const auto policy = thrust::cuda::par(allocator).on(stream);

    const auto dtype_uint8 = torch::TensorOptions{}.dtype(torch::kUInt8).device(p.device());
    const auto dtype_int64 = torch::TensorOptions{}.dtype(torch::kInt64).device(p.device());
    const auto dtype_scalar = torch::TensorOptions{}.dtype(p.dtype()).device(p.device());

    const auto N = p.size(0);

    auto sparse_crow_indices = torch::empty({ N + 1 }, dtype_int64).contiguous();
    sparse_crow_indices.index_put_({ 0 }, 0);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            p.scalar_type(),
            "compute_nnz",
            [&]()
            {
                auto p_ = p.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();

                batch_transform_reduce(
                        sparse_crow_indices.data_ptr<int64_t>() + 1,
                        N,
                        N,
                        [p_, r_truncation] C10_DEVICE(const int64_t j, const int64_t i) -> int64_t
                        {
                            auto p_i = glm::vec3{ p_[i][0], p_[i][1], p_[i][2] };
                            auto p_j = glm::vec3{ p_[j][0], p_[j][1], p_[j][2] };

                            return int64_t{ glm::distance(p_i, p_j) < r_truncation };
                        },
                        thrust::plus<int64_t>{},
                        int64_t{ 0 },
                        p.device(),
                        stream);

                thrust::inclusive_scan(policy,
                                       sparse_crow_indices.data_ptr<int64_t>(),
                                       sparse_crow_indices.data_ptr<int64_t>() + (N + 1),
                                       sparse_crow_indices.data_ptr<int64_t>());
                AT_CUDA_CHECK(cudaStreamSynchronize(stream));
            });

    auto nnz = sparse_crow_indices[N].cpu().item<int64_t>();

    auto sparse_crow_indices_host = sparse_crow_indices.to(torch::kCPU);
    auto sparse_col_indices = torch::empty({ nnz }, dtype_int64).contiguous();
    auto sparse_values = torch::empty({ nnz }, dtype_scalar).contiguous();

    for (int64_t row = 0; row < N; ++row)
    {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                p.scalar_type(),
                "collect_nonzeros",
                [&]()
                {
                    auto p_ = p.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();

                    auto lop_gram_kernel = LopGramKernel{ N, use_h };

                    IGK_DISPATCH_FUNCTORS_BY_STRING_1(
                            kernel,
                            kind,
                            lop_gram_kernel,
                            [&]()
                            {
                                auto f = cuda::proclaim_return_type<thrust::tuple<int64_t, scalar_t, bool>>(
                                        [p_, r_truncation, variance, h, row, functor] C10_DEVICE(
                                                const int64_t j) -> thrust::tuple<int64_t, scalar_t, bool>
                                        {
                                            auto p_i = glm::vec3{ p_[row][0], p_[row][1], p_[row][2] };
                                            auto p_j = glm::vec3{ p_[j][0], p_[j][1], p_[j][2] };

                                            return { j,
                                                     functor(p_i, p_j, variance, h),
                                                     glm::distance(p_i, p_j) < r_truncation };
                                        });

                                void* d_temp_storage = nullptr;
                                size_t temp_storage_bytes = 0;
                                auto num_selected_out = torch::empty({ 1 }, dtype_int64).contiguous();

                                // NOTE: Once cub::DeviceSelect::If supports 64-bit num_items, this per-row
                                // computation can be
                                //       widened to the whole matrix
                                AT_CUDA_CHECK(cub::DeviceSelect::If(
                                        d_temp_storage,
                                        temp_storage_bytes,
                                        thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0), f),
                                        thrust::make_zip_iterator(thrust::make_tuple(
                                                sparse_col_indices.data_ptr<int64_t>() +
                                                        sparse_crow_indices_host[row].item<int64_t>(),
                                                sparse_values.data_ptr<scalar_t>() +
                                                        sparse_crow_indices_host[row].item<int64_t>(),
                                                thrust::make_discard_iterator())),
                                        num_selected_out.data_ptr<int64_t>(),
                                        static_cast<int>(N),
                                        thrust::make_zip_function(
                                                [] C10_HOST_DEVICE([[maybe_unused]] const int64_t col,
                                                                   [[maybe_unused]] const scalar_t value,
                                                                   const bool use) { return use; }),
                                        stream));
                                AT_CUDA_CHECK(cudaStreamSynchronize(stream));

                                auto temp_storage =
                                        torch::empty({ static_cast<int64_t>(temp_storage_bytes) }, dtype_uint8)
                                                .contiguous();

                                AT_CUDA_CHECK(cub::DeviceSelect::If(
                                        temp_storage.data_ptr(),
                                        temp_storage_bytes,
                                        thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0), f),
                                        thrust::make_zip_iterator(thrust::make_tuple(
                                                sparse_col_indices.data_ptr<int64_t>() +
                                                        sparse_crow_indices_host[row].item<int64_t>(),
                                                sparse_values.data_ptr<scalar_t>() +
                                                        sparse_crow_indices_host[row].item<int64_t>(),
                                                thrust::make_discard_iterator())),
                                        num_selected_out.data_ptr<int64_t>(),
                                        static_cast<int>(N),
                                        thrust::make_zip_function(
                                                [] C10_HOST_DEVICE([[maybe_unused]] const int64_t col,
                                                                   [[maybe_unused]] const scalar_t value,
                                                                   const bool use) { return use; }),
                                        stream));
                                AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                            });
                });
    }

    return torch::sparse_csr_tensor(sparse_crow_indices, sparse_col_indices, sparse_values, { N, N }, dtype_scalar);
}

} // namespace igk::sparse
