#include <ATen/Dispatch.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/ThrustAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <glm/gtc/vec1.hpp>
#include <glm/vec3.hpp>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <torch/types.h>

#include <igk/batch_transform_reduce.cuh>
#include <igk/dispatch.h>
#include <igk/kernels.h>
#include <igk/math.h>

namespace igk
{

struct LopKdeKernel
{
    inline C10_HOST_DEVICE float
    operator()(const glm::vec3& p, const glm::vec3& q, const float variance, const float h) const
    {
        return lop_kernel(p, q, variance, h);
    }
};

template <>
struct dispatch_traits<LopKdeKernel>
{
    static constexpr std::string_view kind = "lop";
};

torch::Tensor
kde_cuda(const torch::Tensor& p,
         const torch::Tensor& q,
         const std::string& kernel,
         const float h,
         const std::optional<torch::Tensor>& weights)
{
    TORCH_CHECK_EQ(p.device(), q.device());
    TORCH_CHECK_EQ(p.dtype(), q.dtype());
    TORCH_CHECK_EQ(p.dim(), 2);
    TORCH_CHECK_EQ(p.size(1), 3);
    TORCH_CHECK_EQ(q.dim(), 2);
    TORCH_CHECK_EQ(q.size(1), 3);
    TORCH_CHECK_GT(h, 0.f);
    if (weights)
    {
        TORCH_CHECK_EQ(weights->device(), p.device());
        TORCH_CHECK_EQ(weights->dtype(), p.dtype());
        TORCH_CHECK_EQ(weights->dim(), 1);
        TORCH_CHECK_EQ(weights->size(0), p.size(0));
    }

    at::cuda::CUDAGuard device_guard{ q.device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    at::cuda::ThrustAllocator allocator;
    const auto policy = thrust::cuda::par(allocator).on(stream);

    const auto dtype_scalar = torch::TensorOptions{}.dtype(q.dtype()).device(q.device());
    const auto dtype_uint8 = torch::TensorOptions{}.dtype(torch::kUInt8).device(q.device());
    const auto N = p.size(0);
    const auto M = q.size(0);

    if (N == 0)
    {
        return torch::zeros({ M }, dtype_scalar);
    }
    if (M == 0)
    {
        return torch::empty({ 0 }, dtype_scalar);
    }

    auto w = weights ? weights->contiguous() : torch::ones({ N }, dtype_scalar).contiguous();

    auto hat_f = torch::empty({ M }, dtype_scalar).contiguous();
    auto means_raw = torch::empty({ M * static_cast<int64_t>(sizeof(Mean<glm::vec1>)) }, dtype_uint8).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            q.scalar_type(),
            "kde",
            [&]()
            {
                auto p_ = p.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();
                auto q_ = q.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();
                auto w_ = w.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>();
                auto variance = 1.f / 32.f;

                auto lop_kde_kernel = LopKdeKernel{};

                IGK_DISPATCH_FUNCTORS_BY_STRING_1(
                        kernel,
                        kind,
                        lop_kde_kernel,
                        [&]()
                        {
                            batch_transform_reduce(
                                    static_cast<Mean<glm::vec1>*>(means_raw.data_ptr()),
                                    M,
                                    N,
                                    [p_, q_, w_, h, variance, functor] C10_DEVICE(const int64_t j,
                                                                                  const int64_t i) -> Mean<glm::vec1>
                                    {
                                        auto p_i = glm::vec3{ p_[i][0], p_[i][1], p_[i][2] };
                                        auto q_j = glm::vec3{ q_[j][0], q_[j][1], q_[j][2] };

                                        auto w_i = w_[i];

                                        auto weight = Mean<glm::vec1>{};
                                        weight.add_sample(
                                                glm::vec1{ static_cast<float>(w_i) * functor(p_i, q_j, variance, h) });

                                        return weight;
                                    },
                                    thrust::plus<Mean<glm::vec1>>{},
                                    Mean<glm::vec1>{},
                                    q.device(),
                                    stream);
                        });

                thrust::transform(policy,
                                  static_cast<Mean<glm::vec1>*>(means_raw.data_ptr()),
                                  static_cast<Mean<glm::vec1>*>(means_raw.data_ptr()) + M,
                                  hat_f.data_ptr<scalar_t>(),
                                  [h] C10_HOST_DEVICE(const Mean<glm::vec1>& value) -> scalar_t
                                  { return (1.f / (h * h * h)) * value.mean().x; });
                AT_CUDA_CHECK(cudaStreamSynchronize(stream));
            });

    return hat_f;
}

} // namespace igk
