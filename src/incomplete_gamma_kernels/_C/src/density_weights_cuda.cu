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
#include <igk/kernels.h>
#include <igk/math.h>
#include <igk/mean_shift.h>
#include <igk/sparse/gram.h>
#include <igk/sparse/linalg/cg.h>

namespace igk
{

torch::Tensor
none_attraction_weights(const torch::Tensor& p, [[maybe_unused]] const float h)
{
    const auto dtype_scalar = torch::TensorOptions{}.dtype(p.dtype()).device(p.device());
    const auto N = p.size(0);

    return torch::ones({ N }, dtype_scalar);
}

torch::Tensor
none_repulsion_weights(const torch::Tensor& p, const float h)
{
    return none_attraction_weights(p, h);
}

torch::Tensor
wlop_attraction_weights(const torch::Tensor& p, const float h)
{
    at::cuda::CUDAGuard device_guard{ p.device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    at::cuda::ThrustAllocator allocator;
    const auto policy = thrust::cuda::par(allocator).on(stream);

    const auto dtype_scalar = torch::TensorOptions{}.dtype(p.dtype()).device(p.device());
    const auto dtype_uint8 = torch::TensorOptions{}.dtype(torch::kUInt8).device(p.device());
    const auto N = p.size(0);

    auto weights = torch::empty({ N }, dtype_scalar).contiguous();
    auto means_raw = torch::empty({ N * static_cast<int64_t>(sizeof(Mean<glm::vec1>)) }, dtype_uint8).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            p.scalar_type(),
            "wlop_attraction_weights",
            [&]()
            {
                auto p_ = p.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();

                batch_transform_reduce(
                        static_cast<Mean<glm::vec1>*>(means_raw.data_ptr()),
                        N,
                        N,
                        [p_, h] C10_DEVICE(const int64_t j, const int64_t i) -> Mean<glm::vec1>
                        {
                            auto p_i = glm::vec3{ p_[i][0], p_[i][1], p_[i][2] };
                            auto p_j = glm::vec3{ p_[j][0], p_[j][1], p_[j][2] };

                            auto weight = Mean<glm::vec1>{};
                            weight.add_sample(glm::vec1{ theta_kernel(p_i, p_j, h) });

                            return weight;
                        },
                        thrust::plus<Mean<glm::vec1>>{},
                        Mean<glm::vec1>{},
                        p.device(),
                        stream);

                thrust::transform(policy,
                                  static_cast<Mean<glm::vec1>*>(means_raw.data_ptr()),
                                  static_cast<Mean<glm::vec1>*>(means_raw.data_ptr()) + N,
                                  weights.data_ptr<scalar_t>(),
                                  [N] C10_HOST_DEVICE(const Mean<glm::vec1>& value) -> scalar_t
                                  {
                                      // Computing mean value is numerically more robust than direct sum
                                      return 1.f / (value.mean().x * static_cast<float>(N));
                                  });
                AT_CUDA_CHECK(cudaStreamSynchronize(stream));
            });

    return weights;
}

torch::Tensor
wlop_repulsion_weights(const torch::Tensor& p, const float h)
{
    return wlop_attraction_weights(p, h).reciprocal_();
}

torch::Tensor
ours_simple_attraction_weights(const torch::Tensor& p, const float h)
{
    return kde(p, p, "lop", h).reciprocal_();
}

torch::Tensor
ours_simple_repulsion_weights(const torch::Tensor& p, const float h)
{
    return kde(p, p, "lop", h);
}

torch::Tensor
ours_attraction_weights(const torch::Tensor& p, const float h)
{
    const auto dtype_scalar = torch::TensorOptions{}.dtype(p.dtype()).device(p.device());
    const auto N = p.size(0);

    auto G = igk::sparse::gram(p, "lop", 1.f / 32.f, 0.5f * h, true, h);
    auto ones = torch::ones({ N }, dtype_scalar);

    auto weights = torch::Tensor{};
    std::tie(weights, std::ignore) = igk::sparse::linalg::cg(G, ones);

    return weights;
}

torch::Tensor
ours_repulsion_weights(const torch::Tensor& p, const float h)
{
    return ours_simple_repulsion_weights(p, h);
}

torch::Tensor
density_weights_cuda(const torch::Tensor& p, const std::string& scheme, const std::string& energy_term, const float h)
{
    TORCH_CHECK_EQ(p.dim(), 2);
    TORCH_CHECK_EQ(p.size(1), 3);
    TORCH_CHECK_GT(h, 0.f);

    if (energy_term == "attraction")
    {
        if (scheme == "none")
        {
            return none_attraction_weights(p, h);
        }
        else if (scheme == "wlop")
        {
            return wlop_attraction_weights(p, h);
        }
        else if (scheme == "ours_simple")
        {
            return ours_simple_attraction_weights(p, h);
        }
        else if (scheme == "ours")
        {
            return ours_attraction_weights(p, h);
        }
        else
        {
            TORCH_CHECK(false, "Unsupported scheme \"" + scheme + "\".");
        }
    }
    else if (energy_term == "repulsion")
    {
        if (scheme == "none")
        {
            return none_repulsion_weights(p, h);
        }
        else if (scheme == "wlop")
        {
            return wlop_repulsion_weights(p, h);
        }
        else if (scheme == "ours_simple")
        {
            return ours_simple_repulsion_weights(p, h);
        }
        else if (scheme == "ours")
        {
            return ours_repulsion_weights(p, h);
        }
        else
        {
            TORCH_CHECK(false, "Unsupported scheme \"" + scheme + "\".");
        }
    }
    else
    {
        TORCH_CHECK(false, "Unsupported energy_term \"" + energy_term + "\".");
    }
}

} // namespace igk
