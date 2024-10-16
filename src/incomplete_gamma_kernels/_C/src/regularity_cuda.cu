#include <limits>

#include <ATen/Dispatch.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/ThrustAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <glm/geometric.hpp>
#include <glm/gtc/vec1.hpp>
#include <glm/vec3.hpp>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <torch/types.h>

#include <igk/batch_transform_reduce.cuh>
#include <igk/math.h>

namespace igk
{

torch::Tensor
nearest_neighbor_distances(const torch::Tensor& p)
{
    TORCH_CHECK_EQ(p.dim(), 2);
    TORCH_CHECK_EQ(p.size(1), 3);

    at::cuda::CUDAGuard device_guard{ p.device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    const auto dtype_scalar = torch::TensorOptions{}.dtype(p.dtype()).device(p.device());
    const auto N = p.size(0);

    auto distances = torch::empty({ N }, dtype_scalar).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            p.scalar_type(),
            "nearest_neighbor_distances",
            [&]()
            {
                auto p_ = p.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();
                auto scalar_max = std::numeric_limits<scalar_t>::max();

                batch_transform_reduce(
                        distances.data_ptr<scalar_t>(),
                        N,
                        N,
                        [p_, scalar_max] C10_DEVICE(const int64_t j, const int64_t i) -> scalar_t
                        {
                            auto p_i = glm::vec3{ p_[i][0], p_[i][1], p_[i][2] };
                            auto p_j = glm::vec3{ p_[j][0], p_[j][1], p_[j][2] };

                            return i == j ? scalar_max : static_cast<scalar_t>(glm::distance(p_i, p_j));
                        },
                        thrust::minimum<scalar_t>{},
                        scalar_max,
                        p.device(),
                        stream);
            });

    return distances;
}

float
regularity_cuda(const torch::Tensor& p)
{
    TORCH_CHECK_EQ(p.dim(), 2);
    TORCH_CHECK_EQ(p.size(1), 3);

    at::cuda::CUDAGuard device_guard{ p.device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    at::cuda::ThrustAllocator allocator;
    const auto policy = thrust::cuda::par(allocator).on(stream);

    auto nn_distances = nearest_neighbor_distances(p).contiguous();
    const auto N = nn_distances.size(0);

    float sigma = 0.f;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(p.scalar_type(),
                                        "regularity",
                                        [&]()
                                        {
                                            auto variance_raw = thrust::transform_reduce(
                                                    policy,
                                                    nn_distances.data_ptr<scalar_t>(),
                                                    nn_distances.data_ptr<scalar_t>() + N,
                                                    [] C10_HOST_DEVICE(const scalar_t d) -> Variance<glm::vec1>
                                                    {
                                                        auto weight = Variance<glm::vec1>{};
                                                        weight.add_sample(glm::vec1{ static_cast<float>(d) });
                                                        return weight;
                                                    },
                                                    Variance<glm::vec1>{},
                                                    thrust::plus<Variance<glm::vec1>>{});

                                            sigma = static_cast<float>(std::sqrt(variance_raw.variance().x));
                                        });

    return sigma;
}

} // namespace igk
