#include <limits>

#include <ATen/Dispatch.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/ThrustAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <glm/vec3.hpp>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <torch/types.h>

#include <igk/batch_transform_reduce.cuh>
#include <igk/density_weights.h>
#include <igk/dispatch.h>
#include <igk/kernels.h>
#include <igk/math.h>

namespace igk
{

struct ExactAlphaKernel
{
    inline C10_HOST_DEVICE float
    operator()(const glm::vec3& p, const glm::vec3& q, const float h, const float shift) const
    {
        return alpha_kernel(p, q, h, shift);
    }
};

struct GeneralizedAlphaKernel
{
    explicit GeneralizedAlphaKernel(const float p_norm)
      : p_norm_(p_norm)
    {
    }

    inline C10_HOST_DEVICE float
    operator()(const glm::vec3& p, const glm::vec3& q, const float h, const float shift) const
    {
        return generalized_alpha_kernel(p, q, h, p_norm_, shift);
    }

private:
    float p_norm_;
};

struct ApproximatedAlphaKernel
{
    struct ApproxParams
    {
        ApproxParams() = default;

        ApproxParams(const glm::vec3& w_k, const glm::vec3& sigma_k)
          : w_k_(w_k)
          , sigma_k_(sigma_k)
        {
        }

        inline static ApproxParams
        clop()
        {
            return { glm::vec3{ 97.761f, 29.886f, 11.453f }, glm::vec3{ 0.01010f, 0.03287f, 0.11772f } };
        }

        inline static ApproxParams
        ours()
        {
            return { glm::vec3{ 61.50926f, 11.93220f, 5.06884f }, glm::vec3{ 0.02102f, 0.07289f, 0.15700f } };
        }

        inline static ApproxParams
        ours_consistent()
        {
            return { glm::vec3{ 46.40851f, 9.63450f, 2.67366f }, glm::vec3{ 0.03118f, 0.10582f, sqrtf(1.f / 32.f) } };
        }

        glm::vec3 w_k_ = glm::vec3{ 0.f };
        glm::vec3 sigma_k_ = glm::vec3{ 1.f };
    };

    explicit ApproximatedAlphaKernel(const std::optional<std::string>& params)
    {
        if (params)
        {
            if (*params == "clop")
            {
                params_ = ApproxParams::clop();
            }
            else if (*params == "ours")
            {
                params_ = ApproxParams::ours();
            }
            else if (*params == "ours_consistent")
            {
                params_ = ApproxParams::ours_consistent();
            }
            else
            {
                TORCH_CHECK(false, "Unsupported params \"" + *params + "\".");
            }
        }
    }

    inline C10_HOST_DEVICE float
    operator()(const glm::vec3& p, const glm::vec3& q, const float h, const float shift) const
    {
        return approximated_alpha_kernel(p, q, h, params_.w_k_, params_.sigma_k_, shift);
    }

private:
    ApproxParams params_;
};

template <>
struct dispatch_traits<ExactAlphaKernel>
{
    static constexpr std::string_view kind = "exact";
};

template <>
struct dispatch_traits<GeneralizedAlphaKernel>
{
    static constexpr std::string_view kind = "generalized";
};

template <>
struct dispatch_traits<ApproximatedAlphaKernel>
{
    static constexpr std::string_view kind = "approximated";
};

torch::Tensor
attraction_vectors(const torch::Tensor& p,
                   const torch::Tensor& q,
                   const float h,
                   const torch::Tensor& weights,
                   const bool initial_iteration,
                   const std::string& attraction_kernel,
                   const std::optional<float>& attraction_p_norm,
                   const std::optional<std::string>& attraction_approx_params)
{
    at::cuda::CUDAGuard device_guard{ q.device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    at::cuda::ThrustAllocator allocator;
    const auto policy = thrust::cuda::par(allocator).on(stream);

    const auto dtype_scalar = torch::TensorOptions{}.dtype(q.dtype()).device(q.device());
    const auto dtype_uint8 = torch::TensorOptions{}.dtype(torch::kUInt8).device(q.device());
    const auto N = p.size(0);
    const auto M = q.size(0);

    auto shifts = torch::empty({ M }, dtype_scalar).contiguous();
    auto vectors = torch::empty(q.sizes(), dtype_scalar).contiguous();
    auto means_raw = torch::empty({ M * static_cast<int64_t>(sizeof(Mean<glm::vec3>)) }, dtype_uint8).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(),
                                        "prepare_shifts",
                                        [&]()
                                        {
                                            auto p_ = p.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();
                                            auto q_ = q.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();

                                            batch_transform_reduce(
                                                    shifts.data_ptr<scalar_t>(),
                                                    M,
                                                    N,
                                                    [p_, q_, h] C10_DEVICE(const int64_t j, const int64_t i) -> scalar_t
                                                    {
                                                        auto p_i = glm::vec3{ p_[i][0], p_[i][1], p_[i][2] };
                                                        auto q_j = glm::vec3{ q_[j][0], q_[j][1], q_[j][2] };

                                                        return -log_theta_kernel(p_i, q_j, h);
                                                    },
                                                    thrust::minimum<scalar_t>{},
                                                    std::numeric_limits<scalar_t>::max(),
                                                    q.device(),
                                                    stream);
                                        });

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            q.scalar_type(),
            "attraction_vectors",
            [&]()
            {
                auto p_ = p.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();
                auto q_ = q.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();
                auto shifts_ = shifts.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>();
                auto weights_ = weights.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>();

                auto exact_alpha_kernel = ExactAlphaKernel{};
                auto generalized_alpha_kernel = GeneralizedAlphaKernel{ attraction_p_norm ? *attraction_p_norm : 1.f };
                auto approximated_alpha_kernel = ApproximatedAlphaKernel{
                    attraction_kernel == "approximated"
                            ? (attraction_approx_params ? attraction_approx_params : std::optional<std::string>{ "" })
                            : std::nullopt
                };

                IGK_DISPATCH_FUNCTORS_BY_STRING_3(
                        attraction_kernel,
                        kind,
                        exact_alpha_kernel,
                        generalized_alpha_kernel,
                        approximated_alpha_kernel,
                        [&]()
                        {
                            batch_transform_reduce(
                                    static_cast<Mean<glm::vec3>*>(means_raw.data_ptr()),
                                    M,
                                    N,
                                    [p_, q_, shifts_, weights_, h, initial_iteration, functor] C10_DEVICE(
                                            const int64_t j,
                                            const int64_t i) -> Mean<glm::vec3>
                                    {
                                        auto p_i = glm::vec3{ p_[i][0], p_[i][1], p_[i][2] };
                                        auto q_j = glm::vec3{ q_[j][0], q_[j][1], q_[j][2] };
                                        auto shift = shifts_[j];

                                        auto alpha_ij = initial_iteration ? theta_kernel(p_i, q_j, h, shift)
                                                                          : functor(p_i, q_j, h, shift);

                                        auto w_ij = weights_[i];

                                        auto weighted_attraction = Mean<glm::vec3>{};
                                        weighted_attraction.add_sample(p_i - q_j, w_ij * alpha_ij);

                                        return weighted_attraction;
                                    },
                                    thrust::plus<Mean<glm::vec3>>{},
                                    Mean<glm::vec3>{},
                                    q.device(),
                                    stream);
                        });

                static_assert(sizeof(thrust::tuple<scalar_t, scalar_t, scalar_t>) == 3 * sizeof(scalar_t),
                              "Expected no padding in tuple for type casting");
                thrust::transform(policy,
                                  static_cast<Mean<glm::vec3>*>(means_raw.data_ptr()),
                                  static_cast<Mean<glm::vec3>*>(means_raw.data_ptr()) + M,
                                  static_cast<thrust::tuple<scalar_t, scalar_t, scalar_t>*>(vectors.data_ptr()),
                                  [] C10_HOST_DEVICE(
                                          const Mean<glm::vec3>& value) -> thrust::tuple<scalar_t, scalar_t, scalar_t> {
                                      return { value.mean().x, value.mean().y, value.mean().z };
                                  });
                AT_CUDA_CHECK(cudaStreamSynchronize(stream));
            });

    return vectors;
}

torch::Tensor
repulsion_vectors(const torch::Tensor& q,
                  const std::string& function,
                  const float h,
                  const torch::Tensor& weights,
                  const bool initial_iteration)
{
    if (initial_iteration)
    {
        return torch::zeros_like(q);
    }

    bool use_wlop_eta;
    if (function == "original")
    {
        use_wlop_eta = false;
    }
    else if (function == "wlop")
    {
        use_wlop_eta = true;
    }
    else
    {
        TORCH_CHECK(false, "Unsupported repulsion function \"" + function + "\".");
    }

    at::cuda::CUDAGuard device_guard{ q.device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    at::cuda::ThrustAllocator allocator;
    const auto policy = thrust::cuda::par(allocator).on(stream);

    const auto dtype_scalar = torch::TensorOptions{}.dtype(q.dtype()).device(q.device());
    const auto dtype_uint8 = torch::TensorOptions{}.dtype(torch::kUInt8).device(q.device());
    const auto M = q.size(0);

    auto vectors = torch::empty(q.sizes(), dtype_scalar).contiguous();
    auto means_raw = torch::empty({ M * static_cast<int64_t>(sizeof(Mean<glm::vec3>)) }, dtype_uint8).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            q.scalar_type(),
            "repulsion_vectors",
            [&]()
            {
                auto q_ = q.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();
                auto weights_ = weights.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>();

                batch_transform_reduce(
                        static_cast<Mean<glm::vec3>*>(means_raw.data_ptr()),
                        M,
                        M,
                        [q_, weights_, h, use_wlop_eta] C10_DEVICE(const int64_t j, const int64_t i) -> Mean<glm::vec3>
                        {
                            // Skipping self is crucial
                            if (i == j)
                            {
                                return Mean<glm::vec3>{};
                            }

                            auto q_i = glm::vec3{ q_[i][0], q_[i][1], q_[i][2] };
                            auto q_j = glm::vec3{ q_[j][0], q_[j][1], q_[j][2] };

                            auto d_eta_dr = use_wlop_eta ? wlop_d_eta_d_r(q_i, q_j) : original_d_eta_d_r(q_i, q_j);
                            auto beta_ij = alpha_kernel(q_i, q_j, h) * fabsf(d_eta_dr);

                            auto w_ij = weights_[i];

                            auto weighted_repulsion = Mean<glm::vec3>{};
                            weighted_repulsion.add_sample(q_i - q_j, w_ij * beta_ij);

                            return weighted_repulsion;
                        },
                        thrust::plus<Mean<glm::vec3>>{},
                        Mean<glm::vec3>{},
                        q.device(),
                        stream);

                static_assert(sizeof(thrust::tuple<scalar_t, scalar_t, scalar_t>) == 3 * sizeof(scalar_t),
                              "Expected no padding in tuple for type casting");
                thrust::transform(
                        policy,
                        static_cast<Mean<glm::vec3>*>(means_raw.data_ptr()),
                        static_cast<Mean<glm::vec3>*>(means_raw.data_ptr()) + M,
                        static_cast<thrust::tuple<scalar_t, scalar_t, scalar_t>*>(vectors.data_ptr()),
                        [] C10_HOST_DEVICE(const Mean<glm::vec3>& w) -> thrust::tuple<scalar_t, scalar_t, scalar_t> {
                            return { w.mean().x, w.mean().y, w.mean().z };
                        });
                AT_CUDA_CHECK(cudaStreamSynchronize(stream));
            });

    return vectors;
}

float
bounding_box_diagonal(const torch::Tensor& p)
{
    auto [p_min, _min_indices] = torch::min(p, 0);
    auto [p_max, _max_indices] = torch::max(p, 0);

    auto p_min_host = p_min.cpu();
    auto p_max_host = p_max.cpu();

    return glm::distance(
            glm::vec3{ p_min_host[0].item<float>(), p_min_host[1].item<float>(), p_min_host[2].item<float>() },
            glm::vec3{ p_max_host[0].item<float>(), p_max_host[1].item<float>(), p_max_host[2].item<float>() });
}

float
window_size(const torch::Tensor& p, const float h_percent_bb)
{
    if (p.size(0) == 1)
    {
        // Assume bb_diag = 1 for a single point
        return (h_percent_bb / 100.f);
    }
    else
    {
        return (h_percent_bb / 100.f) * bounding_box_diagonal(p);
    }
}

torch::Tensor
lop_cuda(const torch::Tensor& p,
         const torch::Tensor& q0,
         const std::string& attraction_kernel,
         const std::string& density_weight_scheme,
         const std::string& repulsion_function,
         const float mu,
         const float h_percent_bb,
         const int iterations,
         const std::optional<float>& attraction_p_norm,
         const std::optional<std::string>& attraction_approx_params,
         const bool verbose)
{
    TORCH_CHECK_EQ(p.device(), q0.device());
    TORCH_CHECK_EQ(p.dtype(), q0.dtype());
    TORCH_CHECK_EQ(p.dim(), 2);
    TORCH_CHECK_EQ(p.size(1), 3);
    TORCH_CHECK_EQ(q0.dim(), 2);
    TORCH_CHECK_EQ(q0.size(1), 3);
    TORCH_CHECK_GT(h_percent_bb, 0.f);
    TORCH_CHECK_GE(iterations, 0);

    auto q = q0.clone();

    if (p.size(0) == 0 || q.size(0) == 0)
    {
        return q;
    }

    auto h = window_size(p, h_percent_bb);
    auto attraction_weights = density_weights(p, density_weight_scheme, "attraction", h);

    for (auto i = 0; i < iterations; ++i)
    {
        if (verbose)
        {
            printf("lop [%3d/%3d]: Projecting ...\n", i + 1, iterations);
        }

        auto attraction = attraction_vectors(p,
                                             q,
                                             h,
                                             attraction_weights,
                                             i == 0,
                                             attraction_kernel,
                                             attraction_p_norm,
                                             attraction_approx_params);

        auto repulsion_weights = density_weights(q, density_weight_scheme, "repulsion", h);
        auto repulsion = repulsion_vectors(q, repulsion_function, h, repulsion_weights, i == 0);

        q += attraction;
        q -= (mu * repulsion);

        if (verbose)
        {
            auto attraction_lengths = (attraction * attraction).sum(1).sqrt() * (100.f / bounding_box_diagonal(p));
            auto repulsion_lengths =
                    ((-mu * repulsion) * (-mu * repulsion)).sum(1).sqrt() * (100.f / bounding_box_diagonal(p));
            auto combined_lengths = ((attraction - mu * repulsion) * (attraction - mu * repulsion)).sum(1).sqrt() *
                                    (100.f / bounding_box_diagonal(p));

            printf("lop [%3d/%3d]: Projecting ... done: Attraction %6.4f (%6.4f), Repulsion %6.4f (%6.4f), Combined "
                   "%6.4f (%6.4f)\n",
                   i + 1,
                   iterations,
                   attraction_lengths.mean().to(torch::kFloat64).cpu().item<double>(),
                   attraction_lengths.std().to(torch::kFloat64).cpu().item<double>(),
                   repulsion_lengths.mean().to(torch::kFloat64).cpu().item<double>(),
                   repulsion_lengths.std().to(torch::kFloat64).cpu().item<double>(),
                   combined_lengths.mean().to(torch::kFloat64).cpu().item<double>(),
                   combined_lengths.std().to(torch::kFloat64).cpu().item<double>());
        }
    }

    return q;
}

} // namespace igk
