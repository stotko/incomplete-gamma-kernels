#pragma once

#include <c10/macros/Macros.h>
#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/vec3.hpp>
#include <torch/types.h>

#include <igk/math.h>

namespace igk
{

inline C10_HOST_DEVICE float
log_theta_kernel(const glm::vec3& p, const glm::vec3& q, const float h, const float shift = 0.f)
{
    auto r = glm::distance(p, q);

    return -(r / (h / 4.f)) * (r / (h / 4.f)) + shift;
}

inline C10_HOST_DEVICE float
theta_kernel(const glm::vec3& p, const glm::vec3& q, const float h, const float shift = 0.f)
{
    auto r = glm::distance(p, q);

    return expf(-(r / (h / 4.f)) * (r / (h / 4.f)) + shift);
}

inline C10_HOST_DEVICE float
alpha_kernel(const glm::vec3& p, const glm::vec3& q, const float h, const float shift = 0.f)
{
    auto r = glm::distance(p, q);

    return expf(-(r / (h / 4.f)) * (r / (h / 4.f)) + shift) / fmaxf(r, epsilon());
}

inline C10_HOST_DEVICE float
generalized_alpha_kernel(const glm::vec3& p,
                         const glm::vec3& q,
                         const float h,
                         const float p_norm = 1.f,
                         const float shift = 0.f)
{
    auto r = glm::distance(p, q);

    return expf(-(r / (h / 4.f)) * (r / (h / 4.f)) + shift) * powf(fmaxf(r, epsilon()), p_norm - 2.f);
}

inline C10_HOST_DEVICE float
approximated_alpha_kernel(const glm::vec3& p,
                          const glm::vec3& q,
                          const float h,
                          const glm::vec3& w_k,
                          const glm::vec3& sigma_k,
                          const float shift = 0.f)
{
    auto r = glm::distance(p, q);

    return w_k.x * expf(-0.5f * (r * r) / ((sigma_k.x * sigma_k.x) * (h * h)) + shift) +
           w_k.y * expf(-0.5f * (r * r) / ((sigma_k.y * sigma_k.y) * (h * h)) + shift) +
           w_k.z * expf(-0.5f * (r * r) / ((sigma_k.z * sigma_k.z) * (h * h)) + shift);
}

inline C10_HOST_DEVICE float
original_d_eta_d_r(const glm::vec3& p, const glm::vec3& q)
{
    auto r = glm::distance(p, q);

    return -1.f / fmaxf(r * r * r * r, epsilon());
}

inline C10_HOST_DEVICE float
wlop_d_eta_d_r([[maybe_unused]] const glm::vec3& p, [[maybe_unused]] const glm::vec3& q)
{
    return -1.f;
}

inline C10_HOST_DEVICE float
gaussian_kernel(const glm::vec3& p, const glm::vec3& q, const float variance, const float h = 1.f)
{
    auto r = glm::distance(p, q);
    constexpr auto d = float{ 3 };

    auto c = 1.f / powf(2.f * glm::pi<float>() * variance, d / 2.f);

    return c * expf((-0.5f / variance) * ((r / h) * (r / h)));
}

inline C10_HOST_DEVICE float
lop_kernel(const glm::vec3& p, const glm::vec3& q, const float variance, const float h = 1.f)
{
    auto r = glm::distance(p, q);
    constexpr auto d = float{ 3 };

    auto c = (1.f / powf(2.f * glm::pi<float>() * variance, d / 2.f) * tgammaf((d + 2.f) / 2.f) /
              tgammaf((d + 1.f) / 2.f) * sqrtf(glm::pi<float>()));

    return c * erfcf(sqrtf(0.5f / variance) * (r / h));
}

} // namespace igk
