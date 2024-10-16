#pragma once

#include <c10/macros/Macros.h>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtc/vec1.hpp>
#include <glm/mat2x2.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/matrix.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace glm
{

template <typename T, glm::qualifier Q>
inline C10_HOST_DEVICE glm::vec<1, T, Q>
outerProduct(glm::vec<1, T, Q> const& c, glm::vec<1, T, Q> const& r)
{
    return glm::vec<1, T, Q>{ c.x * r.x };
}

} // namespace glm

namespace igk
{

inline C10_HOST_DEVICE float
epsilon()
{
    return 1e-7f;
}

template <typename T>
class Mean;

#define IGK_SPECIALIZE_MEAN(T, ...)                                                                                    \
    template <>                                                                                                        \
    class Mean<T>                                                                                                      \
    {                                                                                                                  \
    public:                                                                                                            \
        inline C10_HOST_DEVICE void                                                                                    \
        add_sample(const T& sample, const float weight = 1.f)                                                          \
        {                                                                                                              \
            _sum_weights += weight;                                                                                    \
            if (glm::epsilonEqual(_sum_weights, 0.f, epsilon()))                                                       \
            {                                                                                                          \
                return;                                                                                                \
            }                                                                                                          \
            _mean += (weight / _sum_weights) * (sample - _mean);                                                       \
        }                                                                                                              \
                                                                                                                       \
        inline C10_HOST_DEVICE T                                                                                       \
        mean() const                                                                                                   \
        {                                                                                                              \
            return _mean;                                                                                              \
        }                                                                                                              \
                                                                                                                       \
        inline C10_HOST_DEVICE Mean<T>                                                                                 \
        operator+(const Mean<T>& other) const                                                                          \
        {                                                                                                              \
            Mean<T> result;                                                                                            \
            result.add_sample(_mean, _sum_weights);                                                                    \
            result.add_sample(other._mean, other._sum_weights);                                                        \
            return result;                                                                                             \
        }                                                                                                              \
                                                                                                                       \
    private:                                                                                                           \
        T _mean{ __VA_ARGS__ };                                                                                        \
        float _sum_weights{ 0.f };                                                                                     \
    };

IGK_SPECIALIZE_MEAN(glm::vec1, 0.f)
IGK_SPECIALIZE_MEAN(glm::vec2, 0.f, 0.f)
IGK_SPECIALIZE_MEAN(glm::vec3, 0.f, 0.f, 0.f)
IGK_SPECIALIZE_MEAN(glm::vec4, 0.f, 0.f, 0.f, 0.f)

#undef IGK_SPECIALIZE_MEAN

template <typename T>
class Variance;

#define IGK_SPECIALIZE_VARIANCE(T, MAT_T, ...)                                                                         \
    template <>                                                                                                        \
    class Variance<T>                                                                                                  \
    {                                                                                                                  \
    public:                                                                                                            \
        inline C10_HOST_DEVICE void                                                                                    \
        add_sample(const T& sample, const float weight = 1.f)                                                          \
        {                                                                                                              \
            _sum_weights += weight;                                                                                    \
            if (glm::epsilonEqual(_sum_weights, 0.f, epsilon()))                                                       \
            {                                                                                                          \
                return;                                                                                                \
            }                                                                                                          \
            auto delta = sample - _mean;                                                                               \
            _mean += (weight / _sum_weights) * delta;                                                                  \
            _scatter += weight * glm::outerProduct(delta, sample - _mean);                                             \
        }                                                                                                              \
                                                                                                                       \
        inline C10_HOST_DEVICE MAT_T                                                                                   \
        variance() const                                                                                               \
        {                                                                                                              \
            return glm::epsilonEqual(_sum_weights, 0.f, epsilon()) ? MAT_T{ 0.f } : _scatter / _sum_weights;           \
        }                                                                                                              \
                                                                                                                       \
        inline C10_HOST_DEVICE Variance<T>                                                                             \
        operator+(const Variance<T>& other) const                                                                      \
        {                                                                                                              \
            if (glm::epsilonEqual(_sum_weights, 0.f, epsilon()))                                                       \
            {                                                                                                          \
                return other;                                                                                          \
            }                                                                                                          \
            if (glm::epsilonEqual(other._sum_weights, 0.f, epsilon()))                                                 \
            {                                                                                                          \
                return *this;                                                                                          \
            }                                                                                                          \
                                                                                                                       \
            Variance<T> result;                                                                                        \
            result._sum_weights = _sum_weights + other._sum_weights;                                                   \
                                                                                                                       \
            if (glm::epsilonEqual(result._sum_weights, 0.f, epsilon()))                                                \
            {                                                                                                          \
                return Variance<T>{};                                                                                  \
            }                                                                                                          \
                                                                                                                       \
            auto delta = other._mean - _mean;                                                                          \
            result._mean = _mean + (other._sum_weights / result._sum_weights) * delta;                                 \
            result._scatter =                                                                                          \
                    _scatter + other._scatter +                                                                        \
                    (_sum_weights * other._sum_weights / result._sum_weights) * glm::outerProduct(delta, delta);       \
                                                                                                                       \
            return result;                                                                                             \
        }                                                                                                              \
                                                                                                                       \
    private:                                                                                                           \
        T _mean{ __VA_ARGS__ };                                                                                        \
        MAT_T _scatter{ 0.f };                                                                                         \
        float _sum_weights{ 0.f };                                                                                     \
    };

IGK_SPECIALIZE_VARIANCE(glm::vec1, glm::vec1, 0.f)
IGK_SPECIALIZE_VARIANCE(glm::vec2, glm::mat2, 0.f, 0.f)
IGK_SPECIALIZE_VARIANCE(glm::vec3, glm::mat3, 0.f, 0.f, 0.f)
IGK_SPECIALIZE_VARIANCE(glm::vec4, glm::mat4, 0.f, 0.f, 0.f, 0.f)

#undef IGK_SPECIALIZE_VARIANCE

} // namespace igk
