#pragma once

#include <c10/macros/Macros.h>
#include <glm/vec2.hpp>

#include <igk/cuda_utils.h>

namespace igk
{

inline C10_HOST_DEVICE int64_t
numel(const glm::i64vec2& sizes)
{
    return int64_t{ sizes.x * sizes.y };
}

inline C10_HOST_DEVICE glm::i64vec2
unravel_index(const int64_t index, const glm::i64vec2& shape)
{
    CUDA_DEVICE_CHECK(index >= 0);
    CUDA_DEVICE_CHECK(index < numel(shape));

    auto multi_index = glm::i64vec2{ index % shape.x, index / shape.x };

    CUDA_DEVICE_CHECK(multi_index.x >= 0);
    CUDA_DEVICE_CHECK(multi_index.x < shape.x);
    CUDA_DEVICE_CHECK(multi_index.y >= 0);
    CUDA_DEVICE_CHECK(multi_index.y < shape.y);

    return multi_index;
}

inline C10_HOST_DEVICE int64_t
ravel_multi_index(const glm::i64vec2& multi_index, const glm::i64vec2& shape)
{
    CUDA_DEVICE_CHECK(multi_index.x >= 0);
    CUDA_DEVICE_CHECK(multi_index.x < shape.x);
    CUDA_DEVICE_CHECK(multi_index.y >= 0);
    CUDA_DEVICE_CHECK(multi_index.y < shape.y);

    auto index = int64_t{ multi_index.x + multi_index.y * shape.x };

    CUDA_DEVICE_CHECK(index >= 0);
    CUDA_DEVICE_CHECK(index < numel(shape));

    return index;
}

} // namespace igk
