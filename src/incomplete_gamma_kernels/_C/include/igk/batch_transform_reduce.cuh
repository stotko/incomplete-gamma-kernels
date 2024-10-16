#pragma once

#include <ATen/cuda/Exceptions.h>
#include <c10/macros/Macros.h>
#include <cub/device/device_segmented_reduce.cuh>
#include <glm/vec2.hpp>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <torch/types.h>

#include <igk/arange_iterator.h>
#include <igk/indexing.h>

namespace igk
{

namespace detail
{

template <typename UnaryFunction>
struct BatchIndexWrapper
{
    inline explicit BatchIndexWrapper(UnaryFunction unary_op, int64_t num_batches, int64_t num_items_per_batch)
      : unary_op_(unary_op)
      , num_batches_(num_batches)
      , num_items_per_batch_(num_items_per_batch)
    {
    }

    inline C10_HOST_DEVICE auto
    operator()(const int64_t k)
    {
        auto index = unravel_index(k, glm::i64vec2{ num_items_per_batch_, num_batches_ });
        auto batch = index.y;
        auto item_in_batch = index.x;

        return unary_op_(batch, item_in_batch);
    }

    UnaryFunction unary_op_;
    int64_t num_batches_;
    int64_t num_items_per_batch_;
};

} // namespace detail

template <typename OutputIterator, typename UnaryFunction, typename BinaryFunction, typename T>
inline void
batch_transform_reduce(OutputIterator output,
                       int64_t num_batches,
                       int64_t num_items_per_batch,
                       UnaryFunction unary_op,
                       BinaryFunction binary_op,
                       T initial_value,
                       torch::Device device = torch::kCUDA,
                       cudaStream_t stream = 0)
{
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    auto wrapped_unary_op = detail::BatchIndexWrapper<UnaryFunction>{ unary_op, num_batches, num_items_per_batch };

    AT_CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(
            d_temp_storage,
            temp_storage_bytes,
            thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0), wrapped_unary_op),
            output,
            static_cast<int>(num_batches),
            arange_iterator<int64_t>(0, num_items_per_batch),
            arange_iterator<int64_t>(0, num_items_per_batch) + 1,
            binary_op,
            initial_value,
            stream));
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    const auto dtype_uint8 = torch::TensorOptions{}.dtype(torch::kUInt8).device(device);
    auto temp_storage = torch::empty({ static_cast<int64_t>(temp_storage_bytes) }, dtype_uint8).contiguous();

    AT_CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0), wrapped_unary_op),
            output,
            static_cast<int>(num_batches),
            arange_iterator<int64_t>(0, num_items_per_batch),
            arange_iterator<int64_t>(0, num_items_per_batch) + 1,
            binary_op,
            initial_value,
            stream));
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace igk
