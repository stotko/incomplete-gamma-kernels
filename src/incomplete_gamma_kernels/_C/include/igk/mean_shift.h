#pragma once

#include <optional>
#include <string>

#include <torch/types.h>

namespace igk
{

torch::Tensor
kde(const torch::Tensor& p,
    const torch::Tensor& q,
    const std::string& kernel,
    const float h,
    const std::optional<torch::Tensor>& weights = std::nullopt);

} // namespace igk
