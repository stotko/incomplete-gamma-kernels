#pragma once

#include <optional>
#include <string>

#include <torch/types.h>

namespace igk
{

torch::Tensor
lop(const torch::Tensor& p,
    const torch::Tensor& q0,
    const std::string& attraction_kernel,
    const std::string& density_weight_scheme,
    const std::string& repulsion_function,
    const float mu,
    const float h_percent_bb,
    const int iterations,
    const std::optional<float>& attraction_p_norm = std::nullopt,
    const std::optional<std::string>& attraction_approx_params = std::nullopt,
    const bool verbose = false);

} // namespace igk
