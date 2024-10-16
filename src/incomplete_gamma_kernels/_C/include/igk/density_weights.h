#pragma once

#include <string>

#include <torch/types.h>

namespace igk
{

torch::Tensor
density_weights(const torch::Tensor& p, const std::string& scheme, const std::string& energy_term, const float h);

} // namespace igk
