#pragma once

#include <string>

#include <torch/types.h>

namespace igk::sparse
{

torch::Tensor
gram(const torch::Tensor& p,
     const std::string& kernel,
     const float variance,
     const float r_truncation = 0.f,
     const bool use_h = false,
     const float h = 1.f);

} // namespace igk::sparse
