#pragma once

#include <torch/types.h>

namespace igk
{

float
regularity(const torch::Tensor& p);

} // namespace igk
