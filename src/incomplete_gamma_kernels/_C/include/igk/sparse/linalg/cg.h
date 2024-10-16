#pragma once

#include <functional>
#include <optional>
#include <tuple>

#include <torch/types.h>

namespace igk::sparse::linalg
{

std::tuple<torch::Tensor, int>
cg(const torch::Tensor& A,
   const torch::Tensor& b,
   const std::optional<torch::Tensor>& x0 = std::nullopt,
   const std::optional<int>& maxiter = std::nullopt,
   const std::optional<torch::Tensor>& M = std::nullopt,
   const std::optional<std::function<void(const torch::Tensor&)>>& callback = std::nullopt,
   const float rtol = 1e-5f,
   const float atol = 0.0f);

} // namespace igk::sparse::linalg
