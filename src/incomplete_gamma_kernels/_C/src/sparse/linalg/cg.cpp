#include <igk/sparse/linalg/cg.h>

#include <ATen/cuda/Exceptions.h>

namespace igk::sparse::linalg
{

std::tuple<torch::Tensor, int>
cg_cuda(const torch::Tensor& A,
        const torch::Tensor& b,
        const std::optional<torch::Tensor>& x0,
        const std::optional<int>& maxiter,
        const std::optional<torch::Tensor>& M,
        const std::optional<std::function<void(const torch::Tensor&)>>& callback,
        const float rtol,
        const float atol);

std::tuple<torch::Tensor, int>
cg(const torch::Tensor& A,
   const torch::Tensor& b,
   const std::optional<torch::Tensor>& x0,
   const std::optional<int>& maxiter,
   const std::optional<torch::Tensor>& M,
   const std::optional<std::function<void(const torch::Tensor&)>>& callback,
   const float rtol,
   const float atol)
{
    if (A.is_cuda())
    {
        return cg_cuda(A, b, x0, maxiter, M, callback, rtol, atol);
    }

    TORCH_CHECK(false, "No backend implementation available for device \"" + A.device().str() + "\".");
}

} // namespace igk::sparse::linalg
