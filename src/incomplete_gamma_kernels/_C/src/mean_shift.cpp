#include <igk/mean_shift.h>

#include <c10/util/Exception.h>

namespace igk
{

torch::Tensor
kde_cuda(const torch::Tensor& p,
         const torch::Tensor& q,
         const std::string& kernel,
         const float h,
         const std::optional<torch::Tensor>& weights);

torch::Tensor
kde(const torch::Tensor& p,
    const torch::Tensor& q,
    const std::string& kernel,
    const float h,
    const std::optional<torch::Tensor>& weights)
{
    if (p.is_cuda())
    {
        return kde_cuda(p, q, kernel, h, weights);
    }

    TORCH_CHECK(false, "No backend implementation available for device \"" + p.device().str() + "\".");
}

} // namespace igk
