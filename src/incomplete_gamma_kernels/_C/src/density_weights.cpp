#include <igk/density_weights.h>

#include <c10/util/Exception.h>

namespace igk
{

torch::Tensor
density_weights_cuda(const torch::Tensor& p, const std::string& scheme, const std::string& energy_term, const float h);

torch::Tensor
density_weights(const torch::Tensor& p, const std::string& scheme, const std::string& energy_term, const float h)
{
    if (p.is_cuda())
    {
        return density_weights_cuda(p, scheme, energy_term, h);
    }

    TORCH_CHECK(false, "No backend implementation available for device \"" + p.device().str() + "\".");
}

} // namespace igk
