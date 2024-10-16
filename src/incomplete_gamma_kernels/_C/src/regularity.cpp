#include <igk/regularity.h>

#include <c10/util/Exception.h>

namespace igk
{

float
regularity_cuda(const torch::Tensor& p);

float
regularity(const torch::Tensor& p)
{
    if (p.is_cuda())
    {
        return regularity_cuda(p);
    }

    TORCH_CHECK(false, "No backend implementation available for device \"" + p.device().str() + "\".");
}

} // namespace igk
