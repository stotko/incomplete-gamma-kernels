#include <igk/kernels.h>

#include <c10/util/Exception.h>

namespace igk::sparse
{

torch::Tensor
gram_cuda(const torch::Tensor& p,
          const std::string& kernel,
          const float variance,
          const float r_truncation,
          const bool use_h,
          const float h);

torch::Tensor
gram(const torch::Tensor& p,
     const std::string& kernel,
     const float variance,
     const float r_truncation,
     const bool use_h,
     const float h)
{
    if (p.is_cuda())
    {
        return gram_cuda(p, kernel, variance, r_truncation, use_h, h);
    }

    TORCH_CHECK(false, "No backend implementation available for device \"" + p.device().str() + "\".");
}

} // namespace igk::sparse
