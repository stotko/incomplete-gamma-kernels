#include <igk/lop.h>

#include <c10/util/Exception.h>

namespace igk
{

torch::Tensor
lop_cuda(const torch::Tensor& p,
         const torch::Tensor& q0,
         const std::string& attraction_kernel,
         const std::string& density_weight_scheme,
         const std::string& repulsion_function,
         const float mu,
         const float h_percent_bb,
         const int iterations,
         const std::optional<float>& attraction_p_norm,
         const std::optional<std::string>& attraction_approx_params,
         const bool verbose);

torch::Tensor
lop(const torch::Tensor& p,
    const torch::Tensor& q0,
    const std::string& attraction_kernel,
    const std::string& density_weight_scheme,
    const std::string& repulsion_function,
    const float mu,
    const float h_percent_bb,
    const int iterations,
    const std::optional<float>& attraction_p_norm,
    const std::optional<std::string>& attraction_approx_params,
    const bool verbose)
{
    if (p.is_cuda())
    {
        return lop_cuda(p,
                        q0,
                        attraction_kernel,
                        density_weight_scheme,
                        repulsion_function,
                        mu,
                        h_percent_bb,
                        iterations,
                        attraction_p_norm,
                        attraction_approx_params,
                        verbose);
    }

    TORCH_CHECK(false, "No backend implementation available for device \"" + p.device().str() + "\".");
}

} // namespace igk
