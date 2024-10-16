#include <torch/python.h>

// Not included by torch/python.h but required for std::function
#include <pybind11/functional.h>

#include <igk/density_weights.h>
#include <igk/lop.h>
#include <igk/mean_shift.h>
#include <igk/regularity.h>
#include <igk/sparse/gram.h>
#include <igk/sparse/linalg/cg.h>

using namespace pybind11::literals;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    auto m_sparse = m.def_submodule("sparse", "Sparse extensions");
    m_sparse.def("gram",
                 &igk::sparse::gram,
                 "p"_a,
                 "kernel"_a,
                 "variance"_a,
                 "r_truncation"_a = 0.0f,
                 "use_h"_a = false,
                 "h"_a = 1.0f,
                 R"(
        Compute the truncated Gram matrix `G_ij = K(p[i, :], p[j, :])`.

        Note
        ----
            Not differentiable.

        Note
        ----
            CUDA backend only.

        Parameters
        ----------
        p
            3-dimensional points matrix of size N x 3.
        kernel
            Kernel `K` to use for inner product. Options: `lop`.
        variance
            Variance of the kernel `K`.
        r_truncation
            Minimum distance to keep `G_ij`, i.e. if `|p[i, :] - p[j, :]| < r_truncation`, it is implicitly set to `0`.
        use_h
            Whether to use the windows size `h` and to apply the Mean Shift normalization factor `1 / (N * h^3)`.
        h
            Window size applied when the kernel `K` should be used in the Mean Shift context.

        Returns
        -------
        Sparse Gram matrix `G` of size N x N.
    )");

    auto m_sparse_linalg = m_sparse.def_submodule("linalg", "Sparse linalg extensions");
    m_sparse_linalg.def("cg",
                        &igk::sparse::linalg::cg,
                        "A"_a,
                        "b"_a,
                        "x0"_a = py::none(),
                        py::kw_only(),
                        "maxiter"_a = py::none(),
                        "M"_a = py::none(),
                        "callback"_a = py::none(),
                        "rtol"_a = 1e-5f,
                        "atol"_a = 0.0f,
                        R"(
        Solve the sparse equation system using Conjugate Gradient.

        Note
        ----
            Not differentiable.

        Note
        ----
            CUDA backend only.

        Parameters
        ----------
        A
            Sparse matrix of size N x N
        b
            Dense vector of size N
        x0
            Optional initial guess given as a dense vector of size N.
        maxiter
            Optional number of maximum iterations, if None then 10 * N will be used
        M
            Optional sparse matrix M for preconditioning, must be None and will be ignored since preconditioning is not implemented
        callback
            Optional callback which is called after each iteration k with the current solution xk, i.e. callback(xk)
        rtol
            Relative tolerance to reach convergence.
        atol
            Absolute tolerance to reach convergence.

        Returns
        -------
        x
            Solution as a dense vector of size N
        info
            0 if converged, number of iterations otherwise
    )");

    m.def("density_weights",
          &igk::density_weights,
          "p"_a,
          "scheme"_a,
          "energy_term"_a,
          "h"_a,
          R"(
        Compute density weights for the points.

        Note
        ----
            Not differentiable.

        Note
        ----
            CUDA backend only.

        Parameters
        ----------
        p
            3-dimensional points matrix of size N x 3.
        scheme
            Scheme and kernel `K` to use for computing the weights. Options: `none`, `wlop`, `ours_simple`, `ours`.
        energy_term
            Energy term `E` in LOP for which to compute the weights. Options: `attraction`, `repulsion`.
        h
            Window size applied to the kernel `K`.

        Returns
        -------
        Vector with density weights of size N.
    )");

    m.def("kde",
          &igk::kde,
          "p"_a,
          "q"_a,
          "kernel"_a,
          "h"_a,
          "weights"_a = py::none(),
          R"(
        Compute kernel density estimate of source points.

        Note
        ----
            Not differentiable.

        Note
        ----
            CUDA backend only.

        Parameters
        ----------
        p
            3-dimensional target points matrix of size N x 3.
        q
            3-dimensional source points matrix of size M x 3.
        kernel
            Kernel `K` to use for density estimate. Options: `lop`.
        h
            Window size applied to the kernel `K`.
        weights
            Optional weights for the target points `p` of size N.

        Returns
        -------
        Vector with kernel density estimate of size M.
    )");

    m.def("regularity",
          &igk::regularity,
          "p"_a,
          R"(
        Compute point regularity, i.e. standard deviation of nearest neighbor distances.

        Note
        ----
            Not differentiable.

        Note
        ----
            CUDA backend only.

        Parameters
        ----------
        p
            3-dimensional target points matrix of size N x 3.

        Returns
        -------
        Regularity value of points
    )");

    m.def("lop",
          &igk::lop,
          "p"_a,
          "q0"_a,
          py::kw_only(),
          "attraction_kernel"_a,
          "density_weight_scheme"_a,
          "repulsion_function"_a,
          "mu"_a,
          "h_percent_bb"_a,
          "iterations"_a,
          "attraction_p_norm"_a = py::none(),
          "attraction_approx_params"_a = py::none(),
          "verbose"_a = false,
          R"(
        Perform LOP on the points.

        Note
        ----
            Not differentiable.

        Note
        ----
            CUDA backend only.

        Parameters
        ----------
        p
            3-dimensional target points matrix of size N x 3.
        q0
            3-dimensional source points matrix of size M x 3.
        attraction_kernel
            Kernel `alpha` to use for attraction force. Options: `exact`, `generalized`, `approximated`.
        density_weight_scheme
            Scheme and kernel `K` to use for density weights. Options: `none`, `wlop`, `ours`, `ours_full`.
        repulsion_function
            Eta function to use for repulsion force. Options: `original` (eta(r) = 1/(3 r^3)), `wlop` (eta(r) = -r).
        mu
            Balance between attraction and repulsion term.
        h_percent_bb
            Window size for kernel `K` specified in \"percent bounding box diagonal (% BB)\" of `p`.
        iterations
            Number of projection iterations.
        attraction_p_norm
            Optional p-norm for generalized alpha kernels. Falls back to default alpha kernel (p = 1) if not specified.
        attraction_approx_params
            Parameter set `w_k` and `sigma_k` to use for approximated alpha kernels. Options: `clop`, `ours`, `ours_consistent`.
        verbose
            Whether to print intermediate progress.

        Returns
        -------
        3-dimensional projected points matrix of size M x 3.
    )");
}
