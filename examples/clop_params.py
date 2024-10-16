from __future__ import annotations

import functools

import numpy as np
import scipy.optimize
import torch

import incomplete_gamma_kernels as igk


def lop_approx_wrapper(x, *opt_params, consistent=False):
    if consistent:
        assert len(opt_params) == 4
    else:
        assert len(opt_params) == 5

    return igk.approximated_lop_kernel(
        torch.from_numpy(x),
        params=igk.ApproxParams(
            w_k=torch.tensor([*opt_params[:2], 1.0]),
            sigma_k=torch.tensor(opt_params[2:] if not consistent else [*opt_params[2:], np.sqrt(1.0 / 32.0)]),
        ),
    ).numpy()


def alpha_approx_wrapper(x, s, params):
    return igk.approximated_alpha_kernel(
        torch.from_numpy(x),
        params=igk.ApproxParams(
            w_k=s * params.w_k,
            sigma_k=params.sigma_k,
        ),
    ).numpy()


def fit_ours_no_scale(params: igk.ApproxParams) -> igk.ApproxParams:
    samples = 1000000

    x = torch.linspace(0.0, 1.0, samples)
    y = igk.lop_kernel(x, variance=1.0 / 32.0)

    initial_params = [
        params.w_k[0].item() / params.w_k[2].item(),
        params.w_k[1].item() / params.w_k[2].item(),
        params.sigma_k[0].item(),
        params.sigma_k[1].item(),
        params.sigma_k[2].item(),
    ]

    opt_params, covariance = scipy.optimize.curve_fit(
        f=lop_approx_wrapper,
        xdata=x.numpy(),
        ydata=y.numpy(),
        p0=initial_params,
    )
    opt_params_error = np.sqrt(np.diag(covariance))

    y_fit = torch.from_numpy(lop_approx_wrapper(x.numpy(), *opt_params))

    param_names = ["    w_1", "    w_2", "sigma_1", "sigma_2", "sigma_3"]
    for i in range(opt_params.shape[0]):
        print(f"{param_names[i]} = {opt_params[i]:8.5f} +- {opt_params_error[i]:7.5f}")

    print("Integrated error:", torch.trapezoid(torch.abs(y - y_fit), x).item())
    print("   Max abs error:", torch.max(torch.abs(y - y_fit)).item())
    print("   Max rel error:", torch.max(torch.abs(y - y_fit) / y).item())

    return igk.ApproxParams(
        w_k=torch.tensor([*opt_params[:2], 1.0]),
        sigma_k=torch.tensor(opt_params[2:]),
    )


def fit_ours_consistent(params: igk.ApproxParams) -> igk.ApproxParams:
    samples = 1000000

    x = torch.linspace(0, 1, samples)
    y = igk.lop_kernel(x, variance=1.0 / 32.0)

    initial_params = [
        params.w_k[0].item() / params.w_k[2].item(),
        params.w_k[1].item() / params.w_k[2].item(),
        params.sigma_k[0].item(),
        params.sigma_k[1].item(),
    ]

    lop_approx_wrapper_ = functools.partial(lop_approx_wrapper, consistent=True)

    opt_params, covariance = scipy.optimize.curve_fit(
        f=lop_approx_wrapper_,
        xdata=x.numpy(),
        ydata=y.numpy(),
        p0=initial_params,
    )
    opt_params_error = np.sqrt(np.diag(covariance))

    y_fit = torch.from_numpy(lop_approx_wrapper_(x.numpy(), *opt_params))

    param_names = ["    w_1", "    w_2", "sigma_1", "sigma_2"]
    for i in range(opt_params.shape[0]):
        print(f"{param_names[i]} = {opt_params[i]:8.5f} +- {opt_params_error[i]:7.5f}")

    print("Integrated error:", torch.trapezoid(torch.abs(y - y_fit), x).item())
    print("   Max abs error:", torch.max(torch.abs(y - y_fit)).item())
    print("   Max rel error:", torch.max(torch.abs(y - y_fit) / y).item())

    return igk.ApproxParams(
        w_k=torch.tensor([*opt_params[:2], 1.0]),
        sigma_k=torch.tensor([*opt_params[2:], np.sqrt(1.0 / 32.0)]),
    )


def fit_scale(params: igk.ApproxParams) -> igk.ApproxParams:
    samples = 1000000

    x = torch.linspace(0.01, 1, samples)
    y = igk.alpha_kernel(x)

    alpha_approx_wrapper_ = functools.partial(alpha_approx_wrapper, params=params)

    initial_params = [1]
    opt_params, covariance = scipy.optimize.curve_fit(
        f=alpha_approx_wrapper_, xdata=x.numpy(), ydata=y.numpy(), p0=initial_params
    )
    opt_params_error = np.sqrt(np.diag(covariance))

    y_fit = torch.from_numpy(alpha_approx_wrapper_(x.numpy(), *opt_params))

    param_names = ["      s"]
    for i in range(opt_params.shape[0]):
        print(f"{param_names[i]} = {opt_params[i]:8.5f} +- {opt_params_error[i]:7.5f}")

    print("Integrated error:", torch.trapezoid(torch.abs(y - y_fit), x).item())
    print("   Max abs error:", torch.max(torch.abs(y - y_fit)).item())
    print("   Max rel error:", torch.max(torch.abs(y - y_fit) / y).item())

    return igk.ApproxParams(
        w_k=opt_params[0] * params.w_k,
        sigma_k=params.sigma_k,
    )


def std_lop_approx(params: igk.ApproxParams, d: int) -> float:
    return torch.sqrt(
        torch.sum(params.w_k * (params.sigma_k ** (d + 4))) / torch.sum(params.w_k * (params.sigma_k ** (d + 2)))
    ).item()


def std_lop_approx_d_inf(params: igk.ApproxParams) -> float:
    sigma_max, _ = torch.max(params.sigma_k, dim=0)
    return torch.sqrt(sigma_max**2).item()


def std_lop(d: int) -> float:
    return torch.sqrt(torch.tensor((d + 1) / (d + 2) * (1.0 / 32.0))).item()


def std_lop_d_inf() -> float:
    return torch.sqrt(torch.tensor(1.0 / 32.0)).item()


def main() -> None:
    params = igk.ApproxParams.clop()
    print("")
    print("--- CLOP :")
    print(params)
    print("")

    params = fit_ours_no_scale(params)
    params = fit_scale(params)

    print("")
    print("--- Ours :")
    print(params)
    print("")

    params = igk.ApproxParams.clop()
    params = fit_ours_consistent(params)
    params = fit_scale(params)

    print("")
    print("--- Ours (Consistent) :")
    print(params)
    print("")

    print("")
    print("--- Ratio of Standard Deviations :")

    print("                   d = 1              d = 2              d = 3              d = 4              d -> inf")

    print("CLOP:             ", end=" ")
    for d in range(1, 5):
        print(std_lop_approx(igk.ApproxParams.clop(), d) / std_lop(d), end=" ")
    print(std_lop_approx_d_inf(igk.ApproxParams.clop()) / std_lop_d_inf(), end=" ")
    print("")

    print("Ours:             ", end=" ")
    for d in range(1, 5):
        print(std_lop_approx(igk.ApproxParams.ours(), d) / std_lop(d), end=" ")
    print(std_lop_approx_d_inf(igk.ApproxParams.ours()) / std_lop_d_inf(), end=" ")
    print("")

    print("Ours (Consistent):", end=" ")
    for d in range(1, 5):
        print(std_lop_approx(igk.ApproxParams.ours_consistent(), d) / std_lop(d), end=" ")
    print(std_lop_approx_d_inf(igk.ApproxParams.ours_consistent()) / std_lop_d_inf(), end=" ")
    print("")


if __name__ == "__main__":
    main()
