from __future__ import annotations

from dataclasses import dataclass

import scipy.special
import torch


def incomplete_gamma_kernel(
    x: torch.Tensor,
    *,
    p: float,
    variance: float,
    dim: int | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if dim is None:
        d = 1
        r = torch.abs(x.to(dtype))
    else:
        d = x.shape[dim]
        r = torch.linalg.vector_norm(x, dim=dim, dtype=dtype)
    c = 1.0 / (2.0 * torch.pi * variance) ** (d / 2.0) * (d / 2.0) * scipy.special.beta(d / 2.0, p / 2.0)

    return c * torch.special.gammaincc(torch.tensor([p / 2.0], device=x.device), 0.5 * (r * r) / variance)


def gaussian_kernel(
    x: torch.Tensor,
    *,
    variance: float,
    dim: int | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if dim is None:
        d = 1
        r = torch.abs(x.to(dtype))
    else:
        d = x.shape[dim]
        r = torch.linalg.vector_norm(x, dim=dim, dtype=dtype)
    c = 1.0 / (2.0 * torch.pi * variance) ** (d / 2.0)

    return c * torch.exp(-0.5 * (r * r) / variance)


def lop_kernel(
    x: torch.Tensor,
    *,
    variance: float,
    dim: int | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if dim is None:
        d = 1
        r = torch.abs(x.to(dtype))
    else:
        d = x.shape[dim]
        r = torch.linalg.vector_norm(x, dim=dim, dtype=dtype)
    c = (
        1.0
        / (2.0 * torch.pi * variance) ** (d / 2.0)
        * scipy.special.gamma((d + 2.0) / 2.0)
        / scipy.special.gamma((d + 1.0) / 2.0)
        * torch.sqrt(torch.tensor(torch.pi)).item()
    )

    return c * torch.erfc(torch.sqrt(torch.tensor(0.5 / variance)).item() * r)


def theta_kernel(
    x: torch.Tensor,
    *,
    dim: int | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if dim is None:
        r = torch.abs(x.to(dtype))
    else:
        r = torch.linalg.vector_norm(x, dim=dim, dtype=dtype)

    return torch.exp(-(r * r) / (1.0 / 4.0) ** 2)


def alpha_kernel(
    x: torch.Tensor,
    *,
    dim: int | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if dim is None:
        r = torch.abs(x.to(dtype))
    else:
        r = torch.linalg.vector_norm(x, dim=dim, dtype=dtype)

    return theta_kernel(x, dim=dim, dtype=dtype) / r


@dataclass()
class ApproxParams:
    w_k: torch.Tensor
    sigma_k: torch.Tensor

    @staticmethod
    def clop() -> ApproxParams:
        return ApproxParams(
            w_k=torch.tensor([97.761, 29.886, 11.453]),
            sigma_k=torch.tensor([0.01010, 0.03287, 0.11772]),
        )

    @staticmethod
    def ours() -> ApproxParams:
        return ApproxParams(
            w_k=torch.tensor([61.50926, 11.93220, 5.06884]),
            sigma_k=torch.tensor([0.02102, 0.07289, 0.15700]),
        )

    @staticmethod
    def ours_consistent() -> ApproxParams:
        return ApproxParams(
            w_k=torch.tensor([46.40851, 9.63450, 2.67366]),
            sigma_k=torch.tensor([0.03118, 0.10582, torch.sqrt(torch.tensor(1.0 / 32.0)).item()]),
        )


def approximated_lop_kernel(
    x: torch.Tensor,
    *,
    params: ApproxParams,
    dim: int | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if dim is None:
        d = 1
        r = torch.abs(x.to(dtype))
    else:
        d = x.shape[dim]
        r = torch.linalg.vector_norm(x, dim=dim, dtype=dtype)

    values = torch.zeros_like(r)
    denom = 0.0
    for i in range(len(params.w_k)):
        values += params.sigma_k[i] ** 2 * params.w_k[i] * torch.exp(-0.5 * (r * r) / params.sigma_k[i] ** 2)
        denom += params.sigma_k[i] ** 2 * params.w_k[i] * (2.0 * torch.pi * params.sigma_k[i] ** 2) ** (d / 2.0)

    return values / denom


def approximated_alpha_kernel(
    x: torch.Tensor,
    *,
    params: ApproxParams,
    dim: int | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if dim is None:
        r = torch.abs(x.to(dtype))
    else:
        r = torch.linalg.vector_norm(x, dim=dim, dtype=dtype)

    values = torch.zeros_like(r)
    for i in range(len(params.w_k)):
        values += params.w_k[i] * torch.exp(-0.5 * (r * r) / params.sigma_k[i] ** 2)

    return values
