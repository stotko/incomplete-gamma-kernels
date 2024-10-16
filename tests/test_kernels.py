from __future__ import annotations

import numpy as np
import pytest
import torch

import incomplete_gamma_kernels as igk

DEVICE = torch.device("cuda")


@pytest.mark.parametrize("dim", [None, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("sigma", [0.01, 0.1, 1, 10])
def test_gaussian_kernel(dim: int | None, sigma: float) -> None:
    d = dim if dim is not None else 1

    num_points = int(np.floor(1e8 ** (1.0 / d)))
    x_max = 6.0 * sigma
    dx = 2.0 * x_max / (num_points - 1.0)

    x_i = torch.linspace(-x_max, x_max, num_points, device=DEVICE)
    x = torch.stack(torch.meshgrid(d * [x_i], indexing="ij"), dim=-1)
    if dim is None:
        x.squeeze_()

    y = igk.gaussian_kernel(x, variance=sigma * sigma, dim=-1 if dim is not None else None)

    area = y
    for _ in range(len(y.shape)):
        area = torch.trapezoid(area, dx=dx, dim=-1)

    assert torch.allclose(area, torch.tensor([1.0], device=DEVICE), atol=10 ** (-7 + d))


@pytest.mark.parametrize("dim", [None, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("sigma", [0.01, 0.1, 1, 10])
def test_lop_kernel(dim: int | None, sigma: float) -> None:
    d = dim if dim is not None else 1

    num_points = int(np.floor(1e8 ** (1.0 / d)))
    x_max = 6.0 * sigma
    dx = 2.0 * x_max / (num_points - 1.0)

    x_i = torch.linspace(-x_max, x_max, num_points, device=DEVICE)
    x = torch.stack(torch.meshgrid(d * [x_i], indexing="ij"), dim=-1)
    if dim is None:
        x.squeeze_()

    y = igk.lop_kernel(x, variance=sigma * sigma, dim=-1 if dim is not None else None)

    area = y
    for _ in range(len(y.shape)):
        area = torch.trapezoid(area, dx=dx, dim=-1)

    assert torch.allclose(area, torch.tensor([1.0], device=DEVICE), atol=10 ** (-7 + d))


@pytest.mark.parametrize("dim", [None, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("sigma", [0.01, 0.1, 1, 10])
@pytest.mark.parametrize("p", [0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5])
def test_incomplete_gamma_kernel(dim: int | None, sigma: float, p: float) -> None:
    d = dim if dim is not None else 1

    num_points = int(np.floor(1e8 ** (1.0 / d)))
    x_max = 6.0 * sigma
    dx = 2.0 * x_max / (num_points - 1.0)

    x_i = torch.linspace(-x_max, x_max, num_points, device=DEVICE)
    x = torch.stack(torch.meshgrid(d * [x_i], indexing="ij"), dim=-1)
    if dim is None:
        x.squeeze_()

    y = igk.incomplete_gamma_kernel(x, p=p, variance=sigma * sigma, dim=-1 if dim is not None else None)

    area = y
    for _ in range(len(y.shape)):
        area = torch.trapezoid(area, dx=dx, dim=-1)

    assert torch.allclose(area, torch.tensor([1.0], device=DEVICE), atol=10 ** (-7 + d))


@pytest.mark.parametrize("dim", [None, 1, 2, 3, 4, 5])
@pytest.mark.parametrize(
    "params",
    [igk.ApproxParams.clop(), igk.ApproxParams.ours(), igk.ApproxParams.ours_consistent()],
)
def test_approximated_lop_kernel(dim: int | None, params: igk.ApproxParams) -> None:
    d = dim if dim is not None else 1
    sigma = np.sqrt(1.0 / 32.0)

    num_points = int(np.floor(1e8 ** (1.0 / d)))
    x_max = 6.0 * sigma
    dx = 2.0 * x_max / (num_points - 1.0)

    x_i = torch.linspace(-x_max, x_max, num_points, device=DEVICE)
    x = torch.stack(torch.meshgrid(d * [x_i], indexing="ij"), dim=-1)
    if dim is None:
        x.squeeze_()

    y = igk.approximated_lop_kernel(x, params=params, dim=-1 if dim is not None else None)

    area = y
    for _ in range(len(y.shape)):
        area = torch.trapezoid(area, dx=dx, dim=-1)

    assert torch.allclose(area, torch.tensor([1.0], device=DEVICE), atol=10 ** (-7 + d))
