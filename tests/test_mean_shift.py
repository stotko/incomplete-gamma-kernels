from __future__ import annotations

import pytest
import torch

import incomplete_gamma_kernels as igk

DEVICE = torch.device("cuda")


def _random_points(size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.rand([size, 3], dtype=dtype, device=device)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mean_shift_kde_empty_target(dtype: torch.dtype) -> None:
    p = torch.zeros((0, 3), dtype=dtype, device=DEVICE)
    q = _random_points(10000, dtype=dtype, device=DEVICE)

    h = 0.5

    hat_f = igk.kde(p, q, "lop", h)

    assert hat_f.allclose(torch.zeros_like(hat_f))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mean_shift_kde_empty_source(dtype: torch.dtype) -> None:
    p = _random_points(10000, dtype=dtype, device=DEVICE)
    q = torch.zeros((0, 3), dtype=dtype, device=DEVICE)

    h = 0.5

    hat_f = igk.kde(p, q, "lop", h)

    assert hat_f.shape == (0,)


@pytest.mark.parametrize("size", [10000, 20000, 30000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mean_shift_kde_nonnegative(size: int, dtype: torch.dtype) -> None:
    p = _random_points(size, dtype=dtype, device=DEVICE)
    q = _random_points(size, dtype=dtype, device=DEVICE)

    h = 0.5

    hat_f = igk.kde(p, q, "lop", h)

    assert torch.all(hat_f >= 0.0)


@pytest.mark.parametrize("size", [10000, 20000, 30000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mean_shift_kde_zero_weights(size: int, dtype: torch.dtype) -> None:
    p = _random_points(size, dtype=dtype, device=DEVICE)
    q = _random_points(size, dtype=dtype, device=DEVICE)
    weights = torch.zeros([size], dtype=dtype, device=DEVICE)

    h = 0.5

    hat_f = igk.kde(p, q, "lop", h, weights)

    assert hat_f.allclose(torch.zeros_like(hat_f))


@pytest.mark.parametrize("size", [10000, 20000, 30000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [0.5, 1.0, 1.5])
def test_mean_shift_kde_constant_weights(size: int, dtype: torch.dtype, scale: float) -> None:
    p = _random_points(size, dtype=dtype, device=DEVICE)
    q = _random_points(size, dtype=dtype, device=DEVICE)
    weights = torch.full([size], scale, dtype=dtype, device=DEVICE)

    h = 0.5

    hat_f_ones = igk.kde(p, q, "lop", h)

    hat_f = igk.kde(p, q, "lop", h, weights)

    assert hat_f.allclose(scale * hat_f_ones)
