from __future__ import annotations

import pytest
import torch

import incomplete_gamma_kernels as igk

DEVICE = torch.device("cuda")


def _random_points(size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.rand([size, 3], dtype=dtype, device=device)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_igk_empty_input(dtype: torch.dtype) -> None:
    p = torch.zeros((0, 3), dtype=dtype, device=DEVICE)

    regularity = igk.regularity(p)

    assert regularity == 0.0


@pytest.mark.parametrize("size", [10000, 20000, 30000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_regularity_nonnegative(size: int, dtype: torch.dtype) -> None:
    p = _random_points(size, dtype=dtype, device=DEVICE)

    regularity = igk.regularity(p)

    assert regularity >= 0.0
