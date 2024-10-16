from __future__ import annotations

import pytest
import torch

import incomplete_gamma_kernels as igk

DEVICE = torch.device("cuda")


def _random_points(size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.rand([size, 3], dtype=dtype, device=device)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scheme", ["none", "wlop", "ours_simple", "ours"])
@pytest.mark.parametrize(
    "energy_term",
    [
        "attraction",
        "repulsion",
    ],
)
def test_density_weights_empty_input(dtype: torch.dtype, scheme: str, energy_term: str) -> None:
    p = torch.zeros((0, 3), dtype=dtype, device=DEVICE)

    h = 0.5

    w = igk.density_weights(p, scheme, energy_term, h)

    assert w.shape == (0,)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scheme", ["none", "wlop", "ours_simple", "ours"])
@pytest.mark.parametrize(
    "energy_term",
    [
        "attraction",
        "repulsion",
    ],
)
def test_density_weights_single_point_input(dtype: torch.dtype, scheme: str, energy_term: str) -> None:
    p = torch.tensor([[1.2, 3.4, 5.6]], dtype=dtype, device=DEVICE)

    h = 0.5

    w = igk.density_weights(p, scheme, energy_term, h)

    assert torch.all(w >= 0.0)


@pytest.mark.parametrize("size", [10000, 20000, 30000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scheme", ["none", "wlop", "ours_simple"])  # "ours" is allowed to produce negative weights
@pytest.mark.parametrize(
    "energy_term",
    [
        "attraction",
        "repulsion",
    ],
)
def test_density_weights_nonnegative(size: int, dtype: torch.dtype, scheme: str, energy_term: str) -> None:
    p = _random_points(size, dtype=dtype, device=DEVICE)

    h = 0.5

    w = igk.density_weights(p, scheme, energy_term, h)

    assert torch.all(w >= 0.0)


@pytest.mark.parametrize("size", [10000, 20000, 30000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "energy_term",
    [
        "attraction",
        "repulsion",
    ],
)
def test_density_weights_none(size: int, dtype: torch.dtype, energy_term: str) -> None:
    p = _random_points(size, dtype=dtype, device=DEVICE)

    h = 0.5

    w = igk.density_weights(p, "none", energy_term, h)

    assert w.allclose(torch.ones_like(w))


@pytest.mark.parametrize("size", [10000, 20000, 30000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_density_weights_wlop_attraction(size: int, dtype: torch.dtype) -> None:
    p = _random_points(size, dtype=dtype, device=DEVICE)

    h = 0.5

    w = igk.density_weights(p, "wlop", "attraction", h)

    assert torch.all(w < 1.0)


@pytest.mark.parametrize("size", [10000, 20000, 30000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_density_weights_wlop_repulsion(size: int, dtype: torch.dtype) -> None:
    p = _random_points(size, dtype=dtype, device=DEVICE)

    h = 0.5

    w = igk.density_weights(p, "wlop", "repulsion", h)

    assert torch.all(w > 1.0)


@pytest.mark.parametrize("size", [10000, 20000, 30000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_density_weights_ours_simple_repulsion(size: int, dtype: torch.dtype) -> None:
    p = _random_points(size, dtype=dtype, device=DEVICE)

    h = 0.5

    w = igk.density_weights(p, "ours_simple", "repulsion", h)

    hat_f = igk.kde(p, p, "lop", h)

    assert w.allclose(hat_f)
