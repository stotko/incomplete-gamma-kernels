from __future__ import annotations

import pytest
import torch

import incomplete_gamma_kernels as igk

DEVICE = torch.device("cuda")


def _random_points(size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.rand([size, 3], dtype=dtype, device=device)


@pytest.mark.parametrize("size", [10000, 20000, 30000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("attraction_kernel", ["exact", "approximated", "generalized"])
@pytest.mark.parametrize("density_weight_scheme", ["wlop", "ours_simple", "ours"])
@pytest.mark.parametrize("repulsion_function", ["original", "wlop"])
def test_lop_empty_target(
    size: int,
    dtype: torch.dtype,
    attraction_kernel: str,
    density_weight_scheme: str,
    repulsion_function: str,
) -> None:
    p = torch.zeros((0, 3), dtype=dtype, device=DEVICE)
    q0 = _random_points(size=size, dtype=dtype, device=DEVICE)

    mu = 0.1
    h_percent_bb = 2.0
    iterations = 10
    attraction_p_norm = 1.5
    attraction_approx_params = "clop"

    q = igk.lop(
        p,
        q0,
        attraction_kernel=attraction_kernel,
        density_weight_scheme=density_weight_scheme,
        repulsion_function=repulsion_function,
        mu=mu,
        h_percent_bb=h_percent_bb,
        iterations=iterations,
        attraction_p_norm=attraction_p_norm,
        attraction_approx_params=attraction_approx_params,
    )

    assert torch.allclose(q, q0)


@pytest.mark.parametrize("size", [10000, 20000, 30000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("attraction_kernel", ["exact", "approximated", "generalized"])
@pytest.mark.parametrize("density_weight_scheme", ["wlop", "ours_simple", "ours"])
@pytest.mark.parametrize("repulsion_function", ["original", "wlop"])
def test_lop_empty_source(
    size: int,
    dtype: torch.dtype,
    attraction_kernel: str,
    density_weight_scheme: str,
    repulsion_function: str,
) -> None:
    p = _random_points(size=size, dtype=dtype, device=DEVICE)
    q0 = torch.zeros((0, 3), dtype=dtype, device=DEVICE)

    mu = 0.1
    h_percent_bb = 2.0
    iterations = 10
    attraction_p_norm = 1.5
    attraction_approx_params = "clop"

    q = igk.lop(
        p,
        q0,
        attraction_kernel=attraction_kernel,
        density_weight_scheme=density_weight_scheme,
        repulsion_function=repulsion_function,
        mu=mu,
        h_percent_bb=h_percent_bb,
        iterations=iterations,
        attraction_p_norm=attraction_p_norm,
        attraction_approx_params=attraction_approx_params,
    )

    assert torch.allclose(q, q0)


@pytest.mark.parametrize("size", [10000, 20000, 30000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("attraction_kernel", ["exact", "approximated", "generalized"])
@pytest.mark.parametrize("density_weight_scheme", ["wlop", "ours_simple", "ours"])
@pytest.mark.parametrize("repulsion_function", ["original", "wlop"])
def test_lop_single_point_target(
    size: int,
    dtype: torch.dtype,
    attraction_kernel: str,
    density_weight_scheme: str,
    repulsion_function: str,
) -> None:
    p = torch.tensor([[1.2, 3.4, 5.6]], dtype=dtype, device=DEVICE)
    q0 = _random_points(size=size, dtype=dtype, device=DEVICE)

    mu = 0.0
    h_percent_bb = 2.0
    iterations = 10
    attraction_p_norm = 1.5
    attraction_approx_params = "clop"

    q = igk.lop(
        p,
        q0,
        attraction_kernel=attraction_kernel,
        density_weight_scheme=density_weight_scheme,
        repulsion_function=repulsion_function,
        mu=mu,
        h_percent_bb=h_percent_bb,
        iterations=iterations,
        attraction_p_norm=attraction_p_norm,
        attraction_approx_params=attraction_approx_params,
    )

    assert torch.allclose(q, p)


@pytest.mark.parametrize("size", [10000, 20000, 30000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("attraction_kernel", ["exact", "approximated", "generalized"])
@pytest.mark.parametrize("density_weight_scheme", ["wlop", "ours_simple", "ours"])
@pytest.mark.parametrize("repulsion_function", ["original", "wlop"])
def test_lop_large_translation(
    size: int,
    dtype: torch.dtype,
    attraction_kernel: str,
    density_weight_scheme: str,
    repulsion_function: str,
) -> None:
    plane_depth = 42.0

    p = _random_points(size=size, dtype=dtype, device=DEVICE)
    p[:, 2] = plane_depth

    q0 = _random_points(size=size, dtype=dtype, device=DEVICE)

    mu = 0.4
    h_percent_bb = 2.0
    iterations = 10
    attraction_p_norm = 1.5
    attraction_approx_params = "clop"

    q = igk.lop(
        p,
        q0,
        attraction_kernel=attraction_kernel,
        density_weight_scheme=density_weight_scheme,
        repulsion_function=repulsion_function,
        mu=mu,
        h_percent_bb=h_percent_bb,
        iterations=iterations,
        attraction_p_norm=attraction_p_norm,
        attraction_approx_params=attraction_approx_params,
    )

    assert torch.allclose(q[:, 2], torch.tensor(plane_depth, dtype=dtype, device=DEVICE))
