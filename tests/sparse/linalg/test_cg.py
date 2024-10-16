from __future__ import annotations

import functools

import pytest
import torch

import incomplete_gamma_kernels as igk

DEVICE = torch.device("cuda")


def _random_sparse_psd(size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    a = torch.rand([size, size], dtype=dtype, device=device)

    sym_a = 0.5 * (a + a.t())
    del a

    sym_a[sym_a < 0.9] = 0.0

    psd = size * torch.eye(size, dtype=dtype, device=device) + sym_a
    del sym_a

    return psd.to_sparse_coo()


@pytest.mark.parametrize("size", [10000, 20000, 30000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cg_rtol(size: int, dtype: torch.dtype) -> None:
    A = _random_sparse_psd(size, dtype=dtype, device=DEVICE)
    b = torch.rand([size], dtype=dtype, device=DEVICE)
    x0 = torch.rand([size], dtype=dtype, device=DEVICE)

    assert torch.linalg.norm(A @ x0 - b) >= 1e-1 * size

    x, info = igk.sparse.linalg.cg(A, b, x0, rtol=1e-7)

    assert torch.linalg.norm(A @ x - b) < 1e-5 * size
    assert info == 0


@pytest.mark.parametrize("size", [10000, 20000, 30000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cg_atol(size: int, dtype: torch.dtype) -> None:
    A = _random_sparse_psd(size, dtype=dtype, device=DEVICE)
    b = torch.rand([size], dtype=dtype, device=DEVICE)
    x0 = torch.rand([size], dtype=dtype, device=DEVICE)

    assert torch.linalg.norm(A @ x0 - b) >= 1e-1 * size

    x, info = igk.sparse.linalg.cg(A, b, x0, rtol=0.0, atol=1e-3)

    assert torch.linalg.norm(A @ x - b) < 1e-3
    assert info == 0


@pytest.mark.parametrize("size", [10000, 20000, 30000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cg_without_x0(size: int, dtype: torch.dtype) -> None:
    A = _random_sparse_psd(size, dtype=dtype, device=DEVICE)
    b = torch.rand([size], dtype=dtype, device=DEVICE)

    x, info = igk.sparse.linalg.cg(A, b, rtol=1e-7)

    assert torch.linalg.norm(A @ x - b) < 1e-5 * size
    assert info == 0


def test_cg_unconverged() -> None:
    size = 10000
    dtype = torch.float32
    A = _random_sparse_psd(size, dtype=dtype, device=DEVICE)
    b = torch.rand([size], dtype=dtype, device=DEVICE)
    x0 = torch.rand([size], dtype=dtype, device=DEVICE)

    assert torch.linalg.norm(A @ x0 - b) >= 1e-1 * size

    x, info = igk.sparse.linalg.cg(A, b, x0, maxiter=2)

    assert torch.linalg.norm(A @ x - b) >= 1e-5 * size
    assert info == 2


def test_cg_preconditioning() -> None:
    size = 10000
    dtype = torch.float32
    A = _random_sparse_psd(size, dtype=dtype, device=DEVICE)
    b = torch.rand([size], dtype=dtype, device=DEVICE)
    x0 = torch.rand([size], dtype=dtype, device=DEVICE)

    with pytest.raises(ValueError) as exc_info:
        igk.sparse.linalg.cg(A, b, x0, M=A)

    assert exc_info.type is ValueError


class CallbackError(Exception):
    pass


def cg_callback(xk: torch.Tensor, size: int) -> None:
    assert xk.shape[0] == size

    # Raise custom exception to validate that the callback is actually called
    raise CallbackError


def test_cg_callback() -> None:
    size = 10000
    dtype = torch.float32
    A = _random_sparse_psd(size, dtype=dtype, device=DEVICE)
    b = torch.rand([size], dtype=dtype, device=DEVICE)
    x0 = torch.rand([size], dtype=dtype, device=DEVICE)

    with pytest.raises(CallbackError) as exc_info:
        igk.sparse.linalg.cg(A, b, x0, callback=functools.partial(cg_callback, size=size))

    assert exc_info.type is CallbackError


def _random_gram_matrix(size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    p = torch.rand([size, 3], dtype=dtype, device=device)

    h = 0.5
    variance = 1 / 32
    trunaction_distance = h * 0.5

    return igk.sparse.gram(p, "lop", variance, trunaction_distance, True, h)


@pytest.mark.parametrize("size", [10000, 20000, 30000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cg_gram(size: int, dtype: torch.dtype) -> None:
    A = _random_gram_matrix(size, dtype=dtype, device=DEVICE)
    b = torch.ones([size], dtype=dtype, device=DEVICE)
    x0 = torch.ones([size], dtype=dtype, device=DEVICE)

    x, info = igk.sparse.linalg.cg(A, b, x0)

    assert torch.linalg.norm(A @ x - b) < 1e-5 * size
    assert info == 0
