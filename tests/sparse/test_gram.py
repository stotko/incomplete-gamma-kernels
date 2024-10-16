from __future__ import annotations

import pytest
import torch

import incomplete_gamma_kernels as igk

DEVICE = torch.device("cuda")


def _random_points(size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.rand([size, 3], dtype=dtype, device=device)


def is_symmetric(x: torch.Tensor) -> bool:
    # Sparsity check is a bit fragile
    if x.is_sparse or x.is_sparse_csr:
        x_t = x.t().to_sparse_csr()
        return (
            (x.crow_indices() == x_t.crow_indices()).all()
            and (x.col_indices() == x_t.col_indices()).all()
            and (x.values() == x_t.values()).all()
        )

    return (x.t() == x).all()


def is_positive_definite(x: torch.Tensor) -> bool:
    # Sparse matrices are tested like the dense versions
    C, info = torch.linalg.cholesky_ex(x.to_dense())
    return info.item() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gram_empty_input(dtype: torch.dtype) -> None:
    p = torch.zeros((0, 3), dtype=dtype, device=DEVICE)

    h = 0.5
    variance = 1 / 32
    trunaction_distance = h * 0.5

    G = igk.sparse.gram(p, "lop", variance, trunaction_distance, True, h)

    assert G.shape == (0, 0)
    assert is_symmetric(G)
    assert is_positive_definite(G)


@pytest.mark.parametrize("size", [10000, 20000, 30000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gram(size: int, dtype: torch.dtype) -> None:
    p = _random_points(size, dtype=dtype, device=DEVICE)

    h = 0.5
    variance = 1 / 32
    trunaction_distance = h * 0.5

    G = igk.sparse.gram(p, "lop", variance, trunaction_distance, True, h)

    print("nnz: ", G._nnz() / G.numel() * 100.0, "%")

    assert is_symmetric(G)
    assert is_positive_definite(G)
