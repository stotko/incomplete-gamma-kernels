import sys

required_version = (3, 9)

if sys.version_info[:2] < required_version:  # pragma: no cover
    msg = "%s requires Python %d.%d+" % (__package__, *required_version)
    raise RuntimeError(msg)

del required_version
del sys


import pathlib

import charonload

PROJECT_ROOT_DIRECTORY = pathlib.Path(__file__).parents[2]

VSCODE_STUBS_DIRECTORY = PROJECT_ROOT_DIRECTORY / "typings"


charonload.module_config["_c_igk"] = charonload.Config(
    pathlib.Path(__file__).parent / "_C",
    stubs_directory=VSCODE_STUBS_DIRECTORY,
    # verbose=True,
)


from _c_igk import density_weights, kde, lop, regularity

from . import sparse
from .kernels import (
    ApproxParams,
    alpha_kernel,
    approximated_alpha_kernel,
    approximated_lop_kernel,
    gaussian_kernel,
    incomplete_gamma_kernel,
    lop_kernel,
    theta_kernel,
)

__all__ = [
    "ApproxParams",
    "alpha_kernel",
    "approximated_alpha_kernel",
    "approximated_lop_kernel",
    "density_weights",
    "gaussian_kernel",
    "incomplete_gamma_kernel",
    "kde",
    "lop",
    "lop_kernel",
    "regularity",
    "sparse",
    "theta_kernel",
]
