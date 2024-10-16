from __future__ import annotations

import pathlib

import cmasher as cmr
import matplotlib.colors
import numpy as np
import polyscope as ps
import scipy
import torch
import trimesh

import incomplete_gamma_kernels as igk


def bounding_box_diagonal(p: torch.Tensor) -> float:
    bb_max, _ = torch.max(p, dim=0)
    bb_min, _ = torch.min(p, dim=0)
    return torch.linalg.norm(bb_max - bb_min).item()


def compute_density_and_cache(
    source_points: torch.Tensor,
    target_path: pathlib.Path,
    cache_path: pathlib.Path,
    device: torch.device,
    h_percent_bb: float,
    density_weights: str,
    force_compute: bool = False,
) -> torch.Tensor:
    if not force_compute and cache_path.exists():
        print(f'Loading from cache "{cache_path}"')

        hat_f = torch.load(cache_path, weights_only=True)

        return hat_f
    else:
        print(f"Computing density")
        mesh = trimesh.load_mesh(target_path)
        p = torch.from_numpy(mesh.vertices).to(dtype=torch.float32, device=device)

        q = source_points

        h = h_percent_bb * (bounding_box_diagonal(p) / 100.0)

        hat_f = igk.kde(
            p,
            q,
            kernel="lop",
            h=h,
            weights=igk.density_weights(p, scheme=density_weights, energy_term="attraction", h=h),
        )

        print(f'Storing to cache "{cache_path}"')
        torch.save(hat_f, cache_path)

        return hat_f


def normalization_constant(variance: float, d: int, p: float) -> float:
    return (
        1.0
        / (2.0 * torch.pi * variance) ** (d / 2.0)
        * scipy.special.gamma((d + 2.0) / 2.0)
        / scipy.special.gamma((d + p) / 2.0)
    )


def estimate_wlop_density_normalization_bias(h_percent_bb: float, target: torch.Tensor) -> float:
    # Term 1
    h = h_percent_bb * (bounding_box_diagonal(target) / 100.0)
    norm_density = 1.0 / (target.size(0) * h**3)

    # Term 2
    norm_kernel = normalization_constant(variance=1.0 / 32.0, d=3, p=2.0)

    # Term 3
    c_theta_2 = normalization_constant(variance=1.0 / 32.0, d=2, p=2.0)
    c_theta_3 = normalization_constant(variance=1.0 / 32.0, d=3, p=2.0)
    c_lop_2 = normalization_constant(variance=1.0 / 32.0, d=2, p=1.0)
    c_lop_3 = normalization_constant(variance=1.0 / 32.0, d=3, p=1.0)
    norm_dimension = (c_lop_3 / c_lop_2) * (c_theta_2 / c_theta_3)

    return norm_density * norm_kernel * norm_dimension


def apply_color(dist: torch.Tensor, vmin: float, vmax: float, cmap: matplotlib.colors.Colormap) -> np.ndarray:
    dist_normalized = dist / torch.mean(dist)
    dist_01 = torch.clip((dist_normalized - vmin) / (vmax - vmin), 0.0, 1.0).cpu().numpy()
    return cmap(dist_01)[:, 0:3]


def make_colormap() -> matplotlib.colors.Colormap:
    N = 256
    h = np.full(N, 0.07)
    s = np.linspace(1.0, 0.0, N)
    v = np.linspace(0.9 - 0.1 * (0.15 / 0.85), 1, N)

    rgb = matplotlib.colors.hsv_to_rgb(np.flip(np.vstack((h, s, v)).T, axis=0))

    return matplotlib.colors.ListedColormap(rgb, name="CustomYellow")


def main() -> None:
    DATA_DIR = pathlib.Path(__file__).parents[1] / "data"
    CACHE_DIR = DATA_DIR / "cache"
    DEVICE = torch.device("cuda")

    CACHE_DIR.mkdir(exist_ok=True)

    mesh = trimesh.load_mesh(DATA_DIR / "bird_image.obj")
    bird_image = torch.from_numpy(mesh.vertices).to(dtype=torch.float32, device=DEVICE)

    h_percent_bb = 3.0

    samples = 1000
    x = torch.linspace(0.0, 1.0, samples, dtype=torch.float32, device=DEVICE)
    y = torch.linspace(0.0, 1.0, samples, dtype=torch.float32, device=DEVICE)
    mx, my = torch.meshgrid(x, y, indexing="xy")
    grid = torch.vstack((mx.ravel(), my.ravel(), torch.zeros_like(mx.ravel()))).T

    density_no_weights = compute_density_and_cache(
        source_points=grid,
        target_path=DATA_DIR / "bird_image.obj",
        cache_path=CACHE_DIR / "density_no_weights.pt",
        device=DEVICE,
        h_percent_bb=h_percent_bb,
        density_weights="none",
    )
    density_wlop = compute_density_and_cache(
        source_points=grid,
        target_path=DATA_DIR / "bird_image.obj",
        cache_path=CACHE_DIR / "density_wlop.pt",
        device=DEVICE,
        h_percent_bb=h_percent_bb,
        density_weights="wlop",
    )
    density_ours_simple = compute_density_and_cache(
        source_points=grid,
        target_path=DATA_DIR / "bird_image.obj",
        cache_path=CACHE_DIR / "density_ours_simple.pt",
        device=DEVICE,
        h_percent_bb=h_percent_bb,
        density_weights="ours_simple",
    )
    density_ours = compute_density_and_cache(
        source_points=grid,
        target_path=DATA_DIR / "bird_image.obj",
        cache_path=CACHE_DIR / "density_ours.pt",
        device=DEVICE,
        h_percent_bb=h_percent_bb,
        density_weights="ours",
    )

    print("--- Mean Densities :")
    print("LOP:          ", torch.mean(density_no_weights).item())
    print("WLOP:         ", torch.mean(density_wlop).item())
    print("Ours (Simple):", torch.mean(density_ours_simple).item())
    print("Ours (Full):  ", torch.mean(density_ours).item())
    print("")

    for density in [density_no_weights, density_wlop, density_ours_simple, density_ours]:
        density /= torch.mean(density)

    print("--- Normalized Standard Deviations :")
    print("WLOP:         ", torch.std(density_wlop).item())
    print("Ours (Simple):", torch.std(density_ours_simple).item())
    print("Ours (Full):  ", torch.std(density_ours).item())
    print("")

    print("--- Correction :")
    print("WLOP Density Normalization Bias:", estimate_wlop_density_normalization_bias(h_percent_bb, bird_image))
    print("")

    # No weights color map
    vmin = 0.0
    vmax = 2.0
    cmap = make_colormap()

    color_lop = apply_color(density_no_weights, vmin, vmax, cmap)

    # WLOP, Ours (Simple), Ours color map
    vmin = 0.8
    vmax = 1.2
    cmap = cmr.get_sub_cmap("coolwarm", 0.0, 1.0)

    color_wlop = apply_color(density_wlop, vmin, vmax, cmap)
    color_ours_simple = apply_color(density_ours_simple, vmin, vmax, cmap)
    color_ours = apply_color(density_ours, vmin, vmax, cmap)

    bird_image = bird_image.cpu().numpy()
    grid = grid.cpu().numpy()

    ps.init()

    render_radius_bird = 0.05 / bird_image.shape[0] ** (1.0 / 3.0)
    render_radius_grid = np.sqrt(2.0 * (1.0 / (samples - 1)) ** 2)

    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(render_radius_grid, is_relative=False)

    ps.set_shadow_blur_iters(6)
    ps.set_shadow_darkness(0.45)

    ps_target = ps.register_point_cloud("Target", bird_image, material="flat")
    ps_target.set_radius(render_radius_bird, relative=False)
    ps_target.set_color((0.0, 0.0, 0.0))

    ps_lop = ps.register_point_cloud("Density: No Weights", grid, material="flat", enabled=False)
    ps_lop.set_radius(render_radius_grid, relative=False)
    ps_lop.add_color_quantity("Density", color_lop, enabled=True)

    ps_lop_wlop = ps.register_point_cloud("Density: WLOP", grid, material="flat", enabled=False)
    ps_lop_wlop.set_radius(render_radius_grid, relative=False)
    ps_lop_wlop.add_color_quantity("Density", color_wlop, enabled=True)

    ps_lop_mean_shift = ps.register_point_cloud("Density: Ours (Simple)", grid, material="flat", enabled=False)
    ps_lop_mean_shift.set_radius(render_radius_grid, relative=False)
    ps_lop_mean_shift.add_color_quantity("Density", color_ours_simple, enabled=True)

    ps_lop_rbf = ps.register_point_cloud("Density: Ours", grid, material="flat", enabled=False)
    ps_lop_rbf.set_radius(render_radius_grid, relative=False)
    ps_lop_rbf.add_color_quantity("Density", color_ours, enabled=True)

    ps.show()


if __name__ == "__main__":
    main()
