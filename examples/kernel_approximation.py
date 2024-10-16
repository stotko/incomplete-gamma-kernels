from __future__ import annotations

import pathlib

import polyscope as ps
import torch
import trimesh

import incomplete_gamma_kernels as igk


def project_and_cache(
    source_path: pathlib.Path,
    target_path: pathlib.Path,
    cache_path: pathlib.Path,
    device: torch.device,
    attraction_kernel: str,
    attraction_approx_params: str,
    force_compute: bool = False,
) -> torch.Tensor:
    if not force_compute and cache_path.exists():
        print(f'Loading from cache "{cache_path}"')

        mesh = trimesh.load_mesh(cache_path)
        q = torch.from_numpy(mesh.vertices).to(dtype=torch.float32, device=device)

        return q
    else:
        print(f"Projecting")
        mesh = trimesh.load_mesh(target_path)
        p = torch.from_numpy(mesh.vertices).to(dtype=torch.float32, device=device)

        mesh = trimesh.load_mesh(source_path)
        q0 = torch.from_numpy(mesh.vertices).to(dtype=torch.float32, device=device)

        mu = 0.4
        h_percent_bb = 25.0
        iterations = 100

        q = igk.lop(
            p,
            q0,
            attraction_kernel=attraction_kernel,
            density_weight_scheme="wlop",
            repulsion_function="wlop",
            mu=mu,
            h_percent_bb=h_percent_bb,
            iterations=iterations,
            attraction_approx_params=attraction_approx_params,
            verbose=False,
        )

        print(f'Storing to cache "{cache_path}"')
        mesh_new = trimesh.PointCloud(q.cpu().numpy())
        mesh_new.export(cache_path)

        return q


def find_nearest(y: torch.Tensor, x: torch.Tensor, x0: float) -> float:
    _, index = torch.min(torch.abs(x - x0), dim=0)
    return y[index.item()].item()


def main() -> None:
    DATA_DIR = pathlib.Path(__file__).parents[1] / "data"
    CACHE_DIR = DATA_DIR / "cache"
    DEVICE = torch.device("cuda")

    CACHE_DIR.mkdir(exist_ok=True)

    samples = 1000000

    x = torch.linspace(-1.0, 1.0, samples, device=DEVICE)
    y_lop = igk.lop_kernel(x, variance=1.0 / 32.0)

    N = 10000
    error_clop_l1 = torch.zeros(N)
    error_ours_l1 = torch.zeros(N)
    b = torch.linspace(0.6, 1.4, N)
    for i in range(N):
        y_approx_clop_b = igk.approximated_lop_kernel(
            x,
            params=igk.ApproxParams(
                w_k=igk.ApproxParams.clop().w_k,
                sigma_k=igk.ApproxParams.clop().sigma_k / b[i],
            ),
        )
        y_approx_ours_b = igk.approximated_lop_kernel(
            x,
            params=igk.ApproxParams(
                w_k=igk.ApproxParams.ours().w_k,
                sigma_k=igk.ApproxParams.ours().sigma_k / b[i],
            ),
        )

        error_clop_l1[i] = torch.trapezoid(torch.abs(y_approx_clop_b - y_lop), x)
        error_ours_l1[i] = torch.trapezoid(torch.abs(y_approx_ours_b - y_lop), x)

    b_opt_clop = b[torch.argmin(error_clop_l1).item()].item()
    b_opt_ours = b[torch.argmin(error_ours_l1).item()].item()

    delta_clop_opt = find_nearest(error_clop_l1, b, b_opt_clop)
    delta_ours_opt = find_nearest(error_ours_l1, b, b_opt_ours)

    delta_clop = find_nearest(error_clop_l1, b, 1.0)
    delta_ours = find_nearest(error_ours_l1, b, 1.0)

    print("--- CLOP :")
    print("  Delta:    ", delta_clop)
    print("  Delta_opt:", delta_clop_opt)
    print("  b_opt:    ", b_opt_clop)
    print("")

    print("--- Ours :")
    print("  Delta:    ", delta_ours)
    print("  Delta_opt:", delta_ours_opt)
    print("  b_opt:    ", b_opt_ours)
    print("")

    q_lop = project_and_cache(
        source_path=DATA_DIR / "block.obj",
        target_path=DATA_DIR / "block.obj",
        cache_path=CACHE_DIR / "block_lop.obj",
        device=DEVICE,
        attraction_kernel="exact",
        attraction_approx_params="",
    )

    q_clop = project_and_cache(
        source_path=DATA_DIR / "block.obj",
        target_path=DATA_DIR / "block.obj",
        cache_path=CACHE_DIR / "block_clop.obj",
        device=DEVICE,
        attraction_kernel="approximated",
        attraction_approx_params="clop",
    )

    q_ours = project_and_cache(
        source_path=DATA_DIR / "block.obj",
        target_path=DATA_DIR / "block.obj",
        cache_path=CACHE_DIR / "block_ours.obj",
        device=DEVICE,
        attraction_kernel="approximated",
        attraction_approx_params="ours",
    )

    mesh = trimesh.load_mesh(DATA_DIR / "block.obj")
    block_input = mesh.vertices

    block_lop = q_lop.cpu().numpy()
    block_clop = q_clop.cpu().numpy()
    block_ours = q_ours.cpu().numpy()

    ps.init()

    render_color = (0.890, 0.612, 0.110)

    # ps.set_SSAA_factor(4)

    ps.set_ground_plane_mode("shadow_only")
    ps_block_input = ps.register_point_cloud("Block Input", block_input, enabled=False)
    ps_block_input.set_color(render_color)

    ps_block_lop = ps.register_point_cloud("Block K_LOP", block_lop)
    ps_block_lop.set_color(render_color)

    ps_block_clop = ps.register_point_cloud("Block ^K_LOP - CLOP", block_clop)

    ps_block_ours = ps.register_point_cloud("Block ^K_LOP - Ours", block_ours)

    ps.show()


if __name__ == "__main__":
    main()
