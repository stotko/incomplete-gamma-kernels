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
    p_norm: float,
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
        h_percent_bb = 6.0
        iterations = 30

        q = igk.lop(
            p,
            q0,
            attraction_kernel="generalized",
            density_weight_scheme="wlop",
            repulsion_function="wlop",
            mu=mu,
            h_percent_bb=h_percent_bb,
            iterations=iterations,
            attraction_p_norm=p_norm,
            verbose=False,
        )

        print(f'Storing to cache "{cache_path}"')
        mesh_new = trimesh.PointCloud(q.cpu().numpy())
        mesh_new.export(cache_path)

        return q


def main() -> None:
    DATA_DIR = pathlib.Path(__file__).parents[1] / "data"
    CACHE_DIR = DATA_DIR / "cache"
    DEVICE = torch.device("cuda")

    CACHE_DIR.mkdir(exist_ok=True)

    q_0_0 = project_and_cache(
        source_path=DATA_DIR / "elephant_noisy.obj",
        target_path=DATA_DIR / "elephant_noisy.obj",
        cache_path=CACHE_DIR / "elephant_0_0.obj",
        device=DEVICE,
        p_norm=0.0,
    )

    q_0_5 = project_and_cache(
        source_path=DATA_DIR / "elephant_noisy.obj",
        target_path=DATA_DIR / "elephant_noisy.obj",
        cache_path=CACHE_DIR / "elephant_0_5.obj",
        device=DEVICE,
        p_norm=0.5,
    )

    q_1_0 = project_and_cache(
        source_path=DATA_DIR / "elephant_noisy.obj",
        target_path=DATA_DIR / "elephant_noisy.obj",
        cache_path=CACHE_DIR / "elephant_1_0.obj",
        device=DEVICE,
        p_norm=1.0,
    )

    q_1_5 = project_and_cache(
        source_path=DATA_DIR / "elephant_noisy.obj",
        target_path=DATA_DIR / "elephant_noisy.obj",
        cache_path=CACHE_DIR / "elephant_1_5.obj",
        device=DEVICE,
        p_norm=1.5,
    )

    q_2_0 = project_and_cache(
        source_path=DATA_DIR / "elephant_noisy.obj",
        target_path=DATA_DIR / "elephant_noisy.obj",
        cache_path=CACHE_DIR / "elephant_2_0.obj",
        device=DEVICE,
        p_norm=2.0,
    )

    mesh = trimesh.load_mesh(DATA_DIR / "elephant_noisy.obj")
    elephant_noisy = mesh.vertices

    mesh = trimesh.load_mesh(DATA_DIR / "elephant.obj")
    elephant_original = mesh.vertices

    elephant_0_0 = q_0_0.cpu().numpy()
    elephant_0_5 = q_0_5.cpu().numpy()
    elephant_1_0 = q_1_0.cpu().numpy()
    elephant_1_5 = q_1_5.cpu().numpy()
    elephant_2_0 = q_2_0.cpu().numpy()

    ps.init()

    render_radius = 0.002
    render_color = (0.890, 0.612, 0.110)

    # ps.set_SSAA_factor(4)

    ps.set_ground_plane_mode("shadow_only")
    ps_elephant_noisy = ps.register_point_cloud("Elephant Noisy", elephant_noisy)
    ps_elephant_noisy.set_color(render_color)
    ps_elephant_noisy.set_radius(render_radius)

    ps_elephant_0_0 = ps.register_point_cloud("Elephant p -> 0", elephant_0_0, enabled=False)
    ps_elephant_0_0.set_color(render_color)
    ps_elephant_0_0.set_radius(render_radius)

    ps_elephant_0_5 = ps.register_point_cloud("Elephant p = 0.5", elephant_0_5, enabled=False)
    ps_elephant_0_5.set_color(render_color)
    ps_elephant_0_5.set_radius(render_radius)

    ps_elephant_1_0 = ps.register_point_cloud("Elephant p = 1", elephant_1_0, enabled=False)
    ps_elephant_1_0.set_color(render_color)
    ps_elephant_1_0.set_radius(render_radius)

    ps_elephant_1_5 = ps.register_point_cloud("Elephant p = 1.5", elephant_1_5, enabled=False)
    ps_elephant_1_5.set_color(render_color)
    ps_elephant_1_5.set_radius(render_radius)

    ps_elephant_2_0 = ps.register_point_cloud("Elephant p = 2", elephant_2_0, enabled=False)
    ps_elephant_2_0.set_color(render_color)
    ps_elephant_2_0.set_radius(render_radius)

    ps_elephant_original = ps.register_point_cloud("Elephant Original", elephant_original)
    ps_elephant_original.set_color(render_color)
    ps_elephant_original.set_radius(render_radius)

    ps.show()


if __name__ == "__main__":
    main()
