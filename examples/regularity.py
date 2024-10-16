from __future__ import annotations

import pathlib

import cmasher as cmr
import torch
import trimesh
from matplotlib import pyplot as plt
from tqdm import tqdm

import incomplete_gamma_kernels as igk


def bounding_box_diagonal(p: torch.Tensor) -> float:
    bb_max, _ = torch.max(p, dim=0)
    bb_min, _ = torch.min(p, dim=0)
    return torch.linalg.norm(bb_max - bb_min).item()


def compute_regularity_and_cache(
    source_path: pathlib.Path,
    target_path: pathlib.Path,
    cache_path: pathlib.Path,
    device: torch.device,
    h_percent_bb_list: list[float],
    mu_list: list[float],
    density_weight_scheme: str,
    force_compute: bool = False,
) -> torch.Tensor:
    if not force_compute and cache_path.exists():
        print(f'Loading from cache "{cache_path}"')

        regularity = torch.load(cache_path, weights_only=True)

        return regularity
    else:
        print(f"Computing regularity")

        mesh = trimesh.load_mesh(target_path)
        p = torch.from_numpy(mesh.vertices).to(dtype=torch.float32, device=device)

        mesh = trimesh.load_mesh(source_path)
        q0 = torch.from_numpy(mesh.vertices).to(dtype=torch.float32, device=device)

        regularity = torch.empty((len(mu_list), len(h_percent_bb_list)), dtype=torch.float32, device=device)

        iterations = 100

        for i in tqdm(range(len(mu_list)), desc="  mu", position=0):
            for j in tqdm(range(len(h_percent_bb_list)), desc="  h ", position=1, leave=False):
                regularity[i, j] = igk.regularity(
                    igk.lop(
                        p,
                        q0,
                        attraction_kernel="exact",
                        density_weight_scheme=density_weight_scheme,
                        repulsion_function="wlop",
                        mu=mu_list[i],
                        h_percent_bb=h_percent_bb_list[j],
                        iterations=iterations,
                        verbose=False,
                    )
                )

        print(f'Storing to cache "{cache_path}"')
        torch.save(regularity, cache_path)

        return regularity


def main() -> None:
    DATA_DIR = pathlib.Path(__file__).parents[1] / "data"
    CACHE_DIR = DATA_DIR / "cache"
    DEVICE = torch.device("cuda")

    CACHE_DIR.mkdir(exist_ok=True)

    mesh = trimesh.load_mesh(DATA_DIR / "bird_image.obj")
    bird_image = torch.from_numpy(mesh.vertices).to(dtype=torch.float32, device=DEVICE)

    h_percent_bb_list = torch.linspace(1.5, 4.5, 61, dtype=torch.float32, device=DEVICE).tolist()
    mu_list = torch.linspace(0.0, 0.5, 51, dtype=torch.float32, device=DEVICE).tolist()

    regularity_no_weights = compute_regularity_and_cache(
        source_path=DATA_DIR / "bird_image_subset.obj",
        target_path=DATA_DIR / "bird_image.obj",
        cache_path=CACHE_DIR / "regularity_no_weights.pt",
        device=DEVICE,
        h_percent_bb_list=h_percent_bb_list,
        mu_list=mu_list,
        density_weight_scheme="none",
    )

    regularity_wlop = compute_regularity_and_cache(
        source_path=DATA_DIR / "bird_image_subset.obj",
        target_path=DATA_DIR / "bird_image.obj",
        cache_path=CACHE_DIR / "regularity_wlop.pt",
        device=DEVICE,
        h_percent_bb_list=h_percent_bb_list,
        mu_list=mu_list,
        density_weight_scheme="wlop",
    )

    regularity_ours_simple = compute_regularity_and_cache(
        source_path=DATA_DIR / "bird_image_subset.obj",
        target_path=DATA_DIR / "bird_image.obj",
        cache_path=CACHE_DIR / "regularity_ours_simple.pt",
        device=DEVICE,
        h_percent_bb_list=h_percent_bb_list,
        mu_list=mu_list,
        density_weight_scheme="ours_simple",
    )

    regularity_ours = compute_regularity_and_cache(
        source_path=DATA_DIR / "bird_image_subset.obj",
        target_path=DATA_DIR / "bird_image.obj",
        cache_path=CACHE_DIR / "regularity_ours.pt",
        device=DEVICE,
        h_percent_bb_list=h_percent_bb_list,
        mu_list=mu_list,
        density_weight_scheme="ours",
    )

    percent_bb = bounding_box_diagonal(bird_image) / 100.0
    for reg in [regularity_no_weights, regularity_wlop, regularity_ours_simple, regularity_ours]:
        reg /= percent_bb

    titles = [
        "LOP:          ",
        "WLOP:         ",
        "Ours (Simple):",
        "Ours:         ",
        "WLOP:         ",
        "Ours (Simple):",
        "Ours:         ",
    ]

    regularity = [
        regularity_no_weights,
        regularity_wlop,
        regularity_ours_simple,
        regularity_ours,
        (regularity_wlop - regularity_no_weights) / regularity_no_weights,
        (regularity_ours_simple - regularity_no_weights) / regularity_no_weights,
        (regularity_ours - regularity_no_weights) / regularity_no_weights,
    ]

    min_reg = 0.0
    max_reg = 0.5
    max_ratio = 0.5

    vmin = [
        min_reg,
        min_reg,
        min_reg,
        min_reg,
        -max_ratio,
        -max_ratio,
        -max_ratio,
    ]

    vmax = [
        max_reg,
        max_reg,
        max_reg,
        max_reg,
        max_ratio,
        max_ratio,
        max_ratio,
    ]

    cmap = [
        cmr.get_sub_cmap("turbo", 0.0, 1.0),
        cmr.get_sub_cmap("turbo", 0.0, 1.0),
        cmr.get_sub_cmap("turbo", 0.0, 1.0),
        cmr.get_sub_cmap("turbo", 0.0, 1.0),
        cmr.get_sub_cmap("coolwarm", 0.0, 1.0),
        cmr.get_sub_cmap("coolwarm", 0.0, 1.0),
        cmr.get_sub_cmap("coolwarm", 0.0, 1.0),
    ]

    print("--- Mean Regularity :")
    for i in range(0, 4):
        print(titles[i], torch.mean(regularity[i]).item())
    print("")

    print("--- Mean Delta Regularity w.r.t. LOP :")
    for i in range(4, 7):
        print(titles[i], torch.mean(regularity[i]).item())
    print("")

    delta_ours_simple_wlop = regularity[2] - regularity[1]
    delta_ours_wlop = regularity[3] - regularity[1]
    delta_ours_ours_simple = regularity[3] - regularity[2]

    print("--- Percent Better Regularity :")
    print(
        "Ours (Simple) --> WLOP:         ",
        (delta_ours_simple_wlop[delta_ours_simple_wlop <= 0.0].numel() / regularity[1].numel()) * 100.0,
        "%",
    )
    print(
        "Ours          --> WLOP:         ",
        (delta_ours_wlop[delta_ours_wlop <= 0.0].numel() / regularity[1].numel()) * 100.0,
        "%",
    )
    print(
        "Ours          --> Ours (Simple):",
        (delta_ours_ours_simple[delta_ours_ours_simple <= 0.0].numel() / regularity[2].numel()) * 100.0,
        "%",
    )

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    ax_flatten = ax.flatten()

    ax_flatten[0].set_axis_off()
    ax_flatten[2].set_axis_off()
    for i in range(0, len(titles)):
        if i == 0:
            offset = 1
        else:
            offset = 2

        ax_flatten[i + offset].set_title(titles[i].strip())

        ax_flatten[i + offset].set_xlim(1.5, 4.5)
        ax_flatten[i + offset].set_ylim(0.0, 0.5)

        ax_flatten[i + offset].set_xticks([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
        ax_flatten[i + offset].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax_flatten[i + offset].set_xticklabels([])
        ax_flatten[i + offset].set_yticklabels([])
        ax_flatten[i + offset].tick_params(direction="in", right=True, top=True)

        im = ax_flatten[i + offset].imshow(
            regularity[i].cpu().numpy(),
            origin="lower",
            cmap=cmap[i],
            vmin=vmin[i],
            vmax=vmax[i],
            extent=(1.5, 4.5, 0.0, 0.5),
            aspect=5,
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
