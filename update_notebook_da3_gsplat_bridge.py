import json
from pathlib import Path


NB_PATH = Path("da3_gsplat_colab_pipeline.ipynb")


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(True),
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


bridge_md = r"""
# 15.1 Verified DA3-to-gsplat Trainer Bridge

The current gsplat `examples/simple_trainer.py` expects a COLMAP-like `Parser` and `Dataset` interface.

Observed interface:

- `Parser`
  - `image_names`
  - `image_paths`
  - `camtoworlds`
  - `camera_ids`
  - `camera_indices`
  - `Ks_dict`
  - `params_dict`
  - `imsize_dict`
  - `mask_dict`
  - `points`
  - `points_rgb`
  - `scene_scale`
  - `num_cameras`
- `Dataset.__getitem__`
  - `image`
  - `K`
  - `camtoworld`
  - `image_id`
  - `camera_idx`
  - optional `mask`

This bridge writes:

```text
/content/gsplat/examples/datasets/da3.py
/content/gsplat/examples/da3_train_entry.py
```

The entry script reuses gsplat's official `simple_trainer.py` training `Runner`, initialization, rasterization, densification, and `gsplat.export_splats(...)` export flow, but injects a DA3-compatible Parser/Dataset.
"""


bridge_code = r'''
# ============================================================
# 15.1 Verified DA3-to-gsplat trainer bridge
# ============================================================

from pathlib import Path
import textwrap


def write_da3_gsplat_bridge(config):
    repo_dir = Path(config.gsplat_repo_dir)
    examples_dir = repo_dir / "examples"
    datasets_dir = examples_dir / "datasets"
    simple_trainer_path = examples_dir / "simple_trainer.py"

    if not simple_trainer_path.exists():
        raise FileNotFoundError(f"Missing gsplat simple_trainer.py: {simple_trainer_path}")

    datasets_dir.mkdir(parents=True, exist_ok=True)

    da3_dataset_path = datasets_dir / "da3.py"
    da3_entry_path = examples_dir / "da3_train_entry.py"
    da3_trainer_path = examples_dir / "da3_trainer.py"
    datasets_init_path = datasets_dir / "__init__.py"

    da3_dataset_code = r"""
import json
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from plyfile import PlyData


def _load_points_ply(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing initialization point cloud: {path}")

    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    names = vertex.dtype.names

    points = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )

    if {"red", "green", "blue"}.issubset(set(names)):
        colors = np.stack(
            [
                np.asarray(vertex["red"], dtype=np.uint8),
                np.asarray(vertex["green"], dtype=np.uint8),
                np.asarray(vertex["blue"], dtype=np.uint8),
            ],
            axis=1,
        )
    else:
        colors = np.full((points.shape[0], 3), 127, dtype=np.uint8)

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]

    if points.shape[0] == 0:
        raise ValueError(f"No valid points in {path}")

    return points.astype(np.float32), colors.astype(np.uint8)


class DA3Parser:
    # DA3 transforms.json parser matching gsplat examples/simple_trainer.py needs.

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        load_exposure: bool = False,
    ):
        self.data_dir = str(data_dir)
        self.factor = int(factor)
        self.normalize = bool(normalize)
        self.test_every = max(1, int(test_every))
        self.load_exposure = bool(load_exposure)

        data_dir = Path(data_dir)
        transforms_path = data_dir / "transforms.json"
        points_path = data_dir / "points3D.ply"

        if not transforms_path.exists():
            raise FileNotFoundError(f"Missing transforms.json: {transforms_path}")

        with open(transforms_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        frames = meta.get("frames", [])
        if not frames:
            raise ValueError(f"No frames found in {transforms_path}")

        image_names = []
        image_paths = []
        camtoworlds = []
        camera_ids = []
        Ks_dict = {}
        params_dict = {}
        imsize_dict = {}
        mask_dict = {}

        default_K = np.array(
            [
                [float(meta["fl_x"]), 0.0, float(meta["cx"])],
                [0.0, float(meta["fl_y"]), float(meta["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        default_w = int(meta["w"])
        default_h = int(meta["h"])

        for idx, frame in enumerate(frames):
            rel = frame["file_path"]
            image_path = data_dir / rel
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image referenced by transforms.json: {image_path}")

            c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
            if c2w.shape != (4, 4):
                raise ValueError(f"Expected transform_matrix [4,4] at frame {idx}, got {c2w.shape}")

            if "intrinsic" in frame:
                K = np.asarray(frame["intrinsic"], dtype=np.float32)
            else:
                K = default_K.copy()

            if K.shape != (3, 3):
                raise ValueError(f"Expected K [3,3] at frame {idx}, got {K.shape}")

            camera_id = idx
            image_names.append(Path(rel).name)
            image_paths.append(str(image_path))
            camtoworlds.append(c2w)
            camera_ids.append(camera_id)
            Ks_dict[camera_id] = K
            params_dict[camera_id] = np.array([], dtype=np.float32)

            # Trust actual image size over metadata if available.
            image = imageio.imread(str(image_path))[..., :3]
            h, w = image.shape[:2]
            imsize_dict[camera_id] = (int(w), int(h))
            mask_dict[camera_id] = None

        points, points_rgb = _load_points_ply(points_path)

        self.image_names = image_names
        self.image_paths = image_paths
        self.camtoworlds = np.stack(camtoworlds, axis=0).astype(np.float32)
        self.camera_ids = camera_ids
        self.Ks_dict = Ks_dict
        self.params_dict = params_dict
        self.imsize_dict = imsize_dict
        self.mask_dict = mask_dict
        self.points = points
        self.points_err = np.zeros((points.shape[0],), dtype=np.float32)
        self.points_rgb = points_rgb
        self.point_indices = {name: np.array([], dtype=np.int64) for name in image_names}
        self.transform = np.eye(4, dtype=np.float32)
        self.extconf = {"spiral_radius_scale": 1.0}
        self.bounds = np.array([0.01, 1e10], dtype=np.float32)
        self.exposure_values = [None] * len(image_paths)

        self.camera_id_to_idx = {cid: idx for idx, cid in enumerate(sorted(set(camera_ids)))}
        self.camera_indices = [self.camera_id_to_idx[cid] for cid in camera_ids]
        self.num_cameras = len(self.camera_id_to_idx)

        camera_locations = self.camtoworlds[:, :3, 3]
        center = camera_locations.mean(axis=0)
        camera_extent = np.linalg.norm(camera_locations - center, axis=1).max()
        point_extent = np.linalg.norm(points - points.mean(axis=0), axis=1).max()
        self.scene_scale = float(max(camera_extent, point_extent * 0.1, 1e-3))

        if normalize:
            print("[DA3Parser] normalize=True was requested, but DA3 bridge keeps DA3 world scale unchanged.")


class DA3Dataset:
    # Dataset output schema matched to gsplat examples/simple_trainer.py.

    def __init__(
        self,
        parser: DA3Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths

        indices = np.arange(len(self.parser.image_names))
        if len(indices) == 1:
            self.indices = indices
        elif split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
            if len(self.indices) == 0:
                self.indices = indices
        else:
            self.indices = indices[indices % self.parser.test_every == 0]
            if len(self.indices) == 0:
                self.indices = indices[:1]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = int(self.indices[item])
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()
        camtoworld = self.parser.camtoworlds[index].copy()
        mask = self.parser.mask_dict[camera_id]

        if self.patch_size is not None:
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y
            if mask is not None:
                mask = mask[y : y + self.patch_size, x : x + self.patch_size]

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworld).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": torch.tensor(item, dtype=torch.long),
            "camera_idx": torch.tensor(self.parser.camera_indices[index], dtype=torch.long),
        }

        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        if self.load_depths:
            raise NotImplementedError(
                "DA3 depth_loss samples are not wired in this bridge. "
                "Run with depth_loss=False, or extend DA3Dataset to return points/depths."
            )

        return data
"""

    da3_entry_code = r"""
import argparse
from pathlib import Path

import torch

from datasets.da3 import DA3Dataset, DA3Parser
from gsplat import export_splats
import simple_trainer


FINAL_GAUSSIAN_NOTE = "This is a 3DGS Gaussian PLY, not a triangle mesh PLY."


def export_final_gaussian_ply(runner, save_to: Path) -> None:
    # Explicitly export the trained Gaussian primitives with gsplat.export_splats.
    save_to = Path(save_to)
    save_to.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        if runner.cfg.app_opt:
            rgb = runner.app_module(
                features=runner.splats["features"],
                embed_ids=None,
                dirs=torch.zeros_like(runner.splats["means"][None, :, :]),
                sh_degree=runner.cfg.sh_degree,
            )
            rgb = rgb + runner.splats["colors"]
            rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
            sh0 = simple_trainer.rgb_to_sh(rgb)
            shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
        else:
            sh0 = runner.splats["sh0"]
            shN = runner.splats["shN"]

        export_splats(
            means=runner.splats["means"],
            scales=runner.splats["scales"],
            quats=runner.splats["quats"],
            opacities=runner.splats["opacities"],
            sh0=sh0,
            shN=shN,
            format="ply",
            save_to=str(save_to),
        )

    if not save_to.exists():
        raise RuntimeError(f"gsplat.export_splats did not create {save_to}")

    print(f"Explicit gsplat Gaussian PLY export completed: {save_to}")
    print(FINAL_GAUSSIAN_NOTE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--points-ply", required=True)
    parser.add_argument("--test-every", type=int, default=8)
    parser.add_argument("--data-factor", type=int, default=1)
    parser.add_argument(
        "--final-ply",
        default=None,
        help="Explicit gsplat.export_splats output path. Defaults to <train-dir>/ply/gaussians.ply.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    train_dir = Path(args.train_dir)
    points_ply = Path(args.points_ply)
    final_ply = Path(args.final_ply) if args.final_ply else train_dir / "ply" / "gaussians.ply"

    if not (dataset_dir / "transforms.json").exists():
        raise FileNotFoundError(f"Missing transforms.json: {dataset_dir / 'transforms.json'}")
    if not points_ply.exists():
        raise FileNotFoundError(f"Missing points ply: {points_ply}")

    simple_trainer.Parser = DA3Parser
    simple_trainer.Dataset = DA3Dataset

    cfg = simple_trainer.Config()
    cfg.disable_viewer = True
    cfg.disable_video = True
    cfg.data_type = "colmap"
    cfg.data_dir = str(dataset_dir)
    cfg.data_factor = int(args.data_factor)
    cfg.result_dir = str(train_dir)
    cfg.test_every = max(1, int(args.test_every))
    cfg.batch_size = 1
    cfg.max_steps = int(args.iterations)
    cfg.eval_steps = []
    cfg.save_steps = [int(args.iterations)]
    cfg.save_ply = False
    cfg.ply_steps = []
    cfg.init_type = "sfm"
    cfg.normalize_world_space = False
    cfg.load_exposure = False
    cfg.depth_loss = False

    runner = simple_trainer.Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
    runner.train()
    if hasattr(runner, "export_ppisp_reports"):
        try:
            runner.export_ppisp_reports()
        except Exception as exc:
            print(f"Skipping optional ppisp report export: {exc!r}")
    export_final_gaussian_ply(runner, final_ply)


if __name__ == "__main__":
    main()

"""

    da3_dataset_path.write_text(textwrap.dedent(da3_dataset_code).lstrip(), encoding="utf-8")
    da3_entry_path.write_text(textwrap.dedent(da3_entry_code).lstrip(), encoding="utf-8")
    da3_trainer_path.write_text(
        textwrap.dedent(da3_entry_code).replace("da3_train_entry.py", "da3_trainer.py").lstrip(),
        encoding="utf-8",
    )
    if not datasets_init_path.exists():
        datasets_init_path.write_text("", encoding="utf-8")

    # Use the verified bridge entry instead of a guessed simple_trainer CLI.
    config.gsplat_command_template = (
        "python {repo_dir}/examples/da3_trainer.py "
        "--dataset-dir {dataset_dir} "
        "--train-dir {train_dir} "
        "--iterations {iterations} "
        "--points-ply {points_ply}"
    )

    print("DA3 gsplat bridge written:")
    print(f"  {da3_dataset_path}")
    print(f"  {da3_entry_path}")
    print(f"  {da3_trainer_path}")
    print(f"  {datasets_init_path}")
    print("Updated config.gsplat_command_template:")
    print(config.gsplat_command_template)
    print()
    print("The final training export uses gsplat.export_splats inside simple_trainer.py.")
    print("Exported PLY candidates will be Gaussian PLY files, not triangle mesh PLY files.")

    return da3_dataset_path, da3_trainer_path


da3_dataset_bridge_path, da3_trainer_path = write_da3_gsplat_bridge(config)
'''


nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

cells = nb["cells"]

# Insert bridge before the gsplat training markdown cell.
insert_at = None
for i, cell in enumerate(cells):
    if cell.get("cell_type") == "markdown" and "# 16. gsplat Training" in "".join(cell.get("source", [])):
        insert_at = i
        break

if insert_at is None:
    raise RuntimeError("Could not find section 16 marker")

# Avoid duplicate insertion.
if not any("15.1 Verified DA3-to-gsplat Trainer Bridge" in "".join(c.get("source", [])) for c in cells):
    cells[insert_at:insert_at] = [md(bridge_md), code(bridge_code)]

NB_PATH.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Updated {NB_PATH.resolve()}")
