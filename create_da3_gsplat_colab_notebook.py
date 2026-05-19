import json
from pathlib import Path


NOTEBOOK_PATH = Path("da3_gsplat_colab_pipeline.ipynb")


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


cells = []

cells.append(md(r"""
# RGB Video -> DA3 -> gsplat -> 3DGS Gaussian PLY

This Google Colab Pro notebook converts a short RGB video into a 3D Gaussian Splatting result.

Fixed repositories:

- Depth Anything 3: https://github.com/ByteDance-Seed/depth-anything-3
- gsplat: https://github.com/nerfstudio-project/gsplat

Input:

- RGB video
- Maximum duration: 300 seconds
- Supported formats in later cells: `.mp4`, `.mov`, `.mkv`, `.avi`

Final output:

```text
outputs/{job_id}/result/gaussians.ply
```

Important:

```text
gaussians.ply is a 3DGS Gaussian PLY, not a triangle mesh PLY.
```

A mesh PLY usually stores vertices and faces. A 3DGS Gaussian PLY stores Gaussian primitive attributes such as position, scale, rotation, opacity, and color or spherical harmonics. Opening the final file as a regular mesh may not show the expected surface.

Run order:

1. Project overview
2. Runtime check
3. Google Drive mount
4. Config
5. Directory setup
6. Dependency installation
7. Input video preparation
8. Frame extraction
9. DA3 repository setup
10. DA3 inference adapter
11. DA3 inference execution
12. DA3 result validation
13. Initial point cloud generation
14. gsplat repository setup
15. DA3-to-gsplat dataset conversion
16. gsplat training
17. Final Gaussian PLY export / download / debug utilities
"""))

cells.append(code(r"""
# ============================================================
# 2. Runtime check
# ============================================================

import sys
import os
import json
import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Optional, Literal, Dict, Any, List

print("Python version:")
print(sys.version)
print()

try:
    import torch
    print("torch installed: True")
    print(f"torch version: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version reported by torch: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU name: None")
except Exception as exc:
    print("torch installed: False or failed to import")
    print(f"torch import error: {repr(exc)}")

print()
print("nvidia-smi:")
!nvidia-smi
"""))

cells.append(md(r"""
# 4. Config

All major settings are managed by `PipelineConfig`.

Drive default:

```text
/content/drive/MyDrive/da3_3dgs_colab
```

Local default:

```text
/content/da3_3dgs_colab
```
"""))

cells.append(code(r"""
# ============================================================
# 4. Config
# ============================================================

DA3_REPO_URL = "https://github.com/ByteDance-Seed/depth-anything-3"
GSPLAT_REPO_URL = "https://github.com/nerfstudio-project/gsplat"
VGGT_REPO_URL = "https://github.com/facebookresearch/vggt"


@dataclass
class PipelineConfig:
    use_drive: bool = True
    video_source: Literal["upload", "drive_path"] = "drive_path"
    input_video_path: str = "/content/drive/MyDrive/da3_3dgs_colab/inputs/input_video.mp4"
    max_video_seconds: int = 300

    project_root: Optional[str] = None
    job_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    output_dir: str = ""
    work_dir: str = ""
    frames_dir: str = ""
    da3_output_dir: str = ""
    pointcloud_dir: str = ""
    gsplat_dataset_dir: str = ""
    gsplat_train_dir: str = ""
    result_dir: str = ""
    result_ply_path: str = ""

    frame_fps: float = 2.0
    max_frames: int = 600
    resize_width: Optional[int] = 1008
    resize_height: Optional[int] = 756
    overwrite: bool = False
    resume: bool = True

    da3_repo_url: str = DA3_REPO_URL
    da3_repo_dir: str = "/content/depth-anything-3"
    da3_repo_revision: str = ""
    da3_model_name: str = "DepthAnything3"
    da3_checkpoint_or_hf_id: str = "depth-anything/DA3NESTED-GIANT-LARGE"
    da3_device: str = "cuda"
    da3_use_ray_pose: bool = False
    da3_mock_mode: bool = False
    da3_auto_restart_after_dependency_install: bool = True
    confidence_threshold: float = 0.5

    vggt_repo_url: str = VGGT_REPO_URL
    vggt_repo_dir: str = "/content/vggt"
    vggt_repo_revision: str = ""
    vggt_fallback_policy: Literal["off", "compare_only", "prefer_vggt_colmap"] = "compare_only"
    vggt_use_ba: bool = False
    vggt_fail_open: bool = True
    vggt_scene_dir: str = ""

    point_stride: int = 4
    max_points: int = 1_000_000
    voxel_size: float = 0.01
    wall_plane_filter_enabled: bool = True
    wall_plane_min_luma: float = 175.0
    wall_plane_max_chroma: float = 35.0
    wall_plane_distance_threshold: float = 0.035
    wall_plane_project_distance: float = 0.10
    wall_plane_min_inliers: int = 4000
    wall_plane_max_planes: int = 6
    extrinsics_type: Literal["w2c", "c2w"] = "w2c"
    camera_convention: str = "opencv"

    gsplat_repo_url: str = GSPLAT_REPO_URL
    gsplat_repo_revision: str = ""
    repo_update_policy: Literal["reuse_existing", "pull_latest"] = "reuse_existing"
    install_gsplat_from_pip: bool = False
    gsplat_repo_dir: str = "/content/gsplat"
    gsplat_iterations: int = 7000
    gsplat_command_template: str = (
        "python {repo_dir}/examples/da3_trainer.py "
        "--dataset-dir {dataset_dir} "
        "--train-dir {train_dir} "
        "--iterations {iterations} "
        "--points-ply {points_ply}"
    )
    dry_run: bool = True

    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "ply_note": "This is a 3DGS Gaussian PLY, not a triangle mesh PLY.",
        "da3_repo_url": DA3_REPO_URL,
        "gsplat_repo_url": GSPLAT_REPO_URL,
        "vggt_repo_url": VGGT_REPO_URL,
    })

    def __post_init__(self):
        if self.project_root is None:
            self.project_root = (
                "/content/drive/MyDrive/da3_3dgs_colab"
                if self.use_drive
                else "/content/da3_3dgs_colab"
            )
        self.refresh_paths()
        if self.da3_repo_url != DA3_REPO_URL:
            raise ValueError(f"da3_repo_url must be {DA3_REPO_URL}")
        if self.gsplat_repo_url != GSPLAT_REPO_URL:
            raise ValueError(f"gsplat_repo_url must be {GSPLAT_REPO_URL}")
        if self.vggt_repo_url != VGGT_REPO_URL:
            raise ValueError(f"vggt_repo_url must be {VGGT_REPO_URL}")

    def refresh_paths(self):
        base = Path(self.project_root) / "outputs" / self.job_id
        self.output_dir = str(base)
        self.work_dir = str(base / "work")
        self.frames_dir = str(base / "work" / "frames")
        self.da3_output_dir = str(base / "work" / "da3")
        self.vggt_scene_dir = str(base / "work" / "vggt_scene")
        self.pointcloud_dir = str(base / "pointcloud")
        self.gsplat_dataset_dir = str(base / "work" / "gsplat_dataset")
        self.gsplat_train_dir = str(base / "work" / "gsplat_train")
        self.result_dir = str(base / "result")
        self.result_ply_path = str(base / "result" / "gaussians.ply")


def print_config(config: PipelineConfig):
    data = asdict(config)
    key_width = max(len(k) for k in data)
    value_width = 100
    print("=" * (key_width + value_width + 7))
    print(f"| {'CONFIG KEY'.ljust(key_width)} | {'VALUE'.ljust(value_width)} |")
    print("=" * (key_width + value_width + 7))
    for key, value in data.items():
        value_str = json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else str(value)
        chunks = [value_str[i:i + value_width] for i in range(0, len(value_str), value_width)] or [""]
        print(f"| {key.ljust(key_width)} | {chunks[0].ljust(value_width)} |")
        for chunk in chunks[1:]:
            print(f"| {' '.ljust(key_width)} | {chunk.ljust(value_width)} |")
    print("=" * (key_width + value_width + 7))


config = PipelineConfig(
    use_drive=True,
    video_source="drive_path",
    input_video_path="/content/drive/MyDrive/da3_3dgs_colab/inputs/input_video.mp4",
    job_id="corridor_vggt_wall_test_001",
    frame_fps=1.0,
    max_frames=180,
    resize_width=1008,
    resize_height=756,
    overwrite=False,
    resume=True,
    confidence_threshold=0.55,
    point_stride=3,
    max_points=1_200_000,
    voxel_size=0.008,
    wall_plane_filter_enabled=True,
    wall_plane_min_luma=170.0,
    wall_plane_max_chroma=38.0,
    wall_plane_distance_threshold=0.035,
    wall_plane_project_distance=0.10,
    wall_plane_min_inliers=2500,
    wall_plane_max_planes=8,
    vggt_fallback_policy="compare_only",
    vggt_use_ba=False,
    vggt_fail_open=True,
    da3_mock_mode=False,
    gsplat_iterations=7000,
    dry_run=False,
)

print_config(config)
print(config.metadata["ply_note"])
"""))

cells.append(code(r"""
# ============================================================
# 3. Google Drive mount
# ============================================================

if config.use_drive:
    from google.colab import drive
    drive.mount("/content/drive")
    config.project_root = "/content/drive/MyDrive/da3_3dgs_colab"
    print("Google Drive mounted at /content/drive")
else:
    config.project_root = "/content/da3_3dgs_colab"
    print("Using local /content storage. This does not survive runtime reset.")

config.refresh_paths()
print(f"Project root: {config.project_root}")
print(f"Output dir: {config.output_dir}")
print(f"Final Gaussian PLY path: {config.result_ply_path}")
"""))

cells.append(code(r"""
# ============================================================
# 5. Directory setup
# ============================================================

directories = [
    Path(config.output_dir),
    Path(config.work_dir),
    Path(config.frames_dir),
    Path(config.da3_output_dir),
    Path(config.da3_output_dir) / "depth",
    Path(config.da3_output_dir) / "confidence",
    Path(config.pointcloud_dir),
    Path(config.gsplat_dataset_dir),
    Path(config.gsplat_dataset_dir) / "images",
    Path(config.gsplat_train_dir),
    Path(config.result_dir),
]

for directory in directories:
    directory.mkdir(parents=True, exist_ok=True)
    print(f"created/exists: {directory}")

with open(Path(config.output_dir) / "config.json", "w", encoding="utf-8") as f:
    json.dump(asdict(config), f, indent=2, ensure_ascii=False)

with open(Path(config.output_dir) / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(config.metadata, f, indent=2, ensure_ascii=False)

print("Directory setup complete.")
print(config.metadata["ply_note"])
"""))

cells.append(md(r"""
# 6. Dependency Installation

Only base dependencies are installed here.

This cell does not reinstall torch. Colab's default torch is used first.

DA3 and gsplat installation are handled in later sections.
"""))

cells.append(code(r"""
# ============================================================
# 6. Dependency installation
# ============================================================

!pip install -q \
    numpy \
    opencv-python \
    pillow \
    tqdm \
    matplotlib \
    open3d \
    plyfile \
    pyyaml \
    imageio \
    imageio-ffmpeg \
    scipy

print("Base dependency installation finished. torch was not reinstalled.")
"""))

cells.append(code(r"""
# ============================================================
# 6. Dependency import check
# ============================================================

import importlib

IMPORT_CHECKS = {
    "numpy": "numpy",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "tqdm": "tqdm",
    "matplotlib": "matplotlib",
    "open3d": "open3d",
    "plyfile": "plyfile",
    "yaml": "pyyaml",
    "imageio": "imageio",
    "imageio_ffmpeg": "imageio-ffmpeg",
    "scipy": "scipy",
}

failed = []
for module_name, package_name in IMPORT_CHECKS.items():
    try:
        module = importlib.import_module(module_name)
        print(f"[OK] {module_name} ({package_name}) {getattr(module, '__version__', '')}")
    except Exception as exc:
        print(f"[FAIL] {module_name} ({package_name}): {repr(exc)}")
        failed.append((module_name, package_name, repr(exc)))

try:
    import torch
    print(f"[OK] torch {torch.__version__}, cuda={torch.cuda.is_available()}")
except Exception as exc:
    print(f"[WARN] torch import failed: {repr(exc)}")

if failed:
    raise ImportError(f"Base dependency import check failed: {failed}")
"""))

cells.append(md(r"""
# 7. Input Video Preparation

Prepares the input video from Drive path or Colab upload, validates duration, and writes `video_metadata.json`.
"""))

cells.append(code(r"""
# ============================================================
# 7. Input video preparation
# ============================================================

import cv2

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


def prepare_input_video(config):
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.video_source == "upload":
        from google.colab import files
        uploaded = files.upload()
        if not uploaded or len(uploaded) != 1:
            raise ValueError("Upload exactly one video file.")
        source_path = Path("/content") / next(iter(uploaded.keys()))
    elif config.video_source == "drive_path":
        source_path = Path(config.input_video_path)
    else:
        raise ValueError("config.video_source must be 'upload' or 'drive_path'.")

    if not source_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {source_path}")
    if source_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported extension: {source_path.suffix}")

    prepared_video_path = output_dir / f"input_video{source_path.suffix.lower()}"
    if source_path.resolve() != prepared_video_path.resolve():
        if not prepared_video_path.exists() or config.overwrite:
            shutil.copy2(source_path, prepared_video_path)
    cap = cv2.VideoCapture(str(prepared_video_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {prepared_video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or frame_count <= 0:
        raise ValueError(f"Could not compute duration: fps={fps}, frame_count={frame_count}")
    duration_sec = frame_count / fps
    if duration_sec > config.max_video_seconds:
        raise ValueError(f"Video duration {duration_sec:.2f}s exceeds {config.max_video_seconds}s.")

    metadata = {
        "source_path": str(source_path),
        "prepared_video_path": str(prepared_video_path),
        "extension": source_path.suffix.lower(),
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "width": width,
        "height": height,
        "opencv_color_order": "BGR",
    }
    with open(output_dir / "video_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(json.dumps(metadata, indent=2))
    return prepared_video_path, metadata


input_video_path, video_metadata = prepare_input_video(config)
"""))

cells.append(md(r"""
# 8. Frame Extraction

Extracts RGB frames to `work/frames/frame_000001.jpg` and writes `frames.json`.
"""))

cells.append(code(r"""
# ============================================================
# 8. Frame extraction
# ============================================================

import math
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def extract_video_frames(config, input_video_path):
    frames_dir = Path(config.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames_json_path = Path(config.output_dir) / "frames.json"
    if config.resume and frames_json_path.exists() and not config.overwrite:
        print(f"Skipping frame extraction, using {frames_json_path}")
        return json.loads(frames_json_path.read_text(encoding="utf-8"))

    cap = cv2.VideoCapture(str(input_video_path))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or source_frame_count <= 0:
        cap.release()
        raise ValueError("Invalid video metadata for extraction.")

    duration_sec = source_frame_count / source_fps
    requested_count = max(1, int(math.floor(duration_sec * config.frame_fps)))
    indices = np.linspace(0, source_frame_count - 1, num=requested_count, dtype=np.int64)
    if config.max_frames and len(indices) > config.max_frames:
        keep = np.linspace(0, len(indices) - 1, num=config.max_frames, dtype=np.int64)
        indices = indices[keep]
    indices = np.unique(indices)

    frames = []
    for output_index, frame_index in enumerate(tqdm(indices, desc="Extracting frames"), start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            cap.release()
            raise ValueError(f"Failed to read frame {frame_index}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if config.resize_width and config.resize_height:
            rgb = cv2.resize(rgb, (int(config.resize_width), int(config.resize_height)), interpolation=cv2.INTER_AREA)
        height, width = rgb.shape[:2]
        image_path = frames_dir / f"frame_{output_index:06d}.jpg"
        Image.fromarray(rgb).save(image_path, quality=95)
        frames.append({
            "image_path": str(image_path),
            "original_frame_index": int(frame_index),
            "timestamp_sec": float(frame_index) / source_fps,
            "width": int(width),
            "height": int(height),
        })
    cap.release()
    frames_json_path.write_text(json.dumps(frames, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(frames)} frames and {frames_json_path}")
    return frames


def preview_sample_frames(frames, count=4):
    sample_count = min(count, len(frames))
    ids = np.linspace(0, len(frames) - 1, num=sample_count, dtype=np.int64)
    fig, axes = plt.subplots(1, sample_count, figsize=(4 * sample_count, 4))
    if sample_count == 1:
        axes = [axes]
    for ax, idx in zip(axes, ids):
        item = frames[int(idx)]
        ax.imshow(Image.open(item["image_path"]).convert("RGB"))
        ax.set_title(f"frame {int(idx)+1}\nt={item['timestamp_sec']:.2f}s")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


frames = extract_video_frames(config, input_video_path)
preview_sample_frames(frames)
"""))

cells.append(md(r"""
# 9. Depth Anything 3 Repository Setup

Clones and installs the official DA3 repository, then verifies the official import:

```python
from depth_anything_3.api import DepthAnything3
```
"""))

cells.append(code(r"""
# ============================================================
# 9. DA3 repository setup
# ============================================================

import traceback

EXPECTED_DA3_REPO_URL = "https://github.com/ByteDance-Seed/depth-anything-3"
DA3_REPO_DIR = Path(config.da3_repo_dir)


def run_shell(command, cwd=None, check=True):
    print(f"\n$ {command}")
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {command}")
    return result


def restart_runtime_once_after_da3_dependency_install(config):
    safe_job_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(config.job_id))
    marker_path = Path("/tmp") / f".da3_dependency_runtime_restart_done_{safe_job_id}"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    if marker_path.exists() or not getattr(config, "da3_auto_restart_after_dependency_install", True):
        return
    marker_path.write_text(
        "DA3 dependencies were installed in this Colab runtime and the Python process was restarted once to reload numpy/opencv native modules.\n",
        encoding="utf-8",
    )
    print(
        "\n[INFO] DA3 requirements changed numpy/opencv in the active Python runtime.\n"
        "Colab must restart once before importing depth_anything_3.\n"
        "The runtime will stop now. After it restarts, run the notebook again with the same job_id.\n"
    )
    import os
    os.kill(os.getpid(), 9)


def patch_xformers_triton_vararg_kernel():
    try:
        import xformers.triton.vararg_kernel as vararg_kernel
    except Exception as exc:
        print(f"[WARN] Could not import xformers.triton.vararg_kernel for compatibility patch: {exc!r}")
        return False

    patch_path = Path(vararg_kernel.__file__)
    text = patch_path.read_text(encoding="utf-8")
    old = "    jitted_fn.src = new_src\n    return jitted_fn\n"
    new = (
        "    if hasattr(jitted_fn, \"_unsafe_update_src\"):\n"
        "        jitted_fn._unsafe_update_src(new_src)\n"
        "        try:\n"
        "            jitted_fn.hash = None\n"
        "        except Exception:\n"
        "            pass\n"
        "    else:\n"
        "        jitted_fn.src = new_src\n"
        "    return jitted_fn\n"
    )
    if new in text:
        print(f"[OK] xformers/Triton compatibility patch already present: {patch_path}")
        return True
    if old not in text:
        print(f"[WARN] xformers patch target not found in {patch_path}; leaving file unchanged.")
        return False
    patch_path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"[OK] Patched xformers/Triton compatibility issue: {patch_path}")
    return True


def setup_da3_repository(config):
    if config.da3_repo_url != EXPECTED_DA3_REPO_URL:
        raise ValueError(f"DA3 repo must be {EXPECTED_DA3_REPO_URL}")
    if DA3_REPO_DIR.exists():
        print(f"DA3 repo exists: {DA3_REPO_DIR}")
        if (DA3_REPO_DIR / ".git").exists() and config.repo_update_policy == "pull_latest":
            run_shell("git pull", cwd=DA3_REPO_DIR, check=False)
        else:
            print("Repo update policy is reuse_existing; skipping git pull for reproducibility.")
    else:
        run_shell(f"git clone {config.da3_repo_url} {DA3_REPO_DIR}", cwd="/content")

    if config.da3_repo_revision and (DA3_REPO_DIR / ".git").exists():
        run_shell(f"git checkout {config.da3_repo_revision}", cwd=DA3_REPO_DIR)

    commit_result = run_shell("git rev-parse HEAD", cwd=DA3_REPO_DIR, check=False) if (DA3_REPO_DIR / ".git").exists() else None
    da3_commit = commit_result.stdout.strip() if commit_result and commit_result.returncode == 0 else "unknown"

    structure = {
        "repo_dir": str(DA3_REPO_DIR),
        "src_depth_anything_3_exists": (DA3_REPO_DIR / "src" / "depth_anything_3").exists(),
        "notebooks_exists": (DA3_REPO_DIR / "notebooks").exists(),
        "requirements_txt_exists": (DA3_REPO_DIR / "requirements.txt").exists(),
        "pyproject_toml_exists": (DA3_REPO_DIR / "pyproject.toml").exists(),
        "git_commit": da3_commit,
        "repo_update_policy": config.repo_update_policy,
        "requested_revision": config.da3_repo_revision,
    }
    print(json.dumps(structure, indent=2))
    (Path(config.output_dir) / "da3_repository_setup.json").write_text(json.dumps(structure, indent=2), encoding="utf-8")

    if structure["requirements_txt_exists"]:
        run_shell(f"{sys.executable} -m pip install -r requirements.txt", cwd=DA3_REPO_DIR)
    if structure["pyproject_toml_exists"]:
        run_shell(f"{sys.executable} -m pip install -e .", cwd=DA3_REPO_DIR)
    restart_runtime_once_after_da3_dependency_install(config)
    patch_xformers_triton_vararg_kernel()

    for candidate in [DA3_REPO_DIR / "src", DA3_REPO_DIR]:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    try:
        from depth_anything_3.api import DepthAnything3
    except Exception as exc:
        traceback.print_exc()
        raise RuntimeError("DA3 official API import failed. Do not invent a replacement API.") from exc
    if not hasattr(DepthAnything3, "from_pretrained"):
        raise RuntimeError("DepthAnything3.from_pretrained is missing. TODO: inspect DA3 API.")
    print("[OK] DA3 official API import and from_pretrained verified.")
    return structure


da3_repo_structure = setup_da3_repository(config)
"""))

cells.append(md(r"""
# 10. DA3 Inference Adapter

Defines `DA3Adapter` and `DA3Result`. Mock mode is supported with synthetic depth, confidence, intrinsics, and w2c extrinsics.
"""))

cells.append(code(r"""
# ============================================================
# 10. DA3 inference adapter
# ============================================================

from typing import List, Optional


@dataclass
class DA3Result:
    processed_image_paths: List[str]
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    width: int
    height: int
    extrinsics_type: str
    camera_convention: str
    depth: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    depth_paths: List[str] = field(default_factory=list)
    confidence_paths: List[str] = field(default_factory=list)


class DA3Adapter:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.output_dir = Path(config.da3_output_dir)
        self.depth_dir = self.output_dir / "depth"
        self.confidence_dir = self.output_dir / "confidence"
        self.cameras_json_path = self.output_dir / "cameras.json"
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        self.confidence_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self):
        if self.config.da3_mock_mode:
            print("DA3 mock mode enabled. Skipping real model load.")
            return None
        try:
            from depth_anything_3.api import DepthAnything3
        except Exception as exc:
            raise RuntimeError("Failed to import depth_anything_3.api.DepthAnything3") from exc
        if not hasattr(DepthAnything3, "from_pretrained"):
            raise RuntimeError("DepthAnything3.from_pretrained missing. TODO: inspect DA3 API.")
        self.model = DepthAnything3.from_pretrained(self.config.da3_checkpoint_or_hf_id)
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.config.da3_device)
        if not hasattr(self.model, "inference"):
            raise RuntimeError("Loaded DA3 model has no inference(images). TODO: inspect DA3 API.")
        return self.model

    def infer(self, image_paths: List[Path]) -> DA3Result:
        image_paths = [Path(p) for p in image_paths]
        if self.config.da3_mock_mode:
            return self._mock_infer(image_paths)
        if self.model is None:
            self.load_model()
        images = [Image.open(p).convert("RGB") for p in tqdm(image_paths, desc="Loading DA3 images")]
        prediction = self.model.inference(images)
        self._validate_prediction(prediction, len(images))
        width, height = images[0].size
        return DA3Result(
            processed_image_paths=[str(p) for p in image_paths],
            depth=np.asarray(prediction.depth, dtype=np.float32),
            confidence=np.asarray(prediction.conf, dtype=np.float32),
            intrinsics=np.asarray(prediction.intrinsics, dtype=np.float32),
            extrinsics=np.asarray(prediction.extrinsics, dtype=np.float32),
            width=int(width),
            height=int(height),
            extrinsics_type="w2c",
            camera_convention="opencv",
        )

    def save_outputs(self, result: DA3Result) -> DA3Result:
        if result.depth is None or result.confidence is None:
            raise ValueError("result.depth and result.confidence are required.")
        n = len(result.processed_image_paths)
        if result.extrinsics.shape != (n, 3, 4):
            raise ValueError(f"Expected extrinsics [N,3,4], got {result.extrinsics.shape}")
        if result.intrinsics.shape != (n, 3, 3):
            raise ValueError(f"Expected intrinsics [N,3,3], got {result.intrinsics.shape}")
        frames_out = []
        result.depth_paths = []
        result.confidence_paths = []
        for idx in tqdm(range(n), desc="Saving DA3 outputs"):
            depth_path = self.depth_dir / f"depth_{idx+1:06d}.npy"
            conf_path = self.confidence_dir / f"conf_{idx+1:06d}.npy"
            np.save(depth_path, result.depth[idx].astype(np.float32))
            np.save(conf_path, result.confidence[idx].astype(np.float32))
            result.depth_paths.append(str(depth_path))
            result.confidence_paths.append(str(conf_path))
            frames_out.append({
                "image_path": result.processed_image_paths[idx],
                "depth_path": str(depth_path),
                "confidence_path": str(conf_path),
                "intrinsic": result.intrinsics[idx].astype(float).tolist(),
                "extrinsic": result.extrinsics[idx].astype(float).tolist(),
                "extrinsics_type": "w2c",
                "camera_convention": "opencv",
                "note": "DA3 README describes extrinsics as OpenCV w2c or COLMAP format.",
            })
        cameras_json = {
            "width": result.width,
            "height": result.height,
            "extrinsics_type": "w2c",
            "camera_convention": "opencv",
            "note": "DA3 README describes extrinsics as OpenCV w2c or COLMAP format.",
            "frames": frames_out,
        }
        self.cameras_json_path.write_text(json.dumps(cameras_json, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {self.cameras_json_path}")
        return result

    def _validate_prediction(self, prediction, expected_count):
        missing = [x for x in ["depth", "conf", "extrinsics", "intrinsics"] if not hasattr(prediction, x)]
        if missing:
            raise RuntimeError(f"DA3 prediction missing fields: {missing}")
        extrinsics = np.asarray(prediction.extrinsics)
        intrinsics = np.asarray(prediction.intrinsics)
        if extrinsics.shape != (expected_count, 3, 4):
            raise ValueError(f"Expected extrinsics [N,3,4], got {extrinsics.shape}")
        if intrinsics.shape != (expected_count, 3, 3):
            raise ValueError(f"Expected intrinsics [N,3,3], got {intrinsics.shape}")
        print("DA3 prediction validation passed.")

    def _mock_infer(self, image_paths):
        image = Image.open(image_paths[0]).convert("RGB")
        width, height = image.size
        n = len(image_paths)
        yy = np.linspace(0.4, 2.0, height, dtype=np.float32).reshape(height, 1)
        xx = np.linspace(0.0, 0.3, width, dtype=np.float32).reshape(1, width)
        depth = np.stack([yy + xx + 0.01 * i for i in range(n)], axis=0).astype(np.float32)
        confidence = np.ones((n, height, width), dtype=np.float32)
        fx = fy = float(max(width, height))
        cx = width * 0.5
        cy = height * 0.5
        intrinsics = np.tile(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32), (n, 1, 1))
        extrinsics = np.zeros((n, 3, 4), dtype=np.float32)
        for i in range(n):
            extrinsics[i] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.03 * i]], dtype=np.float32)
        return DA3Result([str(p) for p in image_paths], intrinsics, extrinsics, width, height, "w2c", "opencv", depth, confidence)

    def load_existing_result(self):
        cameras = json.loads(self.cameras_json_path.read_text(encoding="utf-8"))
        frames = cameras["frames"]
        return DA3Result(
            processed_image_paths=[x["image_path"] for x in frames],
            depth_paths=[x["depth_path"] for x in frames],
            confidence_paths=[x["confidence_path"] for x in frames],
            intrinsics=np.asarray([x["intrinsic"] for x in frames], dtype=np.float32),
            extrinsics=np.asarray([x["extrinsic"] for x in frames], dtype=np.float32),
            width=int(cameras["width"]),
            height=int(cameras["height"]),
            extrinsics_type=cameras.get("extrinsics_type", "w2c"),
            camera_convention=cameras.get("camera_convention", "opencv"),
        )
"""))

cells.append(md(r"""
# 11. DA3 Inference Execution

Runs DA3 inference or resumes from existing `cameras.json`, depth files, and confidence files.
"""))

cells.append(code(r"""
# ============================================================
# 11. DA3 inference execution
# ============================================================

def da3_outputs_complete(config, expected_count):
    da3_dir = Path(config.da3_output_dir)
    cameras_path = da3_dir / "cameras.json"
    if not cameras_path.exists():
        return False
    depth_files = sorted((da3_dir / "depth").glob("depth_*.npy"))
    conf_files = sorted((da3_dir / "confidence").glob("conf_*.npy"))
    if len(depth_files) != expected_count or len(conf_files) != expected_count:
        return False
    try:
        cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
        return len(cameras.get("frames", [])) == expected_count
    except Exception:
        return False


try:
    frame_items = json.loads((Path(config.output_dir) / "frames.json").read_text(encoding="utf-8"))
    image_paths = [Path(item["image_path"]) for item in frame_items]
    print(f"DA3 mock mode: {config.da3_mock_mode}")
    print(f"DA3 checkpoint/HF ID: {config.da3_checkpoint_or_hf_id}")
    da3_adapter = DA3Adapter(config)
    if config.resume and not config.overwrite and da3_outputs_complete(config, len(image_paths)):
        print("Complete DA3 outputs exist. Skipping inference.")
        da3_result = da3_adapter.load_existing_result()
    else:
        da3_adapter.load_model()
        da3_result = da3_adapter.infer(image_paths)
        da3_result = da3_adapter.save_outputs(da3_result)
    print("DA3 inference execution completed.")
except Exception as exc:
    print("DA3 inference failed.")
    print("Check DA3 repo install, checkpoint/HF ID, depth_anything_3.api import, and prediction fields.")
    traceback.print_exc()
    raise
"""))

cells.append(md(r"""
# 12. DA3 Result Validation

Validates depth, confidence, and camera outputs, then previews sample depth maps.
"""))

cells.append(code(r"""
# ============================================================
# 12. DA3 result validation
# ============================================================

def validate_da3_results(config):
    frames = json.loads((Path(config.output_dir) / "frames.json").read_text(encoding="utf-8"))
    expected = len(frames)
    cameras_path = Path(config.da3_output_dir) / "cameras.json"
    cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    depth_files = sorted((Path(config.da3_output_dir) / "depth").glob("depth_*.npy"))
    conf_files = sorted((Path(config.da3_output_dir) / "confidence").glob("conf_*.npy"))
    if len(depth_files) != expected:
        raise ValueError(f"Depth count mismatch: {len(depth_files)} vs {expected}")
    if len(conf_files) != expected:
        raise ValueError(f"Confidence count mismatch: {len(conf_files)} vs {expected}")
    if len(cameras.get("frames", [])) != expected:
        raise ValueError(f"Camera count mismatch: {len(cameras.get('frames', []))} vs {expected}")
    for i, item in enumerate(cameras["frames"]):
        if np.asarray(item["intrinsic"]).shape != (3, 3):
            raise ValueError(f"Bad intrinsic shape at {i}")
        if np.asarray(item["extrinsic"]).shape != (3, 4):
            raise ValueError(f"Bad extrinsic shape at {i}")
        depth = np.load(depth_files[i])
        conf = np.load(conf_files[i])
        if depth.shape != conf.shape:
            raise ValueError(f"Depth/conf shape mismatch at {i}: {depth.shape} vs {conf.shape}")
    validation = {
        "status": "ok",
        "frame_count": expected,
        "depth_count": len(depth_files),
        "confidence_count": len(conf_files),
        "camera_count": len(cameras["frames"]),
        "da3_mock_mode": bool(config.da3_mock_mode),
    }
    (Path(config.da3_output_dir) / "da3_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))
    return validation


def preview_depth_maps(config, count=4):
    depth_files = sorted((Path(config.da3_output_dir) / "depth").glob("depth_*.npy"))
    sample_count = min(count, len(depth_files))
    ids = np.linspace(0, len(depth_files) - 1, sample_count, dtype=np.int64)
    fig, axes = plt.subplots(1, sample_count, figsize=(4 * sample_count, 4))
    if sample_count == 1:
        axes = [axes]
    for ax, idx in zip(axes, ids):
        depth = np.load(depth_files[int(idx)])
        valid = np.isfinite(depth)
        vmin, vmax = (np.percentile(depth[valid], 2), np.percentile(depth[valid], 98)) if valid.any() else (0, 1)
        ax.imshow(depth, cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(depth_files[int(idx)].name)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


try:
    da3_validation = validate_da3_results(config)
    preview_depth_maps(config)
except Exception:
    print("DA3 validation failed. Check DA3 repo install, checkpoint/HF ID, import, and prediction fields.")
    traceback.print_exc()
    raise
"""))

cells.append(md(r"""
# 12.5 VGGT Geometry Fallback Probe

Runs VGGT as an optional comparison/fallback route on the same extracted frames.

Default policy is `compare_only`: VGGT writes a COLMAP-style scene under `work/vggt_scene`, but DA3 remains the primary route.

Set `config.vggt_fallback_policy = "prefer_vggt_colmap"` before this cell to train gsplat from VGGT's COLMAP export instead of the DA3 bridge.
"""))

cells.append(code(r"""
# ============================================================
# 12.5 VGGT geometry fallback probe
# ============================================================

def vggt_status_path(config):
    return Path(config.vggt_scene_dir) / "vggt_fallback_status.json"


def write_vggt_status(config, status):
    scene_dir = Path(config.vggt_scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy": config.vggt_fallback_policy,
        "scene_dir": str(scene_dir),
        **status,
    }
    vggt_status_path(config).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def prepare_vggt_scene(config):
    scene_dir = Path(config.vggt_scene_dir)
    images_dir = scene_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    frames = json.loads((Path(config.output_dir) / "frames.json").read_text(encoding="utf-8"))
    copied = []
    for idx, item in enumerate(frames):
        src = Path(item["image_path"])
        dst = images_dir / f"frame_{idx+1:06d}{src.suffix.lower()}"
        if not dst.exists() or config.overwrite:
            shutil.copy2(src, dst)
        copied.append(str(dst))
    return scene_dir, copied


def setup_vggt_repo(config):
    repo_dir = Path(config.vggt_repo_dir)
    if repo_dir.exists():
        print(f"VGGT repo exists: {repo_dir}")
        if (repo_dir / ".git").exists() and config.repo_update_policy == "pull_latest":
            run_shell("git pull", cwd=repo_dir, check=False)
    else:
        run_shell(f"git clone {config.vggt_repo_url} {repo_dir}", cwd="/content")
    if config.vggt_repo_revision and (repo_dir / ".git").exists():
        run_shell(f"git checkout {config.vggt_repo_revision}", cwd=repo_dir)
    if (repo_dir / "requirements.txt").exists():
        run_shell(f"{sys.executable} -m pip install -r requirements.txt", cwd=repo_dir)
    return repo_dir


def run_vggt_fallback_probe(config):
    policy = getattr(config, "vggt_fallback_policy", "off")
    if policy == "off":
        return write_vggt_status(config, {"status": "skipped", "reason": "vggt_fallback_policy=off"})
    try:
        scene_dir, copied = prepare_vggt_scene(config)
        sparse_dir = scene_dir / "sparse"
        if config.resume and sparse_dir.exists() and any(sparse_dir.glob("**/*")) and not config.overwrite:
            return write_vggt_status(config, {
                "status": "reused",
                "image_count": len(copied),
                "sparse_dir": str(sparse_dir),
            })
        repo_dir = setup_vggt_repo(config)
        command = f"{sys.executable} demo_colmap.py --scene_dir={scene_dir}"
        if config.vggt_use_ba:
            command += " --use_ba"
        print("VGGT fallback command:")
        print(command)
        run_shell(command, cwd=repo_dir)
        expected = [
            sparse_dir / "cameras.bin",
            sparse_dir / "images.bin",
            sparse_dir / "points3D.bin",
        ]
        missing = [str(p) for p in expected if not p.exists()]
        status = {
            "status": "ok" if not missing else "incomplete",
            "image_count": len(copied),
            "sparse_dir": str(sparse_dir),
            "missing": missing,
            "note": "VGGT COLMAP export is a comparison/fallback route. DA3 remains primary unless vggt_fallback_policy='prefer_vggt_colmap'.",
        }
        if missing and not config.vggt_fail_open:
            raise RuntimeError(f"VGGT COLMAP export missing files: {missing}")
        return write_vggt_status(config, status)
    except Exception as exc:
        status = {"status": "failed", "error": repr(exc)}
        write_vggt_status(config, status)
        if not config.vggt_fail_open:
            raise
        print("VGGT fallback failed, but vggt_fail_open=True so the DA3 route will continue.")
        return status


vggt_fallback_status = run_vggt_fallback_probe(config)
"""))

cells.append(md(r"""
# 13. Initial Point Cloud Generation

Creates `outputs/{job_id}/pointcloud/init_points.ply` from DA3 depth and cameras.

This file is a regular point cloud PLY for initialization. It is not the final 3DGS Gaussian PLY.
"""))

cells.append(code(r"""
# ============================================================
# 13. Initial point cloud generation
# ============================================================

POINTCLOUD_OUTPUT_DIR = Path(config.pointcloud_dir)
POINTCLOUD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extrinsic_3x4_to_c2w_4x4(extrinsic, extrinsics_type):
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :4] = np.asarray(extrinsic, dtype=np.float32)
    if extrinsics_type == "w2c":
        return np.linalg.inv(mat).astype(np.float32)
    if extrinsics_type == "c2w":
        return mat.astype(np.float32)
    raise ValueError(f"Unsupported extrinsics_type={extrinsics_type}")


def unproject_depth_frame(depth, confidence, rgb, intrinsic, extrinsic, extrinsics_type, config):
    h, w = depth.shape
    if rgb.shape[:2] != (h, w):
        rgb = np.asarray(Image.fromarray(rgb).resize((w, h), resample=Image.BILINEAR), dtype=np.uint8)
    stride = max(1, int(config.point_stride))
    ys = np.arange(0, h, stride, dtype=np.int32)
    xs = np.arange(0, w, stride, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    z = depth[grid_y, grid_x]
    conf = confidence[grid_y, grid_x]
    valid = np.isfinite(z) & (z > 0) & np.isfinite(conf) & (conf >= float(config.confidence_threshold))
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    u = grid_x[valid].astype(np.float32)
    v = grid_y[valid].astype(np.float32)
    z = z[valid].astype(np.float32)
    colors = rgb[grid_y[valid], grid_x[valid], :]
    K = np.asarray(intrinsic, dtype=np.float32)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    points_cam = np.stack([x, y, z], axis=1).astype(np.float32)
    c2w = extrinsic_3x4_to_c2w_4x4(extrinsic, extrinsics_type)
    points_h = np.concatenate([points_cam, np.ones((len(points_cam), 1), dtype=np.float32)], axis=1)
    points = (c2w @ points_h.T).T[:, :3].astype(np.float32)
    finite = np.isfinite(points).all(axis=1)
    return points[finite], colors[finite]


def refine_low_texture_wall_planes(points, colors, config, o3d):
    if not getattr(config, "wall_plane_filter_enabled", False):
        return points, colors, {"enabled": False}
    if len(points) < int(config.wall_plane_min_inliers):
        return points, colors, {"enabled": True, "status": "skipped_too_few_points", "input_points": int(len(points))}

    colors_f = colors.astype(np.float32)
    luma = colors_f.mean(axis=1)
    chroma = colors_f.max(axis=1) - colors_f.min(axis=1)
    wall_mask = (luma >= float(config.wall_plane_min_luma)) & (chroma <= float(config.wall_plane_max_chroma))
    wall_indices = np.flatnonzero(wall_mask)
    if len(wall_indices) < int(config.wall_plane_min_inliers):
        return points, colors, {
            "enabled": True,
            "status": "skipped_not_enough_wall_candidates",
            "wall_candidate_points": int(len(wall_indices)),
        }

    wall_points = points[wall_indices]
    wall_pcd = o3d.geometry.PointCloud()
    wall_pcd.points = o3d.utility.Vector3dVector(wall_points.astype(np.float64))
    remaining = wall_pcd
    remaining_indices = wall_indices.copy()
    planes = []

    for _ in range(int(config.wall_plane_max_planes)):
        if len(remaining_indices) < int(config.wall_plane_min_inliers):
            break
        plane_model, local_inliers = remaining.segment_plane(
            distance_threshold=float(config.wall_plane_distance_threshold),
            ransac_n=3,
            num_iterations=1000,
        )
        local_inliers = np.asarray(local_inliers, dtype=np.int64)
        if len(local_inliers) < int(config.wall_plane_min_inliers):
            break
        global_inliers = remaining_indices[local_inliers]
        planes.append({
            "model": np.asarray(plane_model, dtype=np.float32),
            "inliers": global_inliers,
        })
        keep_local = np.ones(len(remaining_indices), dtype=bool)
        keep_local[local_inliers] = False
        remaining_indices = remaining_indices[keep_local]
        remaining = remaining.select_by_index(local_inliers.tolist(), invert=True)

    if not planes:
        return points, colors, {
            "enabled": True,
            "status": "skipped_no_plane_found",
            "wall_candidate_points": int(len(wall_indices)),
        }

    plane_models = np.stack([p["model"] for p in planes], axis=0)
    normals = plane_models[:, :3]
    offsets = plane_models[:, 3]
    normal_norms = np.maximum(np.linalg.norm(normals, axis=1), 1e-6)
    signed_dist = (wall_points @ normals.T + offsets[None, :]) / normal_norms[None, :]
    nearest = np.argmin(np.abs(signed_dist), axis=1)
    nearest_dist = signed_dist[np.arange(len(wall_points)), nearest]
    near_plane = np.abs(nearest_dist) <= float(config.wall_plane_project_distance)

    projected_wall_points = wall_points.copy()
    for plane_idx in range(len(planes)):
        mask = near_plane & (nearest == plane_idx)
        if not np.any(mask):
            continue
        n = normals[plane_idx] / normal_norms[plane_idx]
        projected_wall_points[mask] = wall_points[mask] - nearest_dist[mask, None] * n[None, :]

    keep = np.ones(len(points), dtype=bool)
    dropped_wall_indices = wall_indices[~near_plane]
    keep[dropped_wall_indices] = False
    refined_points = points.copy()
    refined_points[wall_indices[near_plane]] = projected_wall_points[near_plane]
    stats = {
        "enabled": True,
        "status": "ok",
        "wall_candidate_points": int(len(wall_indices)),
        "planes_found": int(len(planes)),
        "plane_inliers": [int(len(p["inliers"])) for p in planes],
        "projected_wall_points": int(np.count_nonzero(near_plane)),
        "dropped_wall_outliers": int(len(dropped_wall_indices)),
        "points_after_wall_filter": int(np.count_nonzero(keep)),
    }
    return refined_points[keep].astype(np.float32), colors[keep], stats


def generate_initial_point_cloud(config):
    import open3d as o3d
    init_ply_path = POINTCLOUD_OUTPUT_DIR / "init_points.ply"
    init_npz_path = POINTCLOUD_OUTPUT_DIR / "init_points.npz"
    if config.resume and init_ply_path.exists() and init_npz_path.exists() and not config.overwrite:
        print(f"Using existing init point cloud: {init_ply_path}")
        return init_ply_path
    cameras = json.loads((Path(config.da3_output_dir) / "cameras.json").read_text(encoding="utf-8"))
    all_points, all_colors, camera_centers = [], [], []
    for item in tqdm(cameras["frames"], desc="Backprojecting DA3 depth"):
        rgb = np.asarray(Image.open(item["image_path"]).convert("RGB"), dtype=np.uint8)
        depth = np.load(item["depth_path"]).astype(np.float32)
        conf = np.load(item["confidence_path"]).astype(np.float32)
        extrinsics_type = item.get("extrinsics_type", cameras.get("extrinsics_type", config.extrinsics_type))
        c2w = extrinsic_3x4_to_c2w_4x4(item["extrinsic"], extrinsics_type)
        camera_centers.append(c2w[:3, 3])
        points, colors = unproject_depth_frame(depth, conf, rgb, item["intrinsic"], item["extrinsic"], extrinsics_type, config)
        if len(points):
            all_points.append(points)
            all_colors.append(colors)
    if not all_points:
        raise ValueError("No valid points generated. Lower confidence_threshold or point_stride.")
    points = np.concatenate(all_points).astype(np.float32)
    colors = np.concatenate(all_colors).astype(np.uint8)
    before = len(points)
    if config.max_points and len(points) > int(config.max_points):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), size=int(config.max_points), replace=False)
        points, colors = points[idx], colors[idx]
    after_sampling = len(points)
    points, colors, wall_filter_stats = refine_low_texture_wall_planes(points, colors, config, o3d)
    if config.voxel_size and config.voxel_size > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector((colors.astype(np.float32) / 255.0).astype(np.float64))
        pcd = pcd.voxel_down_sample(float(config.voxel_size))
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = np.clip(np.asarray(pcd.colors) * 255, 0, 255).astype(np.uint8)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((colors.astype(np.float32) / 255.0).astype(np.float64))
    o3d.io.write_point_cloud(str(init_ply_path), pcd, write_ascii=False)
    np.savez_compressed(init_npz_path, points=points, colors=colors)
    stats = {
        "note": "init_points.ply is a regular point cloud PLY for 3DGS initialization, not the final Gaussian PLY.",
        "points_before_sampling": int(before),
        "points_after_sampling": int(after_sampling),
        "wall_plane_filter": wall_filter_stats,
        "points_after_voxel_downsample": int(len(points)),
        "init_ply_path": str(init_ply_path),
    }
    (POINTCLOUD_OUTPUT_DIR / "init_points_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print("This is NOT the final 3DGS Gaussian PLY.")
    visualize_camera_trajectory(np.asarray(camera_centers, dtype=np.float32))
    visualize_point_cloud_preview(points, colors)
    return init_ply_path


def visualize_camera_trajectory(camera_centers):
    if len(camera_centers) == 0:
        return
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2], marker="o")
    ax.set_title("Camera trajectory preview")
    plt.show()


def visualize_point_cloud_preview(points, colors, max_plot_points=20000):
    if len(points) > max_plot_points:
        idx = np.random.default_rng(42).choice(len(points), size=max_plot_points, replace=False)
        points, colors = points[idx], colors[idx]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=np.clip(colors / 255.0, 0, 1), s=0.5)
    ax.set_title("Initial point cloud preview")
    plt.show()


init_pointcloud_path = generate_initial_point_cloud(config)
"""))

cells.append(md(r"""
# 14. gsplat Repository Setup

Clones and inspects the official gsplat repository. The notebook does not invent CLI arguments for `simple_trainer.py`.
"""))

cells.append(code(r"""
# ============================================================
# 14. gsplat repository setup
# ============================================================

import re
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


def recover_config_for_gsplat_if_needed():
    if "config" in globals():
        return globals()["config"]

    drive_root = Path("/content/drive/MyDrive/da3_3dgs_colab")
    if not drive_root.exists():
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception as exc:
            print(f"Drive mount attempt skipped/failed: {exc!r}")

    candidates = []
    for root in [drive_root, Path("/content/da3_3dgs_colab")]:
        outputs = root / "outputs"
        if outputs.exists():
            candidates.extend(outputs.glob("*/config.json"))

    preferred = drive_root / "outputs" / "corridor_vggt_wall_test_001" / "config.json"
    if preferred.exists():
        config_path = preferred
    elif candidates:
        config_path = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    else:
        raise NameError("config is not defined and no saved config.json was found. Run cells 4-5 once, then rerun cell 14.")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = SimpleNamespace(**data)
    cfg.gsplat_repo_revision = getattr(cfg, "gsplat_repo_revision", "")
    cfg.repo_update_policy = getattr(cfg, "repo_update_policy", "reuse_existing")
    cfg.install_gsplat_from_pip = getattr(cfg, "install_gsplat_from_pip", False)
    cfg.gsplat_repo_url = getattr(cfg, "gsplat_repo_url", "https://github.com/nerfstudio-project/gsplat")
    cfg.gsplat_repo_dir = getattr(cfg, "gsplat_repo_dir", "/content/gsplat")
    cfg.output_dir = getattr(cfg, "output_dir", str(config_path.parent))
    cfg.result_ply_path = getattr(cfg, "result_ply_path", str(config_path.parent / "result" / "gaussians.ply"))
    print(f"Recovered config from: {config_path}")
    return cfg


config = recover_config_for_gsplat_if_needed()


if "run_shell" not in globals():
    def run_shell(command, cwd=None, check=True):
        print(f"\n$ {command}")
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(result.stdout)
        if check and result.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {result.returncode}: {command}")
        return result

EXPECTED_GSPLAT_REPO_URL = "https://github.com/nerfstudio-project/gsplat"
GSPLAT_REPO_DIR = Path(config.gsplat_repo_dir)


def setup_gsplat_repository(config):
    if config.gsplat_repo_url != EXPECTED_GSPLAT_REPO_URL:
        raise ValueError(f"gsplat repo must be {EXPECTED_GSPLAT_REPO_URL}")
    if config.install_gsplat_from_pip:
        run_shell(f"{sys.executable} -m pip install gsplat")
    if GSPLAT_REPO_DIR.exists():
        print(f"gsplat repo exists: {GSPLAT_REPO_DIR}")
        if (GSPLAT_REPO_DIR / ".git").exists() and config.repo_update_policy == "pull_latest":
            run_shell("git pull", cwd=GSPLAT_REPO_DIR, check=False)
        else:
            print("Repo update policy is reuse_existing; skipping git pull for reproducibility.")
    else:
        run_shell(f"git clone {config.gsplat_repo_url} {GSPLAT_REPO_DIR}", cwd="/content")

    if config.gsplat_repo_revision and (GSPLAT_REPO_DIR / ".git").exists():
        run_shell(f"git checkout {config.gsplat_repo_revision}", cwd=GSPLAT_REPO_DIR)

    glm_header = GSPLAT_REPO_DIR / "gsplat" / "cuda" / "csrc" / "third_party" / "glm" / "glm" / "gtc" / "type_ptr.hpp"
    if (GSPLAT_REPO_DIR / ".git").exists() and not glm_header.exists():
        print(f"Missing gsplat third-party GLM header: {glm_header}")
        print("Initializing gsplat git submodules recursively.")
        run_shell("git submodule update --init --recursive", cwd=GSPLAT_REPO_DIR)
    if not glm_header.exists():
        raise FileNotFoundError(
            f"Missing required gsplat GLM header after submodule init: {glm_header}. "
            "The CUDA extension cannot build without gsplat/cuda/csrc/third_party/glm."
        )

    commit_result = run_shell("git rev-parse HEAD", cwd=GSPLAT_REPO_DIR, check=False) if (GSPLAT_REPO_DIR / ".git").exists() else None
    gsplat_commit = commit_result.stdout.strip() if commit_result and commit_result.returncode == 0 else "unknown"

    examples_dir = GSPLAT_REPO_DIR / "examples"
    simple = examples_dir / "simple_trainer.py"
    structure = {
        "repo_dir": str(GSPLAT_REPO_DIR),
        "gsplat_dir_exists": (GSPLAT_REPO_DIR / "gsplat").exists(),
        "examples_dir_exists": examples_dir.exists(),
        "simple_trainer_exists": simple.exists(),
        "pyproject_toml_exists": (GSPLAT_REPO_DIR / "pyproject.toml").exists(),
        "setup_py_exists": (GSPLAT_REPO_DIR / "setup.py").exists(),
        "glm_header_exists": glm_header.exists(),
        "git_commit": gsplat_commit,
        "repo_update_policy": config.repo_update_policy,
        "requested_revision": config.gsplat_repo_revision,
    }
    print(json.dumps(structure, indent=2))
    requirements_path = examples_dir / "requirements.txt"
    if requirements_path.exists():
        filtered_requirements = Path(config.output_dir) / "gsplat_examples_requirements.filtered.txt"
        skipped_requirements = []
        kept_lines = []
        for line in requirements_path.read_text(encoding="utf-8").splitlines():
            if "ppisp" in line.lower():
                skipped_requirements.append(line)
                continue
            kept_lines.append(line)
        filtered_requirements.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
        print("Installing gsplat example requirements with optional ppisp skipped.")
        print("Skipped requirement lines:", skipped_requirements)
        run_shell(f"{sys.executable} -m pip install -r {filtered_requirements}", cwd=GSPLAT_REPO_DIR)
    install_mode = "pypi_requested" if config.install_gsplat_from_pip else "unknown"
    if not config.install_gsplat_from_pip and ((GSPLAT_REPO_DIR / "pyproject.toml").exists() or (GSPLAT_REPO_DIR / "setup.py").exists()):
        editable_result = run_shell(f"{sys.executable} -m pip install -e .", cwd=GSPLAT_REPO_DIR, check=False)
        if editable_result.returncode == 0:
            install_mode = "editable_source"
        else:
            print("Editable source install failed. Falling back to PyPI gsplat, which official gsplat recommends as the easiest install path.")
            print("Training will run from /content so the local repo does not shadow the PyPI package.")
            run_shell(f"{sys.executable} -m pip install gsplat", cwd="/content")
            install_mode = "pypi_fallback"
    run_shell(f"{sys.executable} -c \"import gsplat; print('gsplat import:', gsplat.__file__)\"", cwd="/content")
    if not simple.exists():
        raise FileNotFoundError(f"Missing {simple}")
    source = simple.read_text(encoding="utf-8")
    flags = sorted(set(re.findall(r'''["'](--[A-Za-z0-9_-]+)["']''', source)))
    inspection = {
        "simple_trainer_path": str(simple),
        "mentions_colmap": "colmap" in source.lower(),
        "mentions_transforms_json": "transforms.json" in source,
        "mentions_data_dir": "data_dir" in source,
        "mentions_result_dir": "result_dir" in source,
        "mentions_max_steps": "max_steps" in source,
        "mentions_iterations": "iterations" in source,
        "mentions_save_ply": "save_ply" in source,
        "cli_flags_found_in_source": flags,
        "install_mode": install_mode,
        "note": "Do not invent CLI flags. Use only verified args from the current simple_trainer.py.",
    }
    print(json.dumps(inspection, indent=2))
    help_result = run_shell(f"{sys.executable} {simple} --help", cwd=GSPLAT_REPO_DIR, check=False)
    inspection["help_returncode"] = help_result.returncode
    inspection["help_output_tail"] = help_result.stdout[-4000:]
    (Path(config.output_dir) / "gsplat_repository_setup.json").write_text(json.dumps({"structure": structure, "inspection": inspection}, indent=2), encoding="utf-8")
    return inspection


gsplat_trainer_inspection = setup_gsplat_repository(config)
"""))

cells.append(md(r"""
# 15. DA3-to-gsplat Dataset Conversion

Creates an intermediate dataset:

- `gsplat_dataset/images/`
- `gsplat_dataset/transforms.json`
- `gsplat_dataset/points3D.ply`

This `transforms.json` format is not guaranteed to be directly compatible with every gsplat trainer. If `simple_trainer.py` requires COLMAP, implement DA3-to-COLMAP conversion or a verified custom loader.
"""))

cells.append(code(r"""
# ============================================================
# 15. DA3-to-gsplat dataset conversion
# ============================================================

def make_c2w(extrinsic, extrinsics_type):
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :4] = np.asarray(extrinsic, dtype=np.float32)
    if extrinsics_type == "w2c":
        return np.linalg.inv(mat).astype(np.float32)
    if extrinsics_type == "c2w":
        return mat.astype(np.float32)
    raise ValueError(f"Unsupported extrinsics_type={extrinsics_type}")


def convert_da3_to_gsplat_dataset(config):
    dataset_dir = Path(config.gsplat_dataset_dir)
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    transforms_path = dataset_dir / "transforms.json"
    points3d_path = dataset_dir / "points3D.ply"
    if config.resume and transforms_path.exists() and points3d_path.exists() and not config.overwrite:
        print("Using existing gsplat intermediate dataset.")
        return transforms_path, points3d_path
    frames = json.loads((Path(config.output_dir) / "frames.json").read_text(encoding="utf-8"))
    cameras = json.loads((Path(config.da3_output_dir) / "cameras.json").read_text(encoding="utf-8"))
    camera_frames = cameras["frames"]
    if len(frames) != len(camera_frames):
        raise ValueError("frames.json and cameras.json counts differ.")
    init_points = Path(config.pointcloud_dir) / "init_points.ply"
    if not init_points.exists():
        raise FileNotFoundError(f"Missing {init_points}")
    K0 = np.asarray(camera_frames[0]["intrinsic"], dtype=np.float32)
    transform_frames = []
    for idx, item in enumerate(camera_frames):
        src = Path(item["image_path"])
        dst = images_dir / f"frame_{idx+1:06d}{src.suffix.lower()}"
        if not dst.exists() or config.overwrite:
            shutil.copy2(src, dst)
        extrinsics_type = item.get("extrinsics_type", cameras.get("extrinsics_type", config.extrinsics_type))
        c2w = make_c2w(item["extrinsic"], extrinsics_type)
        transform_frames.append({"file_path": f"images/{dst.name}", "transform_matrix": c2w.tolist()})
    shutil.copy2(init_points, points3d_path)
    transforms = {
        "fl_x": float(K0[0, 0]),
        "fl_y": float(K0[1, 1]),
        "cx": float(K0[0, 2]),
        "cy": float(K0[1, 2]),
        "w": int(cameras.get("width", frames[0]["width"])),
        "h": int(cameras.get("height", frames[0]["height"])),
        "camera_model": "OPENCV",
        "frames": transform_frames,
        "metadata": {
            "warning": "This transforms.json may require a custom gsplat loader. If simple_trainer requires COLMAP, TODO: implement DA3-to-COLMAP conversion.",
            "points3D_ply": str(points3d_path),
        },
    }
    transforms_path.write_text(json.dumps(transforms, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {transforms_path}")
    print(f"Saved {points3d_path}")
    print("TODO if needed: adapt simple_trainer.py dataset loader or convert to COLMAP format.")
    return transforms_path, points3d_path


transforms_json_path, points3d_ply_path = convert_da3_to_gsplat_dataset(config)
"""))

cells.append(md(r"""
# 16. gsplat Training

Builds the training command from `config.gsplat_command_template`.

If `simple_trainer.py` cannot read this dataset format, the cell stops with a TODO instead of inventing unsupported CLI arguments.
"""))

cells.append(code(r"""
# ============================================================
# 16. gsplat training
# ============================================================

import time
import os


def find_train_ply_candidates(train_dir):
    train_dir = Path(train_dir)
    patterns = ["**/point_cloud.ply", "**/point_cloud_*.ply", "**/gaussians.ply", "**/*.ply"]
    candidates = []
    for pattern in patterns:
        for path in train_dir.glob(pattern):
            if path.is_file() and path not in candidates:
                candidates.append(path)
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)


def build_gsplat_training_command(config):
    dataset_dir = Path(config.gsplat_dataset_dir)
    train_dir = Path(config.gsplat_train_dir)
    repo_dir = Path(config.gsplat_repo_dir)
    points_ply = dataset_dir / "points3D.ply"
    train_dir.mkdir(parents=True, exist_ok=True)
    if getattr(config, "vggt_fallback_policy", "off") == "prefer_vggt_colmap":
        sparse_dir = Path(config.vggt_scene_dir) / "sparse"
        required = [sparse_dir / "cameras.bin", sparse_dir / "images.bin", sparse_dir / "points3D.bin"]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(f"VGGT COLMAP fallback requested, but files are missing: {missing}")
        return (
            f"python {repo_dir}/examples/simple_trainer.py default "
            f"--data_factor 1 "
            f"--data_dir {Path(config.vggt_scene_dir)} "
            f"--result_dir {train_dir}"
        )
    return config.gsplat_command_template.format(
        dataset_dir=str(dataset_dir),
        train_dir=str(train_dir),
        iterations=int(config.gsplat_iterations),
        points_ply=str(points_ply),
        repo_dir=str(repo_dir),
    )


def validate_gsplat_training_route(config, inspection):
    if getattr(config, "vggt_fallback_policy", "off") == "prefer_vggt_colmap":
        print("Using VGGT COLMAP fallback route for gsplat training.")
        return True
    command = config.gsplat_command_template
    uses_simple_trainer = "simple_trainer.py" in command
    if uses_simple_trainer and inspection.get("mentions_colmap") and not inspection.get("mentions_transforms_json"):
        msg = (
            "The current gsplat simple_trainer.py appears COLMAP-oriented and does not directly read transforms.json.\n"
            "TODO: implement DA3-to-COLMAP conversion, add a verified DA3 Dataset/Parser, or use a custom trainer.\n"
            "Do not invent missing gsplat CLI args."
        )
        if config.dry_run:
            print("[DRY RUN WARNING]\n" + msg)
            return False
        raise RuntimeError(msg)
    return True


def run_gsplat_training(config, inspection):
    train_dir = Path(config.gsplat_train_dir)
    repo_dir = Path(config.gsplat_repo_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    command = build_gsplat_training_command(config)
    (train_dir / "train_command.txt").write_text(command + "\n", encoding="utf-8")
    print("gsplat training command:")
    print(command)
    print("OOM mitigation: reduce gsplat_iterations, max_frames, resize_width/resize_height, or max_points.")
    route_ok = validate_gsplat_training_route(config, inspection)
    if config.dry_run:
        print("config.dry_run=True. Training will not run.")
        return {"dry_run": True, "route_ok": route_ok, "ply_candidates": [str(p) for p in find_train_ply_candidates(train_dir)]}
    log_path = train_dir / "train.log"
    start = time.time()
    recent = []
    training_cwd = repo_dir
    training_env = os.environ.copy()
    py_paths = [str(repo_dir), str(repo_dir / "examples")]
    existing_pythonpath = training_env.get("PYTHONPATH", "")
    training_env["PYTHONPATH"] = ":".join(py_paths + ([existing_pythonpath] if existing_pythonpath else []))
    if inspection.get("install_mode") == "pypi_fallback":
        print("Using source repo on PYTHONPATH so simple_trainer.py and gsplat package come from the same checkout.")
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=str(training_cwd), env=training_env, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            log.write(line)
            log.flush()
            recent.append(line.rstrip())
            recent = recent[-25:]
            if any(k in line.lower() for k in ["step", "iter", "loss", "psnr", "save", "error", "cuda", "out of memory"]):
                print(line.rstrip())
        rc = process.wait()
    if rc != 0:
        print("\n".join(recent))
        raise RuntimeError("gsplat training failed. Check train.log and verified CLI args.")
    candidates = find_train_ply_candidates(train_dir)
    (train_dir / "ply_candidates.json").write_text(json.dumps([str(p) for p in candidates], indent=2), encoding="utf-8")
    return {"dry_run": False, "elapsed_sec": time.time() - start, "ply_candidates": [str(p) for p in candidates]}


training_result = run_gsplat_training(config, gsplat_trainer_inspection)
"""))

cells.append(md(r"""
# 17. Final Gaussian PLY Export / Download / Debug Utilities

Copies the latest trained PLY candidate to:

```text
outputs/{job_id}/result/gaussians.ply
```

Important:

```text
gaussians.ply is a 3DGS Gaussian PLY, not a triangle mesh PLY.
```
"""))

cells.append(code(r"""
# ============================================================
# 17. Final Gaussian PLY export / download / debug utilities
# ============================================================

import zipfile

FINAL_GAUSSIAN_NOTE = "This is a 3DGS Gaussian PLY, not a triangle mesh PLY."


def debug_pipeline_state(config):
    def load_json(path):
        path = Path(path)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    frames = load_json(Path(config.output_dir) / "frames.json")
    cameras = load_json(Path(config.da3_output_dir) / "cameras.json")
    transforms = load_json(Path(config.gsplat_dataset_dir) / "transforms.json")
    depth_files = sorted((Path(config.da3_output_dir) / "depth").glob("depth_*.npy"))
    conf_files = sorted((Path(config.da3_output_dir) / "confidence").glob("conf_*.npy"))
    result_ply = Path(config.result_ply_path)
    report = {
        "frame_count": len(frames) if isinstance(frames, list) else None,
        "depth_count": len(depth_files),
        "confidence_count": len(conf_files),
        "cameras_json_frame_count": len(cameras.get("frames", [])) if isinstance(cameras, dict) else None,
        "init_points_ply_exists": (Path(config.pointcloud_dir) / "init_points.ply").exists(),
        "transforms_json_frame_count": len(transforms.get("frames", [])) if isinstance(transforms, dict) else None,
        "train_ply_candidates": [str(p) for p in find_train_ply_candidates(config.gsplat_train_dir)],
        "result_ply_exists": result_ply.exists(),
        "result_ply_size_mb": result_ply.stat().st_size / (1024 * 1024) if result_ply.exists() else None,
        "note": FINAL_GAUSSIAN_NOTE,
    }
    print(json.dumps(report, indent=2))
    (Path(config.output_dir) / "debug_pipeline_state.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def find_latest_ply_candidate(train_dir):
    candidates = find_train_ply_candidates(train_dir)
    if not candidates:
        raise FileNotFoundError("No .ply candidates found in train_dir. Training/export likely did not produce a PLY.")
    for path in candidates:
        print(f"candidate: {path} ({path.stat().st_size / (1024*1024):.2f} MB)")
    return candidates[0]


def export_final_gaussian_ply(config):
    result_dir = Path(config.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    source = find_latest_ply_candidate(config.gsplat_train_dir)
    result = Path(config.result_ply_path)
    expected = Path(config.project_root) / "outputs" / config.job_id / "result" / "gaussians.ply"
    if result != expected:
        raise ValueError(f"result_ply_path must be {expected}, got {result}")
    shutil.copy2(source, result)
    frames = json.loads((Path(config.output_dir) / "frames.json").read_text(encoding="utf-8"))
    metadata = {
        "input_video_path": str(config.input_video_path),
        "job_id": str(config.job_id),
        "frame_count": len(frames),
        "da3_repo_url": str(config.da3_repo_url),
        "da3_model_name": str(config.da3_model_name),
        "da3_checkpoint_or_hf_id": str(config.da3_checkpoint_or_hf_id),
        "gsplat_repo_url": str(config.gsplat_repo_url),
        "confidence_threshold": float(config.confidence_threshold),
        "gsplat_iterations": int(config.gsplat_iterations),
        "source_ply_path": str(source),
        "result_ply_path": str(result),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": FINAL_GAUSSIAN_NOTE,
        "blender_warning": "This PLY stores 3D Gaussian primitives. It is not a triangle mesh.",
    }
    metadata_path = result_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Final Gaussian PLY: {result}")
    print(FINAL_GAUSSIAN_NOTE)
    return result, metadata_path


def package_and_download_result(config):
    result_dir = Path(config.result_dir)
    result_ply = Path(config.result_ply_path)
    metadata = result_dir / "metadata.json"
    cameras = Path(config.da3_output_dir) / "cameras.json"
    config_snapshot = result_dir / "config_snapshot.json"
    train_log = Path(config.gsplat_train_dir) / "train.log"
    config_snapshot.write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8")
    if not train_log.exists():
        train_log = result_dir / "train.log"
        train_log.write_text("train.log was not found at packaging time.\n", encoding="utf-8")
    for required in [result_ply, metadata, cameras, config_snapshot, train_log]:
        if not required.exists():
            raise FileNotFoundError(required)
    zip_path = result_dir / f"{config.job_id}_3dgs_gaussian_result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(result_ply, "gaussians.ply")
        zf.write(metadata, "metadata.json")
        zf.write(cameras, "cameras.json")
        zf.write(config_snapshot, "config_snapshot.json")
        zf.write(train_log, "train.log")
    print(f"ZIP: {zip_path}")
    print(FINAL_GAUSSIAN_NOTE)
    if config.use_drive:
        print(f"Drive result dir: {result_dir}")
    try:
        from google.colab import files
        files.download(str(zip_path))
    except Exception as exc:
        print(f"Automatic download failed: {repr(exc)}")
        print(f"Download manually from: {zip_path}")
    return zip_path


debug_report = debug_pipeline_state(config)
final_ply_path, final_metadata_path = export_final_gaussian_ply(config)
result_zip_path = package_and_download_result(config)
"""))


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "A100",
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.x",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH.resolve()}")
