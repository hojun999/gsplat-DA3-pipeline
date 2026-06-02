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
8. Candidate frame extraction and RGB keyframe selection
9. DA3 repository setup
10. DA3 inference adapter
11. Initial DA3 inference, geometry filtering, and refined DA3 inference
12. Refined DA3 result validation
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
GSPLAT_COMPATIBLE_REVISION = "v1.5.3"
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
    candidate_frames_dir: str = ""
    frames_dir: str = ""
    reports_dir: str = ""
    da3_initial_output_dir: str = ""
    da3_output_dir: str = ""
    pointcloud_dir: str = ""
    gsplat_dataset_dir: str = ""
    gsplat_train_dir: str = ""
    result_dir: str = ""
    result_ply_path: str = ""

    candidate_frame_fps: float = 4.0
    max_candidate_frames: int = 800
    max_frames: int = 300
    resize_width: Optional[int] = 1008
    resize_height: Optional[int] = 756
    overwrite: bool = False
    resume: bool = True
    rgb_min_blur_score: float = 55.0
    rgb_max_similarity: float = 0.98
    rgb_min_time_gap_sec: float = 0.15
    rgb_max_time_gap_sec: float = 0.75

    da3_repo_url: str = DA3_REPO_URL
    da3_repo_dir: str = "/content/depth-anything-3"
    da3_repo_revision: str = ""
    da3_model_name: str = "DepthAnything3"
    da3_checkpoint_or_hf_id: str = "depth-anything/DA3NESTED-GIANT-LARGE"
    da3_device: str = "cuda"
    da3_use_ray_pose: bool = False
    da3_mock_mode: bool = False
    da3_auto_restart_after_dependency_install: bool = True
    confidence_threshold: float = 0.70
    geometry_depth_stride: int = 4
    geometry_neighbor_window: int = 3
    geometry_min_overlap_ratio: float = 0.15
    geometry_max_median_relative_depth_error: float = 0.20
    geometry_pose_mad_multiplier: float = 3.0
    geometry_max_prune_ratio: float = 0.35
    geometry_min_final_frames: int = 24
    geometry_refined_max_anomaly_ratio: float = 0.05

    vggt_repo_url: str = VGGT_REPO_URL
    vggt_repo_dir: str = "/content/vggt"
    vggt_repo_revision: str = ""
    vggt_fallback_policy: Literal["off", "compare_only", "prefer_vggt_colmap"] = "compare_only"
    vggt_use_ba: bool = False
    vggt_fail_open: bool = True
    vggt_auto_fallback_on_da3_failure: bool = True
    vggt_confidence_thresholds: List[float] = field(default_factory=lambda: [5.0, 1.0, 0.2])
    vggt_refine_sparse_scene: bool = True
    vggt_sparse_axis_clip_quantile: float = 0.002
    vggt_sparse_path_distance_mad_multiplier: float = 6.0
    vggt_sparse_min_filtered_points: int = 10_000
    vggt_sparse_min_retained_ratio: float = 0.25
    vggt_pose_step_mad_multiplier: float = 6.0
    vggt_max_pose_jump_ratio: float = 0.15
    vggt_scene_dir: str = ""

    point_stride: int = 4
    point_multiview_filter_enabled: bool = True
    point_multiview_neighbor_window: int = 3
    point_multiview_min_support_views: int = 2
    point_multiview_max_relative_depth_error: float = 0.15
    max_points: int = 1_000_000
    voxel_size: float = 0.01
    wall_plane_filter_enabled: bool = True
    pointcloud_use_wall_filter_result: bool = False
    wall_plane_min_luma: float = 175.0
    wall_plane_max_chroma: float = 35.0
    wall_plane_distance_threshold: float = 0.035
    wall_plane_project_distance: float = 0.10
    wall_plane_min_inliers: int = 4000
    wall_plane_max_planes: int = 6
    extrinsics_type: Literal["w2c", "c2w"] = "w2c"
    camera_convention: str = "opencv"

    gsplat_repo_url: str = GSPLAT_REPO_URL
    gsplat_repo_revision: str = GSPLAT_COMPATIBLE_REVISION
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
        "gsplat_repo_revision": GSPLAT_COMPATIBLE_REVISION,
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
        self.candidate_frames_dir = str(base / "work" / "candidate_frames")
        self.frames_dir = str(base / "work" / "frames")
        self.reports_dir = str(base / "reports")
        self.da3_initial_output_dir = str(base / "work" / "da3_initial")
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
    job_id="corridor_strict_multiview_test_001",
    candidate_frame_fps=4.0,
    max_candidate_frames=800,
    max_frames=300,
    resize_width=1008,
    resize_height=756,
    overwrite=False,
    resume=True,
    confidence_threshold=0.70,
    geometry_depth_stride=4,
    geometry_neighbor_window=3,
    geometry_max_median_relative_depth_error=0.20,
    geometry_pose_mad_multiplier=3.0,
    geometry_refined_max_anomaly_ratio=0.05,
    point_stride=3,
    point_multiview_filter_enabled=True,
    point_multiview_neighbor_window=3,
    point_multiview_min_support_views=2,
    point_multiview_max_relative_depth_error=0.15,
    max_points=1_200_000,
    voxel_size=0.008,
    wall_plane_filter_enabled=True,
    pointcloud_use_wall_filter_result=False,
    wall_plane_min_luma=170.0,
    wall_plane_max_chroma=38.0,
    wall_plane_distance_threshold=0.035,
    wall_plane_project_distance=0.10,
    wall_plane_min_inliers=2500,
    wall_plane_max_planes=8,
    vggt_fallback_policy="compare_only",
    vggt_use_ba=False,
    vggt_fail_open=True,
    vggt_auto_fallback_on_da3_failure=True,
    vggt_confidence_thresholds=[5.0, 1.0, 0.2],
    vggt_refine_sparse_scene=True,
    vggt_sparse_axis_clip_quantile=0.002,
    vggt_sparse_path_distance_mad_multiplier=6.0,
    vggt_sparse_min_filtered_points=10_000,
    vggt_sparse_min_retained_ratio=0.25,
    vggt_pose_step_mad_multiplier=6.0,
    vggt_max_pose_jump_ratio=0.15,
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
    Path(config.candidate_frames_dir),
    Path(config.frames_dir),
    Path(config.reports_dir),
    Path(config.da3_initial_output_dir),
    Path(config.da3_initial_output_dir) / "depth",
    Path(config.da3_initial_output_dir) / "confidence",
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
# 8. Candidate Frame Extraction And RGB Keyframe Selection

Extracts a generous RGB candidate set, removes blurry and near-duplicate frames, preserves temporal coverage, copies selected keyframes to `work/frames`, and writes `frames.json`.
"""))

cells.append(code(r"""
# ============================================================
# 8. Candidate frame extraction and RGB keyframe selection
# ============================================================

import math
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def read_frame_with_fallback(cap, target_index, source_frame_count, search_radius=8):
    target_index = int(target_index)
    source_frame_count = int(source_frame_count)
    offsets = [0]
    for delta in range(1, int(search_radius) + 1):
        offsets.extend([-delta, delta])
    tried = []
    for offset in offsets:
        frame_index = target_index + offset
        if frame_index < 0 or frame_index >= source_frame_count:
            continue
        tried.append(frame_index)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = cap.read()
        if ok and bgr is not None:
            return frame_index, bgr
    raise ValueError(
        f"Failed to read frame {target_index} and nearby frames {tried}. "
        "The video may be truncated or corrupted around this position."
    )


def extract_candidate_video_frames(config, input_video_path):
    candidate_dir = Path(config.candidate_frames_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidates_json_path = Path(config.output_dir) / "candidate_frames.json"
    if config.resume and candidates_json_path.exists() and not config.overwrite:
        candidates = json.loads(candidates_json_path.read_text(encoding="utf-8"))
        if candidates and all(Path(item["image_path"]).exists() for item in candidates):
            print(f"Using existing candidate frames: {candidates_json_path}")
            return candidates
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {input_video_path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or source_frame_count <= 0:
        cap.release()
        raise ValueError("Invalid video metadata for extraction.")

    duration_sec = source_frame_count / source_fps
    requested_count = max(1, int(math.floor(duration_sec * config.candidate_frame_fps)))
    indices = np.linspace(0, source_frame_count - 1, num=requested_count, dtype=np.int64)
    if config.max_candidate_frames and len(indices) > config.max_candidate_frames:
        keep = np.linspace(0, len(indices) - 1, num=config.max_candidate_frames, dtype=np.int64)
        indices = indices[keep]
    indices = np.unique(indices)

    candidates = []
    decoded_frame_indices = set()
    fallback_count = 0
    for requested_frame_index in tqdm(indices, desc="Extracting candidate frames"):
        try:
            frame_index, bgr = read_frame_with_fallback(cap, requested_frame_index, source_frame_count)
        except ValueError:
            cap.release()
            raise
        if frame_index in decoded_frame_indices:
            continue
        decoded_frame_indices.add(frame_index)
        if frame_index != int(requested_frame_index):
            fallback_count += 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if config.resize_width and config.resize_height:
            rgb = cv2.resize(rgb, (int(config.resize_width), int(config.resize_height)), interpolation=cv2.INTER_AREA)
        height, width = rgb.shape[:2]
        image_path = candidate_dir / f"candidate_{len(candidates) + 1:06d}.jpg"
        Image.fromarray(rgb).save(image_path, quality=95)
        candidates.append({
            "image_path": str(image_path),
            "original_frame_index": int(frame_index),
            "requested_frame_index": int(requested_frame_index),
            "timestamp_sec": float(frame_index) / source_fps,
            "width": int(width),
            "height": int(height),
        })
    cap.release()
    candidates_json_path.write_text(json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(candidates)} candidate frames and {candidates_json_path}")
    if fallback_count:
        print(f"Recovered {fallback_count} unreadable sampled frames using nearby decodable frames.")
    return candidates


def blur_score(image_path):
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read RGB frame for blur scoring: {image_path}")
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def phash_bits(image_path):
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read RGB frame for pHash: {image_path}")
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)[:8, :8]
    threshold = float(np.median(dct[1:, :]))
    return (dct > threshold).reshape(-1)


def phash_similarity(previous_hash, current_hash):
    if previous_hash is None:
        return None
    return float(np.mean(previous_hash == current_hash))


def select_rgb_keyframes(config, candidates):
    selected_candidates = []
    report_rows = []
    last_selected_hash = None
    last_selected_timestamp = None
    for candidate in tqdm(candidates, desc="Selecting RGB keyframes"):
        score = blur_score(candidate["image_path"])
        current_hash = phash_bits(candidate["image_path"])
        similarity = phash_similarity(last_selected_hash, current_hash)
        timestamp = float(candidate["timestamp_sec"])
        elapsed = None if last_selected_timestamp is None else timestamp - last_selected_timestamp
        rejection_reason = None
        if score < float(config.rgb_min_blur_score):
            rejection_reason = "blur_too_low"
        elif elapsed is not None and elapsed < float(config.rgb_min_time_gap_sec):
            rejection_reason = "time_gap_too_small"
        elif (
            similarity is not None
            and similarity >= float(config.rgb_max_similarity)
            and elapsed < float(config.rgb_max_time_gap_sec)
        ):
            rejection_reason = "too_similar"
        if rejection_reason is None:
            selected_candidates.append(candidate)
            last_selected_hash = current_hash
            last_selected_timestamp = timestamp
        report_rows.append({
            "original_frame_index": int(candidate["original_frame_index"]),
            "timestamp_sec": timestamp,
            "blur_score": score,
            "similarity_to_last_selected": similarity,
            "selected_before_target_limit": rejection_reason is None,
            "selected": False,
            "rejection_reason": rejection_reason,
        })

    if config.max_frames and len(selected_candidates) > int(config.max_frames):
        keep = np.linspace(0, len(selected_candidates) - 1, num=int(config.max_frames), dtype=np.int64)
        selected_candidates = [selected_candidates[int(index)] for index in np.unique(keep)]
    selected_original_indices = {int(item["original_frame_index"]) for item in selected_candidates}

    frames_dir = Path(config.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for output_index, candidate in enumerate(selected_candidates, start=1):
        image_path = frames_dir / f"frame_{output_index:06d}.jpg"
        shutil.copy2(candidate["image_path"], image_path)
        frames.append({**candidate, "image_path": str(image_path)})
    for row in report_rows:
        if row["original_frame_index"] in selected_original_indices:
            row["selected"] = True
        elif row["selected_before_target_limit"] and row["rejection_reason"] is None:
            row["rejection_reason"] = "target_frame_limit"
    frames_json_path = Path(config.output_dir) / "frames.json"
    frames_json_path.write_text(json.dumps(frames, indent=2, ensure_ascii=False), encoding="utf-8")
    report = {
        "candidate_frame_count": len(candidates),
        "selected_before_target_limit": sum(row["selected_before_target_limit"] for row in report_rows),
        "selected_frame_count": len(frames),
        "config": {
            "candidate_frame_fps": config.candidate_frame_fps,
            "max_candidate_frames": config.max_candidate_frames,
            "max_frames": config.max_frames,
            "rgb_min_blur_score": config.rgb_min_blur_score,
            "rgb_max_similarity": config.rgb_max_similarity,
            "rgb_min_time_gap_sec": config.rgb_min_time_gap_sec,
            "rgb_max_time_gap_sec": config.rgb_max_time_gap_sec,
        },
        "frames": report_rows,
    }
    report_path = Path(config.reports_dir) / "frame_selection_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Selected {len(frames)} RGB keyframes and wrote {frames_json_path}")
    print(f"RGB keyframe report: {report_path}")
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


candidate_frames = extract_candidate_video_frames(config, input_video_path)
frames = select_rgb_keyframes(config, candidate_frames)
if len(frames) < int(config.geometry_min_final_frames):
    raise RuntimeError(
        f"Too few RGB keyframes before DA3: {len(frames)} < {config.geometry_min_final_frames}. "
        "Relax RGB keyframe thresholds or extract more candidates."
    )
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
    def __init__(self, config, output_dir=None):
        self.config = config
        self.model = None
        self.output_dir = Path(output_dir or config.da3_output_dir)
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
        for stale_path in list(self.depth_dir.glob("depth_*.npy")) + list(self.confidence_dir.glob("conf_*.npy")):
            stale_path.unlink()
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
# 11. Initial DA3 Inference, Geometry Filtering, And Refined DA3 Inference

Runs DA3 on RGB-selected keyframes, rejects geometry outliers using trajectory jumps and cross-view depth reprojection error, then reruns DA3 on the refined frame set.
"""))

cells.append(code(r"""
# ============================================================
# 11. Initial DA3 inference, geometry filtering, and refined DA3 inference
# ============================================================

from functools import lru_cache


def da3_outputs_complete(output_dir, expected_image_paths):
    da3_dir = Path(output_dir)
    cameras_path = da3_dir / "cameras.json"
    if not cameras_path.exists():
        return False
    depth_files = sorted((da3_dir / "depth").glob("depth_*.npy"))
    conf_files = sorted((da3_dir / "confidence").glob("conf_*.npy"))
    expected_count = len(expected_image_paths)
    if len(depth_files) != expected_count or len(conf_files) != expected_count:
        return False
    try:
        cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
        camera_frames = cameras.get("frames", [])
        camera_paths = [str(Path(item["image_path"])) for item in camera_frames]
        expected_paths = [str(Path(path)) for path in expected_image_paths]
        return len(camera_frames) == expected_count and camera_paths == expected_paths
    except Exception:
        return False


def run_da3_pass(config, frame_items, output_dir, pass_name, shared_model=None):
    image_paths = [Path(item["image_path"]) for item in frame_items]
    print(f"[{pass_name}] DA3 images: {len(image_paths)}")
    adapter = DA3Adapter(config, output_dir=output_dir)
    adapter.model = shared_model
    if config.resume and not config.overwrite and da3_outputs_complete(output_dir, image_paths):
        print(f"[{pass_name}] Complete matching DA3 outputs exist. Skipping inference.")
        result = adapter.load_existing_result()
    else:
        if adapter.model is None:
            adapter.load_model()
        result = adapter.infer(image_paths)
        result = adapter.save_outputs(result)
    print(f"[{pass_name}] DA3 inference completed: {output_dir}")
    return result, adapter.model


def geometry_c2w(extrinsic):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :4] = np.asarray(extrinsic, dtype=np.float64)
    return np.linalg.inv(matrix)


def rotation_angle_deg(previous_w2c, current_w2c):
    previous_rotation = np.asarray(previous_w2c, dtype=np.float64)[:, :3]
    current_rotation = np.asarray(current_w2c, dtype=np.float64)[:, :3]
    relative = current_rotation @ previous_rotation.T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def robust_upper_threshold(values, multiplier, relative_floor=0.10, absolute_floor=1e-6):
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if len(finite) == 0:
        return float("inf")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    robust_scale = max(mad, abs(median) * float(relative_floor), float(absolute_floor))
    return median + float(multiplier) * robust_scale


def evaluate_da3_geometry(config, output_dir, label):
    cameras_path = Path(output_dir) / "cameras.json"
    cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    frames = cameras.get("frames", [])
    if len(frames) < 2:
        raise RuntimeError(f"[{label}] DA3 geometry evaluation requires at least two frames.")
    manifest = json.loads((Path(config.output_dir) / "frames.json").read_text(encoding="utf-8"))
    timestamps_by_path = {
        str(Path(item["image_path"])): float(item["timestamp_sec"])
        for item in manifest
    }

    c2w_matrices = [geometry_c2w(item["extrinsic"]) for item in frames]
    centers = [matrix[:3, 3] for matrix in c2w_matrices]
    step_distances = [
        float(np.linalg.norm(centers[index] - centers[index - 1]))
        for index in range(1, len(frames))
    ]
    rotation_deltas = [
        rotation_angle_deg(frames[index - 1]["extrinsic"], frames[index]["extrinsic"])
        for index in range(1, len(frames))
    ]
    time_deltas = [
        max(
            timestamps_by_path[str(Path(frames[index]["image_path"]))] - timestamps_by_path[str(Path(frames[index - 1]["image_path"]))],
            1e-6,
        )
        for index in range(1, len(frames))
    ]
    step_speeds = [distance / delta for distance, delta in zip(step_distances, time_deltas)]
    rotation_speeds = [rotation / delta for rotation, delta in zip(rotation_deltas, time_deltas)]
    distance_speed_threshold = robust_upper_threshold(step_speeds, config.geometry_pose_mad_multiplier)
    rotation_speed_threshold = robust_upper_threshold(
        rotation_speeds,
        config.geometry_pose_mad_multiplier,
        absolute_floor=1.0,
    )

    @lru_cache(maxsize=8)
    def load_depth_conf(index):
        item = frames[index]
        return (
            np.load(item["depth_path"]).astype(np.float32),
            np.load(item["confidence_path"]).astype(np.float32),
        )

    def compare_depth_pair(source_index, target_index):
        source_item = frames[source_index]
        target_item = frames[target_index]
        source_depth, source_conf = load_depth_conf(source_index)
        target_depth, target_conf = load_depth_conf(target_index)
        stride = max(1, int(config.geometry_depth_stride))
        height, width = source_depth.shape
        ys = np.arange(0, height, stride, dtype=np.int32)
        xs = np.arange(0, width, stride, dtype=np.int32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        sampled_depth = source_depth[grid_y, grid_x]
        sampled_conf = source_conf[grid_y, grid_x]
        valid_source = (
            np.isfinite(sampled_depth)
            & (sampled_depth > 0)
            & np.isfinite(sampled_conf)
            & (sampled_conf >= float(config.confidence_threshold))
        )
        valid_source_count = int(np.count_nonzero(valid_source))
        if valid_source_count == 0:
            return {"target_index": target_index, "overlap_ratio": 0.0, "median_relative_depth_error": None}
        intrinsic = np.asarray(source_item["intrinsic"], dtype=np.float64)
        z = sampled_depth[valid_source].astype(np.float64)
        u = grid_x[valid_source].astype(np.float64)
        v = grid_y[valid_source].astype(np.float64)
        points_source = np.stack([
            (u - intrinsic[0, 2]) / intrinsic[0, 0] * z,
            (v - intrinsic[1, 2]) / intrinsic[1, 1] * z,
            z,
            np.ones_like(z),
        ], axis=1)
        points_world = (c2w_matrices[source_index] @ points_source.T).T
        target_w2c = np.eye(4, dtype=np.float64)
        target_w2c[:3, :4] = np.asarray(target_item["extrinsic"], dtype=np.float64)
        points_target = (target_w2c @ points_world.T).T[:, :3]
        target_z = points_target[:, 2]
        target_intrinsic = np.asarray(target_item["intrinsic"], dtype=np.float64)
        forward = target_z > 1e-8
        projected_u = np.zeros(len(target_z), dtype=np.int64)
        projected_v = np.zeros(len(target_z), dtype=np.int64)
        projected_u[forward] = np.rint(
            target_intrinsic[0, 0] * points_target[forward, 0] / target_z[forward] + target_intrinsic[0, 2]
        ).astype(np.int64)
        projected_v[forward] = np.rint(
            target_intrinsic[1, 1] * points_target[forward, 1] / target_z[forward] + target_intrinsic[1, 2]
        ).astype(np.int64)
        target_height, target_width = target_depth.shape
        inside = (
            forward
            & (projected_u >= 0)
            & (projected_u < target_width)
            & (projected_v >= 0)
            & (projected_v < target_height)
        )
        if not np.any(inside):
            return {"target_index": target_index, "overlap_ratio": 0.0, "median_relative_depth_error": None}
        projected_u = projected_u[inside]
        projected_v = projected_v[inside]
        target_z = target_z[inside]
        observed_depth = target_depth[projected_v, projected_u]
        observed_conf = target_conf[projected_v, projected_u]
        valid_target = (
            np.isfinite(observed_depth)
            & (observed_depth > 0)
            & np.isfinite(observed_conf)
            & (observed_conf >= float(config.confidence_threshold))
        )
        overlap_ratio = float(np.count_nonzero(valid_target) / valid_source_count)
        if not np.any(valid_target):
            median_error = None
        else:
            relative_error = np.abs(target_z[valid_target] - observed_depth[valid_target]) / np.maximum(observed_depth[valid_target], 1e-6)
            median_error = float(np.median(relative_error))
        return {
            "target_index": target_index,
            "overlap_ratio": overlap_ratio,
            "median_relative_depth_error": median_error,
        }

    frame_reports = []
    anomaly_indices = []
    for index, item in enumerate(tqdm(frames, desc=f"[{label}] Evaluating DA3 geometry")):
        comparisons = []
        lower = max(0, index - int(config.geometry_neighbor_window))
        upper = min(len(frames), index + int(config.geometry_neighbor_window) + 1)
        for target_index in range(lower, upper):
            if target_index != index:
                comparisons.append(compare_depth_pair(index, target_index))
        eligible_errors = [
            comparison["median_relative_depth_error"]
            for comparison in comparisons
            if comparison["overlap_ratio"] >= float(config.geometry_min_overlap_ratio)
            and comparison["median_relative_depth_error"] is not None
        ]
        median_depth_error = float(np.median(eligible_errors)) if eligible_errors else None
        incoming_distance = step_distances[index - 1] if index > 0 else None
        incoming_rotation = rotation_deltas[index - 1] if index > 0 else None
        incoming_time_delta = time_deltas[index - 1] if index > 0 else None
        incoming_step_speed = step_speeds[index - 1] if index > 0 else None
        incoming_rotation_speed = rotation_speeds[index - 1] if index > 0 else None
        translation_jump = bool(index > 0 and incoming_step_speed > distance_speed_threshold)
        rotation_jump = bool(index > 0 and incoming_rotation_speed > rotation_speed_threshold)
        outgoing_rotation_jump = bool(
            index < len(rotation_speeds)
            and rotation_speeds[index] > rotation_speed_threshold
        )
        pose_jump = bool(
            index > 0
            and (translation_jump or rotation_jump)
        )
        depth_inconsistent = bool(
            len(eligible_errors) >= 2
            and median_depth_error > float(config.geometry_max_median_relative_depth_error)
        )
        anomalous = bool(
            index > 0
            and (
                translation_jump
                or depth_inconsistent
                or (rotation_jump and outgoing_rotation_jump)
            )
        )
        if anomalous:
            anomaly_indices.append(index)
        frame_reports.append({
            "index": index,
            "image_path": item["image_path"],
            "incoming_step_distance": incoming_distance,
            "incoming_rotation_deg": incoming_rotation,
            "incoming_time_delta_sec": incoming_time_delta,
            "incoming_step_speed": incoming_step_speed,
            "incoming_rotation_speed_deg_per_sec": incoming_rotation_speed,
            "translation_jump": translation_jump,
            "rotation_jump": rotation_jump,
            "pose_jump": pose_jump,
            "depth_inconsistent": depth_inconsistent,
            "median_relative_depth_error": median_depth_error,
            "eligible_depth_neighbor_count": len(eligible_errors),
            "comparisons": comparisons,
            "anomalous": anomalous,
        })

    report = {
        "label": label,
        "frame_count": len(frames),
        "anomaly_count": len(anomaly_indices),
        "anomaly_ratio": len(anomaly_indices) / len(frames),
        "anomaly_indices": anomaly_indices,
        "step_speed_threshold": distance_speed_threshold,
        "rotation_speed_threshold_deg_per_sec": rotation_speed_threshold,
        "config": {
            "geometry_depth_stride": config.geometry_depth_stride,
            "geometry_neighbor_window": config.geometry_neighbor_window,
            "geometry_min_overlap_ratio": config.geometry_min_overlap_ratio,
            "geometry_max_median_relative_depth_error": config.geometry_max_median_relative_depth_error,
            "geometry_pose_mad_multiplier": config.geometry_pose_mad_multiplier,
        },
        "frames": frame_reports,
    }
    report_path = Path(config.reports_dir) / f"da3_geometry_{label}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{label}] Geometry report: {report_path}")
    print(json.dumps({key: value for key, value in report.items() if key != "frames"}, indent=2))
    return report


def write_refined_frames(config, frame_items, geometry_report):
    anomaly_indices = set(geometry_report["anomaly_indices"])
    refined_frames = [item for index, item in enumerate(frame_items) if index not in anomaly_indices]
    prune_ratio = 1.0 - len(refined_frames) / len(frame_items)
    if prune_ratio > float(config.geometry_max_prune_ratio):
        raise RuntimeError(
            f"DA3 geometry pruning rejected too many frames: {prune_ratio:.1%} > {config.geometry_max_prune_ratio:.1%}. "
            "Inspect the initial trajectory and depth consistency report."
        )
    if len(refined_frames) < int(config.geometry_min_final_frames):
        raise RuntimeError(
            f"Too few geometry-refined frames: {len(refined_frames)} < {config.geometry_min_final_frames}."
        )
    frames_json_path = Path(config.output_dir) / "frames.json"
    frames_json_path.write_text(json.dumps(refined_frames, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "initial_rgb_keyframe_count": len(frame_items),
        "geometry_rejected_count": len(frame_items) - len(refined_frames),
        "geometry_prune_ratio": prune_ratio,
        "refined_frame_count": len(refined_frames),
        "frames_json_path": str(frames_json_path),
    }
    summary_path = Path(config.reports_dir) / "da3_geometry_selection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return refined_frames


try:
    print(f"DA3 mock mode: {config.da3_mock_mode}")
    print(f"DA3 checkpoint/HF ID: {config.da3_checkpoint_or_hf_id}")
    initial_frame_items = json.loads((Path(config.output_dir) / "frames.json").read_text(encoding="utf-8"))
    initial_da3_result, loaded_da3_model = run_da3_pass(
        config,
        initial_frame_items,
        config.da3_initial_output_dir,
        "initial",
    )
    initial_geometry_report = evaluate_da3_geometry(config, config.da3_initial_output_dir, "initial")
    refined_frame_items = write_refined_frames(config, initial_frame_items, initial_geometry_report)
    da3_result, loaded_da3_model = run_da3_pass(
        config,
        refined_frame_items,
        config.da3_output_dir,
        "refined",
        shared_model=loaded_da3_model,
    )
    print("Two-pass DA3 inference execution completed.")
except Exception as exc:
    print("Two-pass DA3 inference failed.")
    print("Check DA3 repo install, checkpoint/HF ID, depth_anything_3.api import, and prediction fields.")
    traceback.print_exc()
    raise
"""))

cells.append(md(r"""
# 12. Refined DA3 Result Validation

Validates refined depth, confidence, camera outputs, and geometry consistency, then previews sample depth maps. The pipeline stops before gsplat when the refined result still contains too many anomalies.
"""))

cells.append(code(r"""
# ============================================================
# 12. Refined DA3 result validation
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
    refined_geometry_report = evaluate_da3_geometry(config, config.da3_output_dir, "refined")
    if refined_geometry_report["anomaly_ratio"] > float(config.geometry_refined_max_anomaly_ratio):
        message = (
            "Refined DA3 geometry still contains too many anomalies: "
            f"{refined_geometry_report['anomaly_ratio']:.1%} > {config.geometry_refined_max_anomaly_ratio:.1%}."
        )
        if getattr(config, "vggt_auto_fallback_on_da3_failure", True):
            config.vggt_fallback_policy = "prefer_vggt_colmap"
            print("[WARN]", message)
            print("[WARN] Switching automatically to VGGT COLMAP fallback. Continue with section 12.5.")
        else:
            raise RuntimeError(message + " Inspect the refined trajectory and depth consistency report before gsplat training.")
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
    run_shell(
        f"{sys.executable} -m pip install pycolmap==3.10.0 pyceres==2.3 hydra-core omegaconf "
        "\"git+https://github.com/jytime/LightGlue.git#egg=lightglue\"",
        cwd=repo_dir,
    )
    return repo_dir


def ply_vertex_count(path):
    path = Path(path)
    if not path.exists():
        return 0
    with path.open("rb") as f:
        for raw_line in f:
            line = raw_line.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return 0


def validate_vggt_sparse_scene(sparse_dir):
    sparse_dir = Path(sparse_dir)
    expected = [
        sparse_dir / "cameras.bin",
        sparse_dir / "images.bin",
        sparse_dir / "points3D.bin",
        sparse_dir / "points.ply",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    point_count = ply_vertex_count(sparse_dir / "points.ply")
    return missing, point_count


def run_vggt_fallback_probe(config):
    policy = getattr(config, "vggt_fallback_policy", "off")
    if policy == "off":
        return write_vggt_status(config, {"status": "skipped", "reason": "vggt_fallback_policy=off"})
    try:
        scene_dir, copied = prepare_vggt_scene(config)
        sparse_dir = scene_dir / "sparse"
        if config.resume and sparse_dir.exists() and any(sparse_dir.glob("**/*")) and not config.overwrite:
            missing, point_count = validate_vggt_sparse_scene(sparse_dir)
            if not missing and point_count > 0:
                return write_vggt_status(config, {
                    "status": "reused",
                    "image_count": len(copied),
                    "sparse_dir": str(sparse_dir),
                    "point_count": point_count,
                })
            print(f"Existing VGGT sparse scene is incomplete or empty; rebuilding it. missing={missing}, point_count={point_count}")
        repo_dir = setup_vggt_repo(config)
        missing = []
        point_count = 0
        selected_threshold = None
        for threshold in config.vggt_confidence_thresholds:
            if sparse_dir.exists():
                shutil.rmtree(sparse_dir)
            command = f"{sys.executable} demo_colmap.py --scene_dir={scene_dir} --conf_thres_value={float(threshold)}"
            if config.vggt_use_ba:
                command += " --use_ba"
            print("VGGT fallback command:")
            print(command)
            run_shell(command, cwd=repo_dir)
            missing, point_count = validate_vggt_sparse_scene(sparse_dir)
            if not missing and point_count > 0:
                selected_threshold = float(threshold)
                break
            print(f"VGGT sparse export was empty or incomplete at confidence threshold {threshold}: missing={missing}, point_count={point_count}")
        status = {
            "status": "ok" if not missing and point_count > 0 else "incomplete",
            "image_count": len(copied),
            "sparse_dir": str(sparse_dir),
            "missing": missing,
            "point_count": point_count,
            "selected_confidence_threshold": selected_threshold,
            "note": "VGGT COLMAP export is a comparison/fallback route. DA3 remains primary unless vggt_fallback_policy='prefer_vggt_colmap'.",
        }
        if (missing or point_count <= 0) and (not config.vggt_fail_open or policy == "prefer_vggt_colmap"):
            raise RuntimeError(f"VGGT COLMAP export is incomplete or empty: missing={missing}, point_count={point_count}")
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
# 12.6 VGGT Sparse Scene Refinement

Validates the VGGT camera trajectory and writes a conservative `sparse/training_points.ply` for gsplat initialization.

The original VGGT `points.ply` remains untouched for comparison. The refined point cloud removes non-finite points, robust spatial outliers, and points unusually far from the estimated camera path. This limits wall-penetrating depth tails before Gaussian densification.
"""))

cells.append(code(r"""
# ============================================================
# 12.6 VGGT sparse scene refinement
# ============================================================

def vggt_robust_upper_threshold(values, multiplier, absolute_floor=0.0):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float(absolute_floor)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(float(absolute_floor), median + float(multiplier) * max(mad, 1e-12))


def load_ply_xyz_rgb(path):
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"].data
    names = set(vertex.dtype.names or [])
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    if {"red", "green", "blue"}.issubset(names):
        colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.uint8)
    else:
        colors = np.full((len(points), 3), 127, dtype=np.uint8)
    return points, colors


def write_ply_xyz_rgb(path, points, colors):
    from plyfile import PlyData, PlyElement

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex = np.empty(
        len(points),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))


def load_vggt_camera_centers(sparse_dir):
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    centers = []
    names = []
    for image_id in sorted(reconstruction.images):
        image = reconstruction.images[image_id]
        cam_from_world = image.cam_from_world
        if callable(cam_from_world):
            cam_from_world = cam_from_world()
        if cam_from_world is None:
            continue
        matrix = np.asarray(cam_from_world.matrix(), dtype=np.float64)
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
        centers.append(-rotation.T @ translation)
        names.append(image.name)
    if not centers:
        raise ValueError(f"No registered cameras in VGGT reconstruction: {sparse_dir}")
    return np.stack(centers), names


def nearest_camera_path_distances(points, camera_centers, chunk_size=20_000):
    distances = []
    for start in range(0, len(points), int(chunk_size)):
        chunk = points[start : start + int(chunk_size)]
        squared = np.sum((chunk[:, None, :] - camera_centers[None, :, :]) ** 2, axis=2)
        distances.append(np.sqrt(np.min(squared, axis=1)))
    return np.concatenate(distances) if distances else np.empty((0,), dtype=np.float64)


def preview_vggt_sparse_refinement(source_points, filtered_points, camera_centers, save_path, max_points=20_000):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    source_ids = rng.choice(len(source_points), size=min(len(source_points), int(max_points)), replace=False)
    filtered_ids = rng.choice(len(filtered_points), size=min(len(filtered_points), int(max_points)), replace=False)
    fig = plt.figure(figsize=(16, 7))
    for position, title, points in [
        (121, "VGGT source points", source_points[source_ids]),
        (122, "VGGT refined training points", filtered_points[filtered_ids]),
    ]:
        ax = fig.add_subplot(position, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.4, alpha=0.35)
        ax.plot(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2], color="red", linewidth=1.5)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.show()


def refine_vggt_sparse_scene(config):
    policy = getattr(config, "vggt_fallback_policy", "off")
    if policy == "off":
        print("VGGT refinement skipped because vggt_fallback_policy=off.")
        return None
    fallback_status = globals().get("vggt_fallback_status", {})
    if policy == "compare_only" and fallback_status.get("status") not in {"ok", "reused"}:
        print(f"Optional VGGT refinement skipped because fallback status is {fallback_status.get('status')!r}.")
        return None
    sparse_dir = Path(config.vggt_scene_dir) / "sparse"
    source_ply = sparse_dir / "points.ply"
    output_ply = sparse_dir / "training_points.ply"
    report_path = Path(config.reports_dir) / "vggt_sparse_refinement_report.json"
    if not source_ply.exists():
        raise FileNotFoundError(f"Missing VGGT source point cloud: {source_ply}")

    points, colors = load_ply_xyz_rgb(source_ply)
    camera_centers, camera_names = load_vggt_camera_centers(sparse_dir)
    if len(points) == 0:
        raise ValueError(f"VGGT source point cloud is empty: {source_ply}")

    finite = np.isfinite(points).all(axis=1)
    finite_points = points[finite]
    finite_colors = colors[finite]
    if len(finite_points) == 0:
        raise ValueError(f"VGGT source point cloud has no finite points: {source_ply}")

    step_distances = np.linalg.norm(np.diff(camera_centers, axis=0), axis=1)
    step_threshold = vggt_robust_upper_threshold(step_distances, config.vggt_pose_step_mad_multiplier)
    pose_jump_indices = (np.flatnonzero(step_distances > step_threshold) + 1).tolist()
    pose_jump_ratio = len(pose_jump_indices) / max(1, len(camera_centers))
    camera_extent = float(np.max(np.linalg.norm(camera_centers - np.mean(camera_centers, axis=0), axis=1)))

    if not getattr(config, "vggt_refine_sparse_scene", True):
        shutil.copy2(source_ply, output_ply)
        print(f"VGGT sparse refinement disabled; copied source point cloud to {output_ply}")
        return output_ply

    center = np.median(finite_points, axis=0)
    _, _, vh = np.linalg.svd(camera_centers - np.median(camera_centers, axis=0), full_matrices=True)
    aligned = (finite_points - center) @ vh.T
    q = float(config.vggt_sparse_axis_clip_quantile)
    if not 0.0 <= q < 0.5:
        raise ValueError(f"vggt_sparse_axis_clip_quantile must be in [0, 0.5), got {q}")
    lower = np.quantile(aligned, q, axis=0)
    upper = np.quantile(aligned, 1.0 - q, axis=0)
    within_axis_bounds = np.all((aligned >= lower) & (aligned <= upper), axis=1)

    path_distances = nearest_camera_path_distances(finite_points, camera_centers)
    path_distance_threshold = vggt_robust_upper_threshold(
        path_distances,
        config.vggt_sparse_path_distance_mad_multiplier,
    )
    near_camera_path = path_distances <= path_distance_threshold
    keep = within_axis_bounds & near_camera_path
    filtered_points = finite_points[keep].astype(np.float32)
    filtered_colors = finite_colors[keep]
    retained_ratio = len(filtered_points) / len(points)

    report = {
        "source_ply": str(source_ply),
        "training_ply": str(output_ply),
        "source_point_count": int(len(points)),
        "finite_point_count": int(len(finite_points)),
        "filtered_point_count": int(len(filtered_points)),
        "retained_ratio": float(retained_ratio),
        "camera_count": int(len(camera_centers)),
        "camera_names": camera_names,
        "camera_step_distance_threshold": float(step_threshold),
        "camera_extent": camera_extent,
        "camera_pose_jump_indices": pose_jump_indices,
        "camera_pose_jump_ratio": float(pose_jump_ratio),
        "path_distance_threshold": float(path_distance_threshold),
        "axis_clip_quantile": q,
        "status": "ok",
    }
    if camera_extent <= 1e-6:
        report["status"] = "rejected_camera_extent"
    elif len(filtered_points) < int(config.vggt_sparse_min_filtered_points):
        report["status"] = "rejected_too_few_points"
    elif retained_ratio < float(config.vggt_sparse_min_retained_ratio):
        report["status"] = "rejected_low_retained_ratio"
    elif pose_jump_ratio > float(config.vggt_max_pose_jump_ratio):
        report["status"] = "rejected_pose_jump_ratio"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "camera_names"}, indent=2))
    if report["status"] != "ok":
        raise RuntimeError(f"VGGT sparse refinement rejected training input: {report['status']}. Inspect {report_path}")
    write_ply_xyz_rgb(output_ply, filtered_points, filtered_colors)
    preview_path = Path(config.reports_dir) / "vggt_sparse_refinement_preview.png"
    preview_vggt_sparse_refinement(finite_points, filtered_points, camera_centers, preview_path)
    print(f"VGGT refined training point cloud: {output_ply}")
    print(f"VGGT sparse refinement preview: {preview_path}")
    return output_ply


vggt_training_points_path = refine_vggt_sparse_scene(config)
"""))

cells.append(md(r"""
# 13. Initial Point Cloud Generation

Creates `outputs/{job_id}/pointcloud/init_points.ply` from DA3 depth and cameras.

Only points supported by neighboring views are kept. The cell also writes wall-filtered and unfiltered comparison PLY files. By default, gsplat uses the conservative unfiltered multi-view result.

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


def extrinsic_3x4_to_w2c_4x4(extrinsic, extrinsics_type):
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :4] = np.asarray(extrinsic, dtype=np.float32)
    if extrinsics_type == "w2c":
        return mat.astype(np.float32)
    if extrinsics_type == "c2w":
        return np.linalg.inv(mat).astype(np.float32)
    raise ValueError(f"Unsupported extrinsics_type={extrinsics_type}")


def pointcloud_generation_signature(config, cameras):
    camera_artifacts = []
    for item in cameras["frames"]:
        depth_path = Path(item["depth_path"])
        confidence_path = Path(item["confidence_path"])
        camera_artifacts.append({
            "image_path": item["image_path"],
            "depth_path": str(depth_path),
            "depth_size": depth_path.stat().st_size,
            "depth_mtime_ns": depth_path.stat().st_mtime_ns,
            "confidence_path": str(confidence_path),
            "confidence_size": confidence_path.stat().st_size,
            "confidence_mtime_ns": confidence_path.stat().st_mtime_ns,
            "intrinsic": item["intrinsic"],
            "extrinsic": item["extrinsic"],
        })
    return {
        "camera_artifacts": camera_artifacts,
        "confidence_threshold": float(config.confidence_threshold),
        "point_stride": int(config.point_stride),
        "point_multiview_filter_enabled": bool(config.point_multiview_filter_enabled),
        "point_multiview_neighbor_window": int(config.point_multiview_neighbor_window),
        "point_multiview_min_support_views": int(config.point_multiview_min_support_views),
        "point_multiview_max_relative_depth_error": float(config.point_multiview_max_relative_depth_error),
        "max_points": int(config.max_points) if config.max_points else None,
        "voxel_size": float(config.voxel_size) if config.voxel_size else None,
        "wall_plane_filter_enabled": bool(config.wall_plane_filter_enabled),
        "pointcloud_use_wall_filter_result": bool(config.pointcloud_use_wall_filter_result),
    }


def filter_points_by_neighbor_support(frame_index, points, colors, cameras, load_depth_conf, config):
    if not getattr(config, "point_multiview_filter_enabled", False) or len(points) == 0:
        return points, colors, {
            "frame_index": frame_index,
            "enabled": False,
            "input_points": int(len(points)),
            "supported_points": int(len(points)),
        }
    camera_frames = cameras["frames"]
    lower = max(0, frame_index - int(config.point_multiview_neighbor_window))
    upper = min(len(camera_frames), frame_index + int(config.point_multiview_neighbor_window) + 1)
    neighbor_indices = [index for index in range(lower, upper) if index != frame_index]
    if not neighbor_indices:
        return points, colors, {
            "frame_index": frame_index,
            "enabled": True,
            "status": "kept_without_neighbors",
            "input_points": int(len(points)),
            "supported_points": int(len(points)),
        }

    support_counts = np.zeros(len(points), dtype=np.int16)
    visible_counts = np.zeros(len(points), dtype=np.int16)
    points_h = np.concatenate([points.astype(np.float32), np.ones((len(points), 1), dtype=np.float32)], axis=1)
    for neighbor_index in neighbor_indices:
        target_item = camera_frames[neighbor_index]
        target_depth, target_confidence = load_depth_conf(neighbor_index)
        extrinsics_type = target_item.get("extrinsics_type", cameras.get("extrinsics_type", config.extrinsics_type))
        target_w2c = extrinsic_3x4_to_w2c_4x4(target_item["extrinsic"], extrinsics_type)
        points_target = (target_w2c @ points_h.T).T[:, :3]
        target_z = points_target[:, 2]
        forward = target_z > 1e-8
        target_intrinsic = np.asarray(target_item["intrinsic"], dtype=np.float32)
        projected_u = np.zeros(len(points), dtype=np.int64)
        projected_v = np.zeros(len(points), dtype=np.int64)
        projected_u[forward] = np.rint(
            target_intrinsic[0, 0] * points_target[forward, 0] / target_z[forward] + target_intrinsic[0, 2]
        ).astype(np.int64)
        projected_v[forward] = np.rint(
            target_intrinsic[1, 1] * points_target[forward, 1] / target_z[forward] + target_intrinsic[1, 2]
        ).astype(np.int64)
        height, width = target_depth.shape
        inside = (
            forward
            & (projected_u >= 0)
            & (projected_u < width)
            & (projected_v >= 0)
            & (projected_v < height)
        )
        inside_indices = np.flatnonzero(inside)
        if len(inside_indices) == 0:
            continue
        observed_depth = target_depth[projected_v[inside], projected_u[inside]]
        observed_confidence = target_confidence[projected_v[inside], projected_u[inside]]
        visible = (
            np.isfinite(observed_depth)
            & (observed_depth > 0)
            & np.isfinite(observed_confidence)
            & (observed_confidence >= float(config.confidence_threshold))
        )
        visible_indices = inside_indices[visible]
        if len(visible_indices) == 0:
            continue
        visible_counts[visible_indices] += 1
        relative_error = np.abs(target_z[visible_indices] - observed_depth[visible]) / np.maximum(observed_depth[visible], 1e-6)
        supported_indices = visible_indices[
            relative_error <= float(config.point_multiview_max_relative_depth_error)
        ]
        support_counts[supported_indices] += 1

    required_support_views = min(
        max(1, int(config.point_multiview_min_support_views)),
        len(neighbor_indices),
    )
    keep = support_counts >= required_support_views
    stats = {
        "frame_index": frame_index,
        "enabled": True,
        "neighbor_indices": neighbor_indices,
        "required_support_views": required_support_views,
        "input_points": int(len(points)),
        "visible_in_any_neighbor": int(np.count_nonzero(visible_counts)),
        "supported_points": int(np.count_nonzero(keep)),
        "rejected_points": int(len(points) - np.count_nonzero(keep)),
    }
    return points[keep], colors[keep], stats


def voxel_downsample_points(points, colors, config, o3d):
    if not config.voxel_size or config.voxel_size <= 0 or len(points) == 0:
        return points, colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((colors.astype(np.float32) / 255.0).astype(np.float64))
    pcd = pcd.voxel_down_sample(float(config.voxel_size))
    return (
        np.asarray(pcd.points, dtype=np.float32),
        np.clip(np.asarray(pcd.colors) * 255, 0, 255).astype(np.uint8),
    )


def write_regular_point_cloud(path, points, colors, o3d):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((colors.astype(np.float32) / 255.0).astype(np.float64))
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def generate_initial_point_cloud(config):
    import open3d as o3d
    from functools import lru_cache

    init_ply_path = POINTCLOUD_OUTPUT_DIR / "init_points.ply"
    init_npz_path = POINTCLOUD_OUTPUT_DIR / "init_points.npz"
    before_wall_ply_path = POINTCLOUD_OUTPUT_DIR / "init_points_multiview_unfiltered.ply"
    wall_filtered_ply_path = POINTCLOUD_OUTPUT_DIR / "init_points_multiview_wall_filtered.ply"
    stats_path = POINTCLOUD_OUTPUT_DIR / "init_points_stats.json"
    cameras = json.loads((Path(config.da3_output_dir) / "cameras.json").read_text(encoding="utf-8"))
    signature = pointcloud_generation_signature(config, cameras)
    if config.resume and init_ply_path.exists() and init_npz_path.exists() and stats_path.exists() and not config.overwrite:
        previous_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if previous_stats.get("generation_signature") == signature:
            print(f"Using existing compatible init point cloud: {init_ply_path}")
            return init_ply_path
        print("Existing init point cloud does not match the current filtering config. Regenerating it.")

    @lru_cache(maxsize=8)
    def load_depth_conf(frame_index):
        item = cameras["frames"][frame_index]
        return (
            np.load(item["depth_path"]).astype(np.float32),
            np.load(item["confidence_path"]).astype(np.float32),
        )

    all_points, all_colors, camera_centers = [], [], []
    support_filter_frames = []
    before_multiview = 0
    after_multiview = 0
    for frame_index, item in enumerate(tqdm(cameras["frames"], desc="Backprojecting and cross-view filtering DA3 depth")):
        rgb = np.asarray(Image.open(item["image_path"]).convert("RGB"), dtype=np.uint8)
        depth, conf = load_depth_conf(frame_index)
        extrinsics_type = item.get("extrinsics_type", cameras.get("extrinsics_type", config.extrinsics_type))
        c2w = extrinsic_3x4_to_c2w_4x4(item["extrinsic"], extrinsics_type)
        camera_centers.append(c2w[:3, 3])
        points, colors = unproject_depth_frame(depth, conf, rgb, item["intrinsic"], item["extrinsic"], extrinsics_type, config)
        before_multiview += len(points)
        points, colors, frame_stats = filter_points_by_neighbor_support(
            frame_index,
            points,
            colors,
            cameras,
            load_depth_conf,
            config,
        )
        after_multiview += len(points)
        support_filter_frames.append(frame_stats)
        if len(points):
            all_points.append(points)
            all_colors.append(colors)
    if not all_points:
        raise ValueError(
            "No valid points generated after multi-view support filtering. "
            "Lower confidence_threshold, point_multiview_min_support_views, or point_multiview_max_relative_depth_error."
        )
    points = np.concatenate(all_points).astype(np.float32)
    colors = np.concatenate(all_colors).astype(np.uint8)
    if config.max_points and len(points) > int(config.max_points):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), size=int(config.max_points), replace=False)
        points, colors = points[idx], colors[idx]
    after_sampling = len(points)

    unfiltered_points, unfiltered_colors = voxel_downsample_points(points, colors, config, o3d)
    write_regular_point_cloud(before_wall_ply_path, unfiltered_points, unfiltered_colors, o3d)
    wall_filtered_points, wall_filtered_colors, wall_filter_stats = refine_low_texture_wall_planes(points, colors, config, o3d)
    wall_filtered_points, wall_filtered_colors = voxel_downsample_points(wall_filtered_points, wall_filtered_colors, config, o3d)
    write_regular_point_cloud(wall_filtered_ply_path, wall_filtered_points, wall_filtered_colors, o3d)

    use_wall_result = bool(config.pointcloud_use_wall_filter_result and config.wall_plane_filter_enabled)
    if use_wall_result:
        final_points, final_colors = wall_filtered_points, wall_filtered_colors
        selected_variant = "multiview_wall_filtered"
    else:
        final_points, final_colors = unfiltered_points, unfiltered_colors
        selected_variant = "multiview_unfiltered"
    write_regular_point_cloud(init_ply_path, final_points, final_colors, o3d)
    np.savez_compressed(init_npz_path, points=final_points, colors=final_colors)
    stats = {
        "note": "init_points.ply is a regular point cloud PLY for 3DGS initialization, not the final Gaussian PLY.",
        "generation_signature": signature,
        "points_before_multiview_filter": int(before_multiview),
        "points_after_multiview_filter": int(after_multiview),
        "points_rejected_by_multiview_filter": int(before_multiview - after_multiview),
        "points_after_sampling": int(after_sampling),
        "points_after_voxel_downsample_without_wall_filter": int(len(unfiltered_points)),
        "wall_plane_filter": wall_filter_stats,
        "points_after_voxel_downsample_with_wall_filter": int(len(wall_filtered_points)),
        "selected_variant": selected_variant,
        "points_in_selected_variant": int(len(final_points)),
        "init_ply_path": str(init_ply_path),
        "multiview_unfiltered_ply_path": str(before_wall_ply_path),
        "multiview_wall_filtered_ply_path": str(wall_filtered_ply_path),
        "support_filter_frames": support_filter_frames,
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in stats.items() if key != "support_filter_frames"}, indent=2))
    print("Compare both point-cloud variants before training:")
    print(f"  unfiltered:    {before_wall_ply_path}")
    print(f"  wall-filtered: {wall_filtered_ply_path}")
    print(f"Selected gsplat initialization variant: {selected_variant}")
    print("This is NOT the final 3DGS Gaussian PLY.")
    visualize_camera_trajectory(np.asarray(camera_centers, dtype=np.float32))
    visualize_point_cloud_preview(unfiltered_points, unfiltered_colors, title="Multi-view point cloud without wall filter")
    visualize_point_cloud_preview(wall_filtered_points, wall_filtered_colors, title="Multi-view point cloud with wall filter")
    return init_ply_path


def visualize_camera_trajectory(camera_centers):
    if len(camera_centers) == 0:
        return
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2], marker="o")
    ax.set_title("Camera trajectory preview")
    plt.show()


def visualize_point_cloud_preview(points, colors, max_plot_points=20000, title="Initial point cloud preview"):
    if len(points) > max_plot_points:
        idx = np.random.default_rng(42).choice(len(points), size=max_plot_points, replace=False)
        points, colors = points[idx], colors[idx]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=np.clip(colors / 255.0, 0, 1), s=0.5)
    ax.set_title(title)
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
    cfg.gsplat_repo_revision = getattr(cfg, "gsplat_repo_revision", "") or "v1.5.3"
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
COMPATIBLE_GSPLAT_REVISION = "v1.5.3"
GSPLAT_REPO_DIR = Path(config.gsplat_repo_dir)


def validate_gsplat_examples_compatibility(examples_dir, requested_revision):
    simple = Path(examples_dir) / "simple_trainer.py"
    if not simple.exists():
        raise FileNotFoundError(f"Missing {simple}")
    source = simple.read_text(encoding="utf-8")
    helper_modules = sorted(set(re.findall(r"^from (gsplat_[A-Za-z0-9_]+) import ", source, flags=re.MULTILINE)))
    missing_helpers = [
        module_name
        for module_name in helper_modules
        if not (Path(examples_dir) / f"{module_name}.py").exists()
    ]
    if missing_helpers:
        raise RuntimeError(
            "The selected gsplat revision has an incomplete examples/simple_trainer.py dependency set. "
            f"Missing local helper modules: {missing_helpers}. "
            f"Requested revision: {requested_revision!r}. "
            f"Use the verified official revision {COMPATIBLE_GSPLAT_REVISION!r}."
        )
    print(f"[OK] gsplat example-local imports verified for revision {requested_revision}.")
    return source


def setup_gsplat_repository(config):
    if config.gsplat_repo_url != EXPECTED_GSPLAT_REPO_URL:
        raise ValueError(f"gsplat repo must be {EXPECTED_GSPLAT_REPO_URL}")
    config.gsplat_repo_revision = getattr(config, "gsplat_repo_revision", "") or COMPATIBLE_GSPLAT_REVISION
    if config.install_gsplat_from_pip:
        run_shell(f"{sys.executable} -m pip install gsplat")
    if GSPLAT_REPO_DIR.exists():
        print(f"gsplat repo exists: {GSPLAT_REPO_DIR}")
        if config.gsplat_repo_revision:
            print(f"Pinned gsplat revision requested: {config.gsplat_repo_revision}. Skipping git pull.")
        elif (GSPLAT_REPO_DIR / ".git").exists() and config.repo_update_policy == "pull_latest":
            run_shell("git pull", cwd=GSPLAT_REPO_DIR, check=False)
        else:
            print("Repo update policy is reuse_existing; skipping git pull for reproducibility.")
    else:
        run_shell(f"git clone {config.gsplat_repo_url} {GSPLAT_REPO_DIR}", cwd="/content")

    if config.gsplat_repo_revision and (GSPLAT_REPO_DIR / ".git").exists():
        run_shell(f"git checkout {config.gsplat_repo_revision}", cwd=GSPLAT_REPO_DIR)

    examples_dir = GSPLAT_REPO_DIR / "examples"
    simple = examples_dir / "simple_trainer.py"
    source = validate_gsplat_examples_compatibility(examples_dir, config.gsplat_repo_revision)

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
            if "ppisp" in line.lower() or "rmbrualla/pycolmap" in line.lower():
                skipped_requirements.append(line)
                continue
            kept_lines.append(line)
        filtered_requirements.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
        print("Installing gsplat example requirements with optional ppisp skipped.")
        print("Skipped requirement lines:", skipped_requirements)
        run_shell(f"{sys.executable} -m pip install -r {filtered_requirements}", cwd=GSPLAT_REPO_DIR)
    run_shell(f"{sys.executable} -m pip install numpy==1.26.4", cwd="/content")
    run_shell(
        f"{sys.executable} -m pip install --no-deps --force-reinstall "
        "\"git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e\"",
        cwd="/content",
    )
    colmap_loader = examples_dir / "datasets" / "colmap.py"
    if colmap_loader.exists():
        loader_text = colmap_loader.read_text(encoding="utf-8")
        loader_text = loader_text.replace(
            "from pycolmap import SceneManager",
            "from pycolmap.scene_manager import SceneManager",
        )
        if "from plyfile import PlyData" not in loader_text:
            loader_text = loader_text.replace(
                "import numpy as np\n",
                "import numpy as np\nfrom plyfile import PlyData\n",
            )
        original_points_loader = (
            "        points = manager.points3D.astype(np.float32)\n"
            "        points_err = manager.point3D_errors.astype(np.float32)\n"
            "        points_rgb = manager.point3D_colors.astype(np.uint8)\n"
        )
        refined_points_loader = original_points_loader + (
            "        refined_points_path = os.path.join(colmap_dir, \"training_points.ply\")\n"
            "        if os.path.exists(refined_points_path):\n"
            "            vertex = PlyData.read(refined_points_path)[\"vertex\"].data\n"
            "            points = np.stack([vertex[\"x\"], vertex[\"y\"], vertex[\"z\"]], axis=1).astype(np.float32)\n"
            "            points_rgb = np.stack([vertex[\"red\"], vertex[\"green\"], vertex[\"blue\"]], axis=1).astype(np.uint8)\n"
            "            points_err = np.zeros((len(points),), dtype=np.float32)\n"
            "            print(f\"[Parser] Using refined VGGT training points: {refined_points_path} ({len(points)} points)\")\n"
        )
        if "Using refined VGGT training points" not in loader_text:
            if original_points_loader not in loader_text:
                raise RuntimeError(f"Could not patch refined VGGT point loader in {colmap_loader}")
            loader_text = loader_text.replace(original_points_loader, refined_points_loader)
        colmap_loader.write_text(loader_text, encoding="utf-8")
    trainer_text = simple.read_text(encoding="utf-8")
    trainer_text = trainer_text.replace("disable_video: bool = False", "disable_video: bool = True")
    trainer_text = trainer_text.replace("save_ply: bool = False", "save_ply: bool = True")
    simple.write_text(trainer_text, encoding="utf-8")
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
    init_points = Path(config.pointcloud_dir) / "init_points.ply"
    if not init_points.exists():
        raise FileNotFoundError(f"Missing {init_points}")
    if (
        config.resume
        and transforms_path.exists()
        and points3d_path.exists()
        and points3d_path.stat().st_mtime >= init_points.stat().st_mtime
        and not config.overwrite
    ):
        print("Using existing gsplat intermediate dataset.")
        return transforms_path, points3d_path
    frames = json.loads((Path(config.output_dir) / "frames.json").read_text(encoding="utf-8"))
    cameras = json.loads((Path(config.da3_output_dir) / "cameras.json").read_text(encoding="utf-8"))
    camera_frames = cameras["frames"]
    if len(frames) != len(camera_frames):
        raise ValueError("frames.json and cameras.json counts differ.")
    K0 = np.asarray(camera_frames[0]["intrinsic"], dtype=np.float32)
    transform_frames = []
    for idx, item in enumerate(camera_frames):
        src = Path(item["image_path"])
        dst = images_dir / f"frame_{idx+1:06d}{src.suffix.lower()}"
        if (
            not dst.exists()
            or config.overwrite
            or dst.stat().st_size != src.stat().st_size
            or dst.stat().st_mtime < src.stat().st_mtime
        ):
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
        required = [
            sparse_dir / "cameras.bin",
            sparse_dir / "images.bin",
            sparse_dir / "points3D.bin",
            sparse_dir / "points.ply",
            sparse_dir / "training_points.ply",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(f"VGGT COLMAP fallback requested, but files are missing: {missing}")
        point_count = ply_vertex_count(sparse_dir / "points.ply")
        if point_count <= 0:
            raise ValueError(f"VGGT COLMAP fallback requested, but the sparse point cloud is empty: {sparse_dir / 'points.ply'}")
        training_point_count = ply_vertex_count(sparse_dir / "training_points.ply")
        if training_point_count <= 0:
            raise ValueError(
                "VGGT sparse refinement did not create a usable training point cloud. "
                "Run section 12.6 before gsplat training."
            )
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
    geometry_reports = sorted(Path(config.reports_dir).glob("*.json"))
    result_ply = Path(config.result_ply_path)
    pointcloud_variants = [
        Path(config.pointcloud_dir) / "init_points.ply",
        Path(config.pointcloud_dir) / "init_points_multiview_unfiltered.ply",
        Path(config.pointcloud_dir) / "init_points_multiview_wall_filtered.ply",
        Path(config.pointcloud_dir) / "init_points_stats.json",
    ]
    report = {
        "frame_count": len(frames) if isinstance(frames, list) else None,
        "depth_count": len(depth_files),
        "confidence_count": len(conf_files),
        "cameras_json_frame_count": len(cameras.get("frames", [])) if isinstance(cameras, dict) else None,
        "init_points_ply_exists": (Path(config.pointcloud_dir) / "init_points.ply").exists(),
        "transforms_json_frame_count": len(transforms.get("frames", [])) if isinstance(transforms, dict) else None,
        "geometry_reports": [str(path) for path in geometry_reports],
        "train_ply_candidates": [str(p) for p in find_train_ply_candidates(config.gsplat_train_dir)],
        "result_ply_exists": result_ply.exists(),
        "result_ply_size_mb": result_ply.stat().st_size / (1024 * 1024) if result_ply.exists() else None,
        "pointcloud_variant_files": [str(path) for path in pointcloud_variants if path.exists()],
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
        "geometry_reports_dir": str(config.reports_dir),
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
    geometry_reports = sorted(Path(config.reports_dir).glob("*.json"))
    pointcloud_variants = [
        Path(config.pointcloud_dir) / "init_points.ply",
        Path(config.pointcloud_dir) / "init_points_multiview_unfiltered.ply",
        Path(config.pointcloud_dir) / "init_points_multiview_wall_filtered.ply",
        Path(config.pointcloud_dir) / "init_points_stats.json",
    ]
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
        for report_path in geometry_reports:
            zf.write(report_path, f"reports/{report_path.name}")
        for pointcloud_path in pointcloud_variants:
            if pointcloud_path.exists():
                zf.write(pointcloud_path, f"pointcloud/{pointcloud_path.name}")
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
