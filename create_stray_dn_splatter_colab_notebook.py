import json
from pathlib import Path


PROJECT_ROOT = Path("capstone_3dgs_project")
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "05_stray_dn_splatter_pipeline.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip("\n").splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(True),
    }


cells = [
    md(
        r"""
# Stray Scanner -> DN-Splatter Gaussian PLY Pipeline

This notebook converts a Stray Scanner RGB-D export into a Nerfstudio-style RGB-D dataset, trains the official DN-Splatter method, and exports a Gaussian PLY.

The final artifact is a **Gaussian splat PLY**, not a triangle mesh.

Official references checked while generating this notebook:

- Stray Scanner format: https://github.com/strayrobots/scanner/blob/main/docs/format.md
- Stray Scanner reference integration script: https://github.com/strayrobots/scanner/blob/main/scripts/integrate.py
- DN-Splatter README: https://github.com/maturk/dn-splatter/blob/main/README.md
- DN-Splatter dependency and CLI registrations: https://github.com/maturk/dn-splatter/blob/main/pyproject.toml
- DN-Splatter `normal-nerfstudio` parser: https://github.com/maturk/dn-splatter/blob/main/dn_splatter/data/normal_nerfstudio.py

Verified repository HEAD commits: Stray Scanner `ec3e1dc9d33f8df2289ede6a5c59f7991d1a6bbb`, DN-Splatter `97588b4290128ce7ba6fdbfaac3020b42b17de4c`.

Important adapter note: current Stray Scanner documentation describes 16-bit millimeter `depth/*.png`, `confidence/*.png`, and headered `odometry.csv`. The official repository also contains an older integration script that reads sequential pose arrays and `depth/*.npy`. This notebook records the detected variant in its manifest and supports both documented PNG depth and legacy NPY depth without silently inventing missing inputs.

Run cells from top to bottom in a fresh Google Colab A100 runtime.
"""
    ),
    md("## 1. Environment Check"),
    code(
        r"""
import os
import sys
import json
import math
import time
import shutil
import zipfile
import platform
import subprocess
import shlex
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import pandas as pd


def run_command(command, cwd=None, check=True, log_path=None, env=None):
    print(f"$ {command}")
    start = time.time()
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    recent = []
    log_file = None
    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")
    try:
        for line in process.stdout:
            print(line, end="")
            recent.append(line.rstrip())
            recent = recent[-40:]
            if log_file:
                log_file.write(line)
                log_file.flush()
        rc = process.wait()
    finally:
        if log_file:
            log_file.close()
    elapsed = time.time() - start
    print(f"\n[exit={rc}] elapsed={elapsed:.1f}s")
    if check and rc != 0:
        print("\nRecent log lines:")
        print("\n".join(recent))
        raise RuntimeError(f"Command failed with exit code {rc}: {command}")
    return rc


def capture_command(command, cwd=None, check=True, log_path=None, env=None):
    print(f"$ {command}")
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    text = completed.stdout
    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(text, encoding="utf-8")
    print("\n".join(text.splitlines()[:80]))
    if len(text.splitlines()) > 80:
        print("... output truncated in cell; full output saved to log.")
    if check and completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {command}")
    return text


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote JSON: {path}")
    return path


def summarize_numbers(values):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not values:
        return {"min": None, "median": None, "max": None}
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


print("Python:", sys.version)
print("Platform:", platform.platform())
print("Working directory:", os.getcwd())
try:
    import torch
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("Torch import failed:", repr(exc))

run_command("nvidia-smi", check=False)
"""
    ),
    md("## 2. Google Drive Mount"),
    code(
        r"""
USE_DRIVE = True

if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    print("Google Drive mounted at /content/drive")
else:
    print("Using Colab runtime storage only.")
"""
    ),
    md("## 3. PipelineConfig Definition"),
    code(
        r"""
@dataclass
class PipelineConfig:
    use_drive: bool = True
    stray_export_source: Literal["drive_path", "upload_zip"] = "drive_path"
    stray_export_path: str = "/content/drive/MyDrive/capstone_3dgs_project/input/stray_scanner/scan_export"
    job_id: str = "stray_room_001"
    scene_name: str = "stray_room_001"

    blur_score_method: str = "laplacian_variance"
    min_blur_score: float = 55.0

    similarity_method: str = "phash"
    max_similarity: float = 0.94

    min_translation_baseline_m: float = 0.04
    min_rotation_baseline_deg: float = 3.0
    max_frames: Optional[int] = 240

    depth_min_m: float = 0.20
    depth_max_m: float = 5.0
    min_valid_depth_ratio: float = 0.20
    min_confidence_ratio: float = 0.20

    stray_pose_convention: str = "stray_t_wc_camera_to_world"
    pose_conversion: str = "identity_stray_c2w_to_nerfstudio"
    depth_unit_scale_factor: float = 0.001

    dn_splatter_max_num_iterations: int = 7000
    dn_splatter_eval_mode: str = "interval"
    dn_splatter_normal_supervision: Literal["depth", "mono"] = "depth"
    dn_splatter_use_depth_loss: bool = True

    overwrite: bool = False
    resume: bool = True
    dry_run: bool = True
    project_root: Optional[str] = None
    local_cache_mode: Literal["rgb_only", "full_export"] = "rgb_only"
    refresh_local_cache: bool = False

    min_selected_frames: int = 24
    pointcloud_depth_stride: int = 8
    pointcloud_max_points: int = 500000
    pointcloud_min_confidence_value: int = 1
    dn_splatter_depth_lambda: float = 0.2
    dn_splatter_depth_loss_type: str = "EdgeAwareLogL1"
    dn_splatter_use_normal_loss: bool = True
    dn_splatter_use_normal_tv_loss: bool = True
    checkpoint_every_iterations: int = 1000
    optional_colab_download: bool = False

    def resolve(self):
        if self.project_root is None:
            self.project_root = (
                "/content/drive/MyDrive/capstone_3dgs_project"
                if self.use_drive
                else "/content/capstone_3dgs_project"
            )
        root = Path(self.project_root)
        scratch = Path("/content") / "stray_dn_splatter_scratch" / self.job_id
        self.job_root = str(root / "runs" / self.job_id)
        self.work_dir = str(Path(self.job_root) / "work")
        self.logs_dir = str(Path(self.job_root) / "logs")
        self.reports_dir = str(Path(self.job_root) / "reports")
        self.runtime_scratch_dir = str(scratch)
        self.rgb_extract_dir = str(scratch / "rgb_extracted")
        self.dn_dataset_dir = str(root / "data" / "dn_splatter" / self.job_id)
        self.dn_output_dir = str(root / "outputs" / "dn_splatter" / self.job_id)
        self.gaussian_export_dir = str(root / "exports" / "gaussian_ply" / self.job_id)
        self.result_dir = str(Path(self.job_root) / "result_package")
        self.upload_extract_dir = str(scratch / "uploaded_stray_export")
        self.local_export_cache_dir = str(scratch / "stray_export")
        return self


# Fast A100 validation preset.
config = PipelineConfig(
    use_drive=USE_DRIVE,
    stray_export_source="drive_path",
    stray_export_path="/content/drive/MyDrive/capstone_3dgs_project/input/stray_scanner/scan_export",
    job_id="stray_room_001",
    scene_name="stray_room_001",
    dn_splatter_max_num_iterations=7000,
    max_frames=240,
    dry_run=True,
    resume=True,
).resolve()

# Recommended A100 quality preset after the dry run and a short successful training:
# config.dn_splatter_max_num_iterations = 30000
# config.max_frames = 600  # Typically use 400 to 800 frames.

for path in [
    config.job_root,
    config.work_dir,
    config.logs_dir,
    config.reports_dir,
    config.runtime_scratch_dir,
    config.rgb_extract_dir,
    config.dn_output_dir,
    config.gaussian_export_dir,
    config.result_dir,
]:
    Path(path).mkdir(parents=True, exist_ok=True)


def config_to_dict(config):
    payload = asdict(config)
    for key, value in vars(config).items():
        if key not in payload:
            payload[key] = value
    return payload


write_json(Path(config.reports_dir) / "config_snapshot.json", config_to_dict(config))
print(json.dumps(config_to_dict(config), indent=2, ensure_ascii=False))
"""
    ),
    md("## 4. Stray Scanner Export Input Preparation"),
    code(
        r"""
def find_stray_export_root(root):
    root = Path(root)
    if (root / "rgb.mp4").exists():
        return root
    matches = [path.parent for path in root.rglob("rgb.mp4")]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No Stray Scanner rgb.mp4 found under: {root}")
    raise RuntimeError(f"Multiple Stray Scanner exports found under {root}: {matches}")


def copy_export_to_local_cache(source_root, cache_root, refresh=False):
    source_root = Path(source_root)
    cache_root = Path(cache_root)
    complete_marker = cache_root / ".stray_export_cache_complete.json"
    if refresh and cache_root.exists():
        shutil.rmtree(cache_root)
    if complete_marker.exists():
        print("Using completed local Stray export cache:", cache_root)
        return cache_root
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        source_files = sorted(path for path in source_root.rglob("*") if path.is_file())
    except OSError as exc:
        raise RuntimeError(
            "Google Drive mount became unavailable while listing the Stray export. "
            "Remount Drive, then rerun this Step 4 cell. Existing local cache files will be reused. "
            f"Original error: {exc!r}"
        ) from exc
    if not source_files:
        raise RuntimeError(f"No files found while caching Stray export: {source_root}")
    total_files = len(source_files)
    progress_every = max(1, min(100, total_files // 20 or 1))
    started = time.time()
    copied_files = 0
    reused_files = 0
    print(f"[Step 4] Caching {total_files} files from Drive. Progress prints every {progress_every} file(s).")
    for processed_count, source_path in enumerate(source_files, start=1):
        relative_path = source_path.relative_to(source_root)
        target_path = cache_root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            source_size = source_path.stat().st_size
            if target_path.exists() and target_path.stat().st_size == source_size:
                reused_files += 1
            else:
                last_error = None
                for attempt in range(1, 4):
                    try:
                        shutil.copy2(source_path, target_path)
                        last_error = None
                        copied_files += 1
                        break
                    except OSError as exc:
                        last_error = exc
                        print(f"[Step 4] Copy retry {attempt}/3 for {relative_path}: {exc}")
                        time.sleep(attempt * 2)
                if last_error is not None:
                    raise last_error
        except OSError as exc:
            raise RuntimeError(
                "Google Drive mount became unavailable while copying the Stray export. "
                f"Stopped at file {processed_count}/{total_files}: {relative_path}. "
                "The partial local cache was preserved. Remount Drive, then rerun this Step 4 cell "
                "to resume without recopying completed files. "
                f"Original error: {exc!r}"
            ) from exc
        if processed_count == 1 or processed_count % progress_every == 0 or processed_count == total_files:
            elapsed = time.time() - started
            rate = processed_count / elapsed if elapsed > 0 else 0.0
            remaining = total_files - processed_count
            eta_seconds = remaining / rate if rate > 0 else None
            eta_text = f"{eta_seconds / 60:.1f} min" if eta_seconds is not None else "unknown"
            print(
                f"[Step 4] {processed_count}/{total_files} ({processed_count / total_files * 100:.1f}%) | "
                f"copied={copied_files} | reused={reused_files} | "
                f"elapsed={elapsed / 60:.1f} min | ETA={eta_text}"
            )
    complete_marker.write_text(
        json.dumps(
            {
                "source_root": str(source_root),
                "cached_file_count": total_files,
                "completed_at": datetime.now().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Completed local Stray export cache:", cache_root)
    return cache_root


def prepare_stray_export(config):
    print("[Step 4] Preparing Stray Scanner export input")
    if config.stray_export_source == "drive_path":
        candidate = Path(config.stray_export_path)
        if not candidate.exists():
            raise FileNotFoundError(f"Stray Scanner export path does not exist: {candidate}")
        root = find_stray_export_root(candidate)
    elif config.stray_export_source == "upload_zip":
        from google.colab import files
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise ValueError("Upload exactly one Stray Scanner ZIP file.")
        zip_name = next(iter(uploaded))
        zip_path = Path("/content") / zip_name
        if zip_path.suffix.lower() != ".zip":
            raise ValueError(f"Expected a .zip upload, got: {zip_path.name}")
        extract_dir = Path(config.upload_extract_dir)
        if extract_dir.exists() and config.overwrite:
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        root = find_stray_export_root(extract_dir)
    else:
        raise ValueError("config.stray_export_source must be 'drive_path' or 'upload_zip'.")
    if config.local_cache_mode == "full_export" and config.stray_export_source == "drive_path":
        cache_root = Path(config.local_export_cache_dir)
        print("Copying or resuming Stray export cache from Drive to Colab local scratch:", cache_root)
        print("This one-time cache copy can take several minutes for thousands of small files.")
        cache_root = copy_export_to_local_cache(root, cache_root, refresh=config.refresh_local_cache)
        root = find_stray_export_root(cache_root)
    elif config.local_cache_mode != "rgb_only":
        raise ValueError("config.local_cache_mode must be 'rgb_only' or 'full_export'.")
    print("Stray export root:", root)
    return root


stray_export_root = prepare_stray_export(config)
"""
    ),
    md("## 5. Export Discovery And Manifest"),
    code(
        r"""
FRAME_ID_PATTERN = re.compile(r"(\d+)")


def frame_id_from_path(path):
    matches = FRAME_ID_PATTERN.findall(Path(path).stem)
    if not matches:
        raise ValueError(f"Could not parse frame id from filename: {path}")
    return int(matches[-1])


def discover_indexed_files(directory, allowed_suffixes):
    directory = Path(directory)
    discovered = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in allowed_suffixes:
            frame_id = frame_id_from_path(path)
            if frame_id in discovered:
                raise RuntimeError(f"Duplicate frame id {frame_id} in {directory}")
            discovered[frame_id] = path
    return discovered


def discover_depth_files(depth_dir):
    return discover_indexed_files(depth_dir, {".png", ".npy"})


def discover_confidence_files(confidence_dir):
    return discover_indexed_files(confidence_dir, {".png", ".npy"})


def discover_stray_export(config):
    root = Path(stray_export_root)
    required_files = ["rgb.mp4", "odometry.csv", "imu.csv", "camera_matrix.csv"]
    required_dirs = ["depth", "confidence"]
    for name in required_files:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Missing required Stray Scanner export file: {name}")
    for name in required_dirs:
        if not (root / name).is_dir():
            raise FileNotFoundError(f"Missing required Stray Scanner export directory: {name}/")
    depth_files = discover_depth_files(root / "depth")
    confidence_files = discover_confidence_files(root / "confidence")
    distortion_dir = root / "distortion"
    distortion_files = sorted(str(path) for path in distortion_dir.glob("*.bin")) if distortion_dir.is_dir() else []
    if not depth_files:
        raise RuntimeError("Missing readable Stray Scanner depth frames in depth/. Expected .png or legacy .npy files.")
    if not confidence_files:
        raise RuntimeError("Missing readable Stray Scanner confidence frames in confidence/. Expected .png or legacy .npy files.")
    payload = {
        "stray_export_root": str(root),
        "required_contract": {
            "files": required_files,
            "directories": [f"{name}/" for name in required_dirs],
        },
        "rgb_video_path": str(root / "rgb.mp4"),
        "odometry_csv_path": str(root / "odometry.csv"),
        "imu_csv_path": str(root / "imu.csv"),
        "camera_matrix_csv_path": str(root / "camera_matrix.csv"),
        "depth_directory": str(root / "depth"),
        "confidence_directory": str(root / "confidence"),
        "optional_distortion_directory": str(distortion_dir) if distortion_dir.is_dir() else None,
        "depth_frame_count": len(depth_files),
        "confidence_frame_count": len(confidence_files),
        "distortion_file_count": len(distortion_files),
        "depth_formats": sorted({path.suffix.lower() for path in depth_files.values()}),
        "confidence_formats": sorted({path.suffix.lower() for path in confidence_files.values()}),
        "depth_files": {str(key): str(value) for key, value in depth_files.items()},
        "confidence_files": {str(key): str(value) for key, value in confidence_files.items()},
        "distortion_files": distortion_files,
        "official_documented_depth_unit": "millimeters",
        "official_documented_confidence_values": [0, 1, 2],
        "adapter_note": "Current docs use PNG. Official repository integration script also contains a legacy NPY reader.",
        "distortion_note": (
            "The documented optional distortion/*.bin lookup tables are recorded but not automatically rectified. "
            "Audit captures with distortion LUTs before a long training run."
        ),
    }
    write_json(Path(config.reports_dir) / "stray_input_manifest.json", payload)
    print("Depth frames:", len(depth_files))
    print("Confidence frames:", len(confidence_files))
    return payload, depth_files, confidence_files


stray_manifest, depth_files, confidence_files = discover_stray_export(config)
"""
    ),
    md("## 6. RGB Frame Extraction"),
    code(
        r"""
DOCUMENTED_ODOMETRY_COLUMNS = {
    "timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw", "fx", "fy", "cx", "cy"
}


def load_odometry_csv(path):
    path = Path(path)
    headered = pd.read_csv(path)
    normalized_columns = {str(column).strip().lower(): column for column in headered.columns}
    if DOCUMENTED_ODOMETRY_COLUMNS.issubset(normalized_columns):
        frame = headered.rename(columns={original: normalized for normalized, original in normalized_columns.items()})
        frame["frame"] = frame["frame"].astype(str).str.extract(r"(\d+)", expand=False).astype(int)
        frame.attrs["stray_odometry_variant"] = "documented_headered_per_frame_intrinsics"
        return frame

    raw = pd.read_csv(path, header=None)
    numeric = raw.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.shape[1] < 7:
        raise RuntimeError(
            "Unsupported odometry.csv format. Expected documented headered columns or a legacy numeric x,y,z,qx,qy,qz,qw table."
        )
    legacy = numeric.iloc[:, :7].copy()
    legacy.columns = ["x", "y", "z", "qx", "qy", "qz", "qw"]
    legacy.insert(0, "frame", np.arange(len(legacy), dtype=int))
    legacy.insert(0, "timestamp", np.arange(len(legacy), dtype=float))
    legacy.attrs["stray_odometry_variant"] = "legacy_sequential_pose_only"
    return legacy


def load_camera_matrix_csv(path):
    intrinsic = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise RuntimeError(f"camera_matrix.csv must contain a 3x3 matrix, got: {intrinsic.shape}")
    if not np.isfinite(intrinsic).all():
        raise RuntimeError("camera_matrix.csv contains non-finite values.")
    return intrinsic


def extract_rgb_frames_from_video(config, rgb_video_path, requested_frame_ids):
    print("[Step 6] Extracting aligned RGB frames from rgb.mp4")
    requested = set(int(value) for value in requested_frame_ids)
    if not requested:
        raise RuntimeError("No RGB frame ids requested after intersecting odometry, depth, and confidence inputs.")
    output_dir = Path(config.rgb_extract_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(rgb_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open Stray Scanner RGB video: {rgb_video_path}")
    video_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted = {}
    failed = []
    max_requested = max(requested)
    frame_id = 0
    while frame_id <= max_requested:
        ok, image = capture.read()
        if not ok:
            break
        if frame_id in requested:
            output_path = output_dir / f"frame_{frame_id:06d}.jpg"
            if output_path.exists() and not config.overwrite:
                extracted[frame_id] = output_path
            else:
                if not cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    failed.append(frame_id)
                else:
                    extracted[frame_id] = output_path
        frame_id += 1
    capture.release()
    missing = sorted(requested - set(extracted))
    failed = sorted(set(failed) | set(missing))
    print("Video frame count:", video_frame_count)
    print("Requested RGB frames:", len(requested))
    print("Extracted RGB frames:", len(extracted))
    if failed:
        print("Failed RGB frame ids (first 40):", failed[:40])
    return extracted, video_frame_count, failed


odometry_df = load_odometry_csv(stray_manifest["odometry_csv_path"])
camera_matrix = load_camera_matrix_csv(stray_manifest["camera_matrix_csv_path"])
candidate_ids_before_rgb = sorted(set(odometry_df["frame"]) & set(depth_files) & set(confidence_files))
rgb_files, rgb_video_frame_count, failed_rgb_frame_ids = extract_rgb_frames_from_video(
    config,
    stray_manifest["rgb_video_path"],
    candidate_ids_before_rgb,
)
print("Odometry variant:", odometry_df.attrs["stray_odometry_variant"])
"""
    ),
    md("## 7. RGB / Depth / Confidence / Pose / Intrinsic Alignment"),
    code(
        r"""
@dataclass
class FrameRecord:
    frame_id: int
    timestamp: float
    rgb_path: str
    depth_path: str
    confidence_path: str
    pose_c2w: np.ndarray
    intrinsic: np.ndarray
    width: int
    height: int
    depth_width: int
    depth_height: int
    confidence_width: int
    confidence_height: int


def quaternion_xyzw_to_matrix(qx, qy, qz, qw):
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid zero or non-finite pose quaternion.")
    x, y, z, w = quat / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def convert_pose_to_nerfstudio(pose_c2w, conversion):
    pose_c2w = np.asarray(pose_c2w, dtype=np.float64)
    if conversion == "identity_stray_c2w_to_nerfstudio":
        return pose_c2w.copy()
    if conversion == "opencv_c2w_to_nerfstudio_opengl":
        converted = pose_c2w.copy()
        converted[:3, 1:3] *= -1.0
        return converted
    raise ValueError(
        "Unsupported config.pose_conversion. Use 'identity_stray_c2w_to_nerfstudio' or "
        "'opencv_c2w_to_nerfstudio_opengl'."
    )


def load_array(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is not None:
        return image
    try:
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if image is not None:
            return image
    except Exception:
        pass
    try:
        from PIL import Image
        with Image.open(path) as pil_image:
            return np.asarray(pil_image)
    except Exception as exc:
        exists = path.exists()
        size_bytes = path.stat().st_size if exists else None
        try:
            signature_hex = path.read_bytes()[:16].hex() if exists else None
        except Exception:
            signature_hex = None
        raise RuntimeError(
            f"Could not read image array after OpenCV and Pillow attempts: {path}. "
            f"exists={exists}, size_bytes={size_bytes}, first_16_bytes_hex={signature_hex}, "
            f"pillow_error={exc!r}"
        ) from exc


def load_array_shape(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return tuple(np.load(path, mmap_mode="r").shape)
    if path.suffix.lower() == ".png":
        import struct
        with open(path, "rb") as handle:
            header = handle.read(26)
        expected_signature = b"\x89PNG\r\n\x1a\n"
        if len(header) >= 26 and header[:8] == expected_signature and header[12:16] == b"IHDR":
            width, height = struct.unpack(">II", header[16:24])
            color_type = header[25]
            if width > 0 and height > 0 and color_type == 0:
                return (height, width)
    try:
        from PIL import Image
        with Image.open(path) as pil_image:
            if pil_image.mode not in {"L", "I", "I;16", "I;16B", "I;16L"}:
                raise RuntimeError(f"Expected a single-channel image, got Pillow mode={pil_image.mode!r}")
            return (pil_image.height, pil_image.width)
    except Exception as exc:
        raise RuntimeError(f"Could not read image-array shape metadata: {path}. error={exc!r}") from exc


def row_intrinsic(row, fallback):
    keys = ["fx", "fy", "cx", "cy"]
    if all(key in row.index and np.isfinite(float(row[key])) for key in keys):
        return np.array([
            [float(row["fx"]), 0.0, float(row["cx"])],
            [0.0, float(row["fy"]), float(row["cy"])],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64), "odometry.csv per-frame fx/fy/cx/cy"
    return np.asarray(fallback, dtype=np.float64), "camera_matrix.csv fallback"


def pose_from_odometry_row(row, conversion):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = quaternion_xyzw_to_matrix(row["qx"], row["qy"], row["qz"], row["qw"])
    pose[:3, 3] = [float(row["x"]), float(row["y"]), float(row["z"])]
    return convert_pose_to_nerfstudio(pose, conversion)


def align_stray_frames(odometry, camera_matrix, rgb_files, depth_files, confidence_files, config):
    records = []
    intrinsic_sources = Counter()
    unreadable_required_data_frames = []
    odometry_by_id = {}
    for _, row in odometry.iterrows():
        frame_id = int(row["frame"])
        if frame_id in odometry_by_id:
            raise RuntimeError(f"Duplicate frame id in odometry.csv: {frame_id}")
        odometry_by_id[frame_id] = row
    common_ids = sorted(set(rgb_files) & set(depth_files) & set(confidence_files) & set(odometry_by_id))
    total_common_ids = len(common_ids)
    progress_every = max(1, min(100, total_common_ids // 20 or 1))
    alignment_started = time.time()
    print(
        f"[Step 7] Aligning {total_common_ids} common frames. "
        f"Progress will print every {progress_every} frame(s)."
    )
    for processed_count, frame_id in enumerate(common_ids, start=1):
        row = odometry_by_id[frame_id]
        try:
            rgb = cv2.imread(str(rgb_files[frame_id]), cv2.IMREAD_COLOR)
            depth_shape = load_array_shape(depth_files[frame_id])
            confidence_shape = load_array_shape(confidence_files[frame_id])
            if rgb is None:
                raise RuntimeError(f"Could not read extracted RGB frame: {rgb_files[frame_id]}")
            if len(depth_shape) != 2:
                raise RuntimeError(f"Depth frame must be single-channel: {depth_files[frame_id]} shape={depth_shape}")
            if len(confidence_shape) != 2:
                raise RuntimeError(
                    f"Confidence frame must be single-channel: {confidence_files[frame_id]} shape={confidence_shape}"
                )
            intrinsic, intrinsic_source = row_intrinsic(row, camera_matrix)
            pose_c2w = pose_from_odometry_row(row, config.pose_conversion)
        except Exception as exc:
            unreadable_required_data_frames.append({
                "frame_id": frame_id,
                "rgb_path": str(rgb_files[frame_id]),
                "depth_path": str(depth_files[frame_id]),
                "confidence_path": str(confidence_files[frame_id]),
                "error": str(exc),
            })
            print(f"Skipping unreadable required-data frame {frame_id}: {exc}")
        else:
            intrinsic_sources[intrinsic_source] += 1
            records.append(FrameRecord(
                frame_id=frame_id,
                timestamp=float(row["timestamp"]),
                rgb_path=str(rgb_files[frame_id]),
                depth_path=str(depth_files[frame_id]),
                confidence_path=str(confidence_files[frame_id]),
                pose_c2w=pose_c2w,
                intrinsic=intrinsic,
                width=int(rgb.shape[1]),
                height=int(rgb.shape[0]),
                depth_width=int(depth_shape[1]),
                depth_height=int(depth_shape[0]),
                confidence_width=int(confidence_shape[1]),
                confidence_height=int(confidence_shape[0]),
            ))
        if processed_count == 1 or processed_count % progress_every == 0 or processed_count == total_common_ids:
            elapsed = time.time() - alignment_started
            rate = processed_count / elapsed if elapsed > 0 else 0.0
            remaining = total_common_ids - processed_count
            eta_seconds = remaining / rate if rate > 0 else None
            eta_text = f"{eta_seconds / 60:.1f} min" if eta_seconds is not None else "unknown"
            print(
                f"[Step 7] {processed_count}/{total_common_ids} "
                f"({processed_count / total_common_ids * 100:.1f}%) | "
                f"aligned={len(records)} | skipped={len(unreadable_required_data_frames)} | "
                f"elapsed={elapsed / 60:.1f} min | ETA={eta_text}"
            )
    return records, dict(intrinsic_sources), odometry_by_id, unreadable_required_data_frames


aligned_records, intrinsic_sources, odometry_by_id, unreadable_required_data_frames = align_stray_frames(
    odometry_df, camera_matrix, rgb_files, depth_files, confidence_files, config
)
video_ids = set(range(rgb_video_frame_count))
modality_ids = set(odometry_by_id) | set(depth_files) | set(confidence_files)
all_ids = video_ids | modality_ids
aligned_ids = {record.frame_id for record in aligned_records}
timestamps = [record.timestamp for record in aligned_records]
timestamp_monotonic = all(a <= b for a, b in zip(timestamps, timestamps[1:]))
alignment_warnings = []
if not timestamp_monotonic:
    alignment_warnings.append("Aligned timestamps are not monotonic.")
if len(aligned_records) != len(candidate_ids_before_rgb):
    alignment_warnings.append("Some common frame ids could not be aligned. Inspect failed RGB extraction and unreadable data.")
if unreadable_required_data_frames:
    alignment_warnings.append(
        f"Skipped {len(unreadable_required_data_frames)} frame(s) with unreadable or invalid required data."
    )

alignment_report = {
    "rgb_video_frame_count": rgb_video_frame_count,
    "odometry_row_count": len(odometry_df),
    "depth_file_count": len(depth_files),
    "confidence_file_count": len(confidence_files),
    "common_frame_id_count": len(set(rgb_files) & set(depth_files) & set(confidence_files) & set(odometry_by_id)),
    "missing_rgb_frame_ids": sorted(modality_ids - video_ids),
    "failed_rgb_extraction_frame_ids": failed_rgb_frame_ids,
    "unreadable_required_data_frames": unreadable_required_data_frames,
    "missing_depth_frame_ids": sorted(all_ids - set(depth_files)),
    "missing_confidence_frame_ids": sorted(all_ids - set(confidence_files)),
    "missing_pose_frame_ids": sorted(all_ids - set(odometry_by_id)),
    "missing_required_data_frame_ids": sorted(all_ids - aligned_ids),
    "intrinsic_source": intrinsic_sources,
    "odometry_variant": odometry_df.attrs["stray_odometry_variant"],
    "aligned_frame_count": len(aligned_records),
    "first_timestamp": timestamps[0] if timestamps else None,
    "last_timestamp": timestamps[-1] if timestamps else None,
    "timestamp_monotonic": timestamp_monotonic,
    "warnings": alignment_warnings,
}
write_json(Path(config.reports_dir) / "frame_alignment_report.json", alignment_report)
print(json.dumps({key: value for key, value in alignment_report.items() if not key.startswith("missing_")}, indent=2))
"""
    ),
    md("## 8. Minimum Format Validation Before Frame Selection"),
    code(
        r"""
def validate_minimum_format(aligned_records, odometry_df, camera_matrix, alignment_report):
    print("[Step 8] Running minimum pre-selection validation")
    required_pose_columns = {"x", "y", "z", "qx", "qy", "qz", "qw"}
    missing_pose_columns = sorted(required_pose_columns - set(odometry_df.columns))
    checks = {
        "rgb_candidate_count": len(rgb_files),
        "depth_candidate_count": len(depth_files),
        "confidence_candidate_count": len(confidence_files),
        "pose_candidate_count": len(odometry_df),
        "common_frame_id_count": alignment_report["common_frame_id_count"],
        "camera_matrix_shape": list(camera_matrix.shape),
        "camera_matrix_finite": bool(np.isfinite(camera_matrix).all()),
        "missing_odometry_pose_fields": missing_pose_columns,
        "timestamp_monotonic": alignment_report["timestamp_monotonic"],
    }
    if camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all():
        raise RuntimeError("Invalid camera_matrix.csv. Expected a finite 3x3 intrinsic matrix.")
    if missing_pose_columns:
        raise RuntimeError(f"Missing odometry pose fields: {missing_pose_columns}")
    if not aligned_records:
        raise RuntimeError("No aligned Stray Scanner RGB/depth/confidence/pose frames found.")
    if not alignment_report["timestamp_monotonic"]:
        raise RuntimeError("Aligned Stray Scanner timestamps are not monotonic. Inspect frame_alignment_report.json.")
    print(json.dumps(checks, indent=2))
    return checks


minimum_validation = validate_minimum_format(aligned_records, odometry_df, camera_matrix, alignment_report)
"""
    ),
    md("## 9-12. Blur, Similarity, Pose Baseline, And max_frames Filtering"),
    code(
        r"""
def blur_score(image_path, method):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read RGB frame for blur scoring: {image_path}")
    if method != "laplacian_variance":
        raise ValueError("Only blur_score_method='laplacian_variance' is implemented.")
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def phash_bits(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read RGB frame for pHash: {image_path}")
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    transformed = cv2.dct(resized)
    low = transformed[:8, :8].flatten()[1:]
    return low >= np.median(low)


def similarity_score(previous_hash, current_hash, method):
    if method != "phash":
        raise ValueError("Only similarity_method='phash' is implemented.")
    if previous_hash is None:
        return None
    return float(1.0 - np.mean(previous_hash != current_hash))


def pose_delta(previous_pose, current_pose):
    if previous_pose is None:
        return 0.0, 0.0
    previous_pose = np.asarray(previous_pose)
    current_pose = np.asarray(current_pose)
    translation = float(np.linalg.norm(current_pose[:3, 3] - previous_pose[:3, 3]))
    relative_rotation = previous_pose[:3, :3].T @ current_pose[:3, :3]
    cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
    rotation_deg = float(np.degrees(np.arccos(cosine)))
    return translation, rotation_deg


def frame_depth_confidence_metrics(record, config):
    depth = load_array(record.depth_path).astype(np.float64)
    confidence = load_array(record.confidence_path)
    depth_m = depth * config.depth_unit_scale_factor
    finite = np.isfinite(depth_m)
    valid_depth = finite & (depth_m >= config.depth_min_m) & (depth_m <= config.depth_max_m)
    valid_confidence = np.isfinite(confidence) & (confidence >= config.pointcloud_min_confidence_value)
    return {
        "valid_depth_ratio": float(np.mean(valid_depth)),
        "confidence_ratio": float(np.mean(valid_confidence)),
    }


def filter_frames(records, config):
    print("[Steps 9-12] Filtering frames: blur -> similarity -> pose baseline -> max_frames")
    selected = []
    rows = []
    last_selected_hash = None
    last_selected_pose = None
    for record in records:
        score = blur_score(record.rgb_path, config.blur_score_method)
        current_hash = phash_bits(record.rgb_path)
        similarity = similarity_score(last_selected_hash, current_hash, config.similarity_method)
        translation, rotation = pose_delta(last_selected_pose, record.pose_c2w)
        rejection_reason = None
        metrics = {"valid_depth_ratio": None, "confidence_ratio": None}
        required_data_error = None
        if config.max_frames is not None and len(selected) >= config.max_frames:
            rejection_reason = "max_frames_limit"
        elif score < config.min_blur_score:
            rejection_reason = "blur_too_low"
        elif similarity is not None and similarity >= config.max_similarity:
            rejection_reason = "too_similar"
        elif last_selected_pose is not None and (
            translation < config.min_translation_baseline_m
            and rotation < config.min_rotation_baseline_deg
        ):
            rejection_reason = "pose_baseline_too_small"
        else:
            try:
                metrics = frame_depth_confidence_metrics(record, config)
                if metrics["valid_depth_ratio"] < config.min_valid_depth_ratio:
                    rejection_reason = "invalid_depth"
                elif metrics["confidence_ratio"] < config.min_confidence_ratio:
                    rejection_reason = "invalid_confidence"
            except Exception as exc:
                rejection_reason = "missing_required_data"
                required_data_error = str(exc)
                print(f"Skipping unreadable required-data frame {record.frame_id}: {exc}")
        is_selected = rejection_reason is None
        if is_selected:
            selected.append(record)
            last_selected_hash = current_hash
            last_selected_pose = record.pose_c2w
        rows.append({
            "frame_id": record.frame_id,
            "timestamp": record.timestamp,
            "blur_score": score,
            "similarity_score": similarity,
            "translation_from_last_selected_m": translation,
            "rotation_from_last_selected_deg": rotation,
            "valid_depth_ratio": metrics["valid_depth_ratio"],
            "confidence_ratio": metrics["confidence_ratio"],
            "selected": is_selected,
            "rejection_reason": rejection_reason,
            "required_data_error": required_data_error,
        })
    missing_required_rows = [{
        "frame_id": frame_id,
        "timestamp": None,
        "blur_score": None,
        "similarity_score": None,
        "translation_from_last_selected_m": None,
        "rotation_from_last_selected_deg": None,
        "valid_depth_ratio": None,
        "confidence_ratio": None,
        "selected": False,
        "rejection_reason": "missing_required_data",
        "required_data_error": "Frame could not be aligned because at least one required modality was absent or invalid.",
    } for frame_id in alignment_report["missing_required_data_frame_ids"]]
    report_rows = sorted(rows + missing_required_rows, key=lambda row: row["frame_id"])
    counts = Counter(row["rejection_reason"] for row in report_rows if row["rejection_reason"])
    report = {
        "filter_order": ["blur", "similarity", "pose_baseline", "max_frames"],
        "thresholds": {
            "min_blur_score": config.min_blur_score,
            "max_similarity": config.max_similarity,
            "min_translation_baseline_m": config.min_translation_baseline_m,
            "min_rotation_baseline_deg": config.min_rotation_baseline_deg,
            "max_frames": config.max_frames,
        },
        "summary": {
            "total_aligned_frames": len(records),
            "total_candidate_frame_ids": len(report_rows),
            "selected_frames": len(selected),
            "rejected_aligned_frames": len(records) - len(selected),
            "missing_required_data_frames": len(missing_required_rows),
            "rejected_frames_including_missing_required_data": len(report_rows) - len(selected),
            "rejection_counts_by_reason": dict(counts),
            "blur": summarize_numbers([row["blur_score"] for row in rows]),
            "similarity": summarize_numbers([row["similarity_score"] for row in rows]),
            "translation_baseline_m": summarize_numbers(
                [row["translation_from_last_selected_m"] for row in rows]
            ),
            "rotation_baseline_deg": summarize_numbers(
                [row["rotation_from_last_selected_deg"] for row in rows]
            ),
        },
        "frames": report_rows,
    }
    write_json(Path(config.reports_dir) / "frame_filter_report.json", report)
    print(json.dumps(report["summary"], indent=2))
    return selected, report


selected_records, frame_filter_report = filter_frames(aligned_records, config)
"""
    ),
    md("## 13. Selected Pose / Depth / Confidence Detailed Validation"),
    code(
        r"""
def validate_selected_records(selected_records, config):
    print("[Step 13] Validating selected pose, depth, and confidence records")
    details = []
    translations = []
    rotations = []
    previous_pose = None
    for record in selected_records:
        depth = load_array(record.depth_path).astype(np.float64)
        confidence = load_array(record.confidence_path).astype(np.float64)
        depth_m = depth * config.depth_unit_scale_factor
        finite = np.isfinite(depth_m)
        valid = finite & (depth_m >= config.depth_min_m) & (depth_m <= config.depth_max_m)
        confidence_valid = np.isfinite(confidence) & (confidence >= config.pointcloud_min_confidence_value)
        translation, rotation = pose_delta(previous_pose, record.pose_c2w)
        if previous_pose is not None:
            translations.append(translation)
            rotations.append(rotation)
        previous_pose = record.pose_c2w
        detail = {
            "frame_id": record.frame_id,
            "timestamp": record.timestamp,
            "rgb_shape": [record.height, record.width],
            "depth_shape": [record.depth_height, record.depth_width],
            "confidence_shape": [record.confidence_height, record.confidence_width],
            "depth_matches_rgb_resolution": (
                record.depth_height == record.height and record.depth_width == record.width
            ),
            "depth_to_rgb_scale": [
                record.depth_width / record.width,
                record.depth_height / record.height,
            ],
            "intrinsic_shape": list(record.intrinsic.shape),
            "intrinsic_finite": bool(np.isfinite(record.intrinsic).all()),
            "pose_shape": list(record.pose_c2w.shape),
            "pose_finite": bool(np.isfinite(record.pose_c2w).all()),
            "depth_nan_ratio": float(np.mean(np.isnan(depth_m))),
            "depth_inf_ratio": float(np.mean(np.isinf(depth_m))),
            "depth_zero_ratio": float(np.mean(depth_m == 0)),
            "valid_depth_ratio": float(np.mean(valid)),
            "confidence_valid_ratio": float(np.mean(confidence_valid)),
            "translation_from_previous_selected_m": translation,
            "rotation_from_previous_selected_deg": rotation,
        }
        if detail["intrinsic_shape"] != [3, 3] or not detail["intrinsic_finite"]:
            raise RuntimeError(f"Invalid intrinsic matrix for frame {record.frame_id}")
        if detail["pose_shape"] != [4, 4] or not detail["pose_finite"]:
            raise RuntimeError(f"Invalid pose matrix for frame {record.frame_id}")
        details.append(detail)
    report = {
        "selected_rgb_count": len(selected_records),
        "selected_depth_count": len(selected_records),
        "selected_confidence_count": len(selected_records),
        "selected_pose_count": len(selected_records),
        "counts_match": True,
        "pose_convention": config.stray_pose_convention,
        "coordinate_conversion": config.pose_conversion,
        "coordinate_conversion_note": (
            "The Stray docs name camera pose fields but do not explicitly document axis directions. "
            "The adapter makes conversion explicit. Audit this setting for the capture version before a long run."
        ),
        "depth_source_unit": "millimeters for documented PNG export; adapter expects the same numeric unit for legacy NPY",
        "depth_unit_scale_factor_to_meters": config.depth_unit_scale_factor,
        "depth_range_m": [config.depth_min_m, config.depth_max_m],
        "confidence_contract": "Official Stray docs: grayscale values 0, 1, or 2; higher is better.",
        "translation_baseline_m": summarize_numbers(translations),
        "rotation_baseline_deg": summarize_numbers(rotations),
        "frames": details,
    }
    write_json(Path(config.reports_dir) / "pose_depth_validation_report.json", report)
    if len(selected_records) < config.min_selected_frames:
        raise RuntimeError(
            "Too few selected frames for DN-Splatter training: "
            f"selected={len(selected_records)}, required>={config.min_selected_frames}. "
            "Relax min_blur_score, max_similarity, or pose baseline thresholds."
        )
    print(json.dumps({key: value for key, value in report.items() if key != "frames"}, indent=2))
    return report


pose_depth_validation_report = validate_selected_records(selected_records, config)
"""
    ),
    md("## 14. DN-Splatter Dataset And Depth-Initialized Point Cloud"),
    code(
        r"""
def write_binary_ply(path, points, colors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if len(points) != len(colors):
        raise ValueError("PLY point and color counts do not match.")
    vertex = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertex)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        vertex.tofile(handle)
    return path


def build_depth_initial_pointcloud(records, config, output_path):
    print("[Step 14] Backprojecting filtered Stray depth to sparse initialization PLY")
    points_parts = []
    colors_parts = []
    stride = config.pointcloud_depth_stride
    for record in records:
        depth_raw = load_array(record.depth_path).astype(np.float64)
        confidence = load_array(record.confidence_path)
        rgb = cv2.imread(record.rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"Could not read RGB for point cloud: {record.rgb_path}")
        depth_m = depth_raw * config.depth_unit_scale_factor
        h, w = depth_m.shape
        rgb_small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        scale_x = w / record.width
        scale_y = h / record.height
        fx = record.intrinsic[0, 0] * scale_x
        fy = record.intrinsic[1, 1] * scale_y
        cx = record.intrinsic[0, 2] * scale_x
        cy = record.intrinsic[1, 2] * scale_y
        vv, uu = np.mgrid[0:h:stride, 0:w:stride]
        sampled_depth = depth_m[0:h:stride, 0:w:stride]
        sampled_confidence = confidence[0:h:stride, 0:w:stride]
        valid = (
            np.isfinite(sampled_depth)
            & (sampled_depth >= config.depth_min_m)
            & (sampled_depth <= config.depth_max_m)
            & np.isfinite(sampled_confidence)
            & (sampled_confidence >= config.pointcloud_min_confidence_value)
        )
        z = sampled_depth[valid]
        x = (uu[valid] - cx) * z / fx
        y = -(vv[valid] - cy) * z / fy
        camera_points = np.stack([x, y, -z, np.ones_like(z)], axis=1)
        world_points = (record.pose_c2w @ camera_points.T).T[:, :3]
        rgb_colors = cv2.cvtColor(rgb_small, cv2.COLOR_BGR2RGB)[0:h:stride, 0:w:stride][valid]
        points_parts.append(world_points.astype(np.float32))
        colors_parts.append(rgb_colors.astype(np.uint8))
    points = np.concatenate(points_parts, axis=0)
    colors = np.concatenate(colors_parts, axis=0)
    if len(points) > config.pointcloud_max_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(points), size=config.pointcloud_max_points, replace=False)
        points = points[indices]
        colors = colors[indices]
    if not len(points):
        raise RuntimeError("Depth initialization point cloud is empty after depth and confidence filtering.")
    write_binary_ply(output_path, points, colors)
    print("Depth initialization PLY points:", len(points))
    print("Depth initialization PLY:", output_path)
    return len(points)


def create_dn_splatter_dataset(records, config):
    dataset_dir = Path(config.dn_dataset_dir)
    if dataset_dir.exists() and config.overwrite:
        shutil.rmtree(dataset_dir)
    images_dir = dataset_dir / "images"
    depth_dir = dataset_dir / "depth"
    confidence_dir = dataset_dir / "confidence"
    for path in [images_dir, depth_dir, confidence_dir]:
        path.mkdir(parents=True, exist_ok=True)
    frames = []
    for record in records:
        stem = f"frame_{record.frame_id:06d}"
        image_path = images_dir / f"{stem}.jpg"
        depth_path = depth_dir / f"{stem}.png"
        confidence_path = confidence_dir / f"{stem}.png"
        shutil.copy2(record.rgb_path, image_path)
        depth = load_array(record.depth_path)
        confidence = load_array(record.confidence_path)
        if not cv2.imwrite(str(depth_path), depth.astype(np.uint16)):
            raise RuntimeError(f"Could not write dataset depth frame: {depth_path}")
        if not cv2.imwrite(str(confidence_path), confidence.astype(np.uint8)):
            raise RuntimeError(f"Could not write dataset confidence frame: {confidence_path}")
        frames.append({
            "file_path": str(image_path.relative_to(dataset_dir)).replace("\\", "/"),
            "depth_file_path": str(depth_path.relative_to(dataset_dir)).replace("\\", "/"),
            "confidence_file_path": str(confidence_path.relative_to(dataset_dir)).replace("\\", "/"),
            "transform_matrix": record.pose_c2w.tolist(),
            "stray_frame_id": record.frame_id,
            "timestamp": record.timestamp,
            "fl_x": float(record.intrinsic[0, 0]),
            "fl_y": float(record.intrinsic[1, 1]),
            "cx": float(record.intrinsic[0, 2]),
            "cy": float(record.intrinsic[1, 2]),
            "w": record.width,
            "h": record.height,
        })
    pointcloud_path = dataset_dir / "stray_depth_init_points.ply"
    point_count = build_depth_initial_pointcloud(records, config, pointcloud_path)
    transforms = {
        "camera_model": "OPENCV",
        "depth_unit_scale_factor": config.depth_unit_scale_factor,
        "pose_convention": "camera_to_world",
        "coordinate_convention": config.pose_conversion,
        "ply_file_path": pointcloud_path.name,
        "frames": frames,
    }
    transforms_path = write_json(dataset_dir / "transforms.json", transforms)
    manifest = {
        "dataset_dir": str(dataset_dir),
        "transforms_path": str(transforms_path),
        "selected_frame_count": len(records),
        "image_count": len(list(images_dir.glob("*.jpg"))),
        "depth_count": len(list(depth_dir.glob("*.png"))),
        "confidence_count": len(list(confidence_dir.glob("*.png"))),
        "initial_pointcloud_path": str(pointcloud_path),
        "initial_pointcloud_points": point_count,
        "dn_splatter_parser": "normal-nerfstudio",
        "dn_depth_key": "depth_file_path",
        "confidence_note": (
            "Stray confidence is preserved and used for validation and point-cloud filtering. "
            "It is not passed as AGS-Mesh depth-normal consistency masks."
        ),
        "final_artifact": "Gaussian PLY, not triangle mesh",
    }
    write_json(Path(config.reports_dir) / "dn_splatter_dataset_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return dataset_dir, transforms_path, pointcloud_path


dn_dataset_dir, transforms_path, initial_pointcloud_path = create_dn_splatter_dataset(selected_records, config)
"""
    ),
    md("## 15. DN-Splatter Installation"),
    code(
        r"""
print("[Step 15] Installing official DN-Splatter")
run_command("nvidia-smi", check=False)
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

run_command("apt-get -qq update", log_path=Path(config.logs_dir) / "apt_update.log")
run_command("apt-get -qq install -y ffmpeg", log_path=Path(config.logs_dir) / "apt_install_ffmpeg.log")
run_command(
    f"{shlex.quote(sys.executable)} -m pip install setuptools==69.5.1",
    log_path=Path(config.logs_dir) / "pip_setuptools.log",
)

dn_repo_dir = Path("/content/dn-splatter")
if (dn_repo_dir / ".git").exists():
    run_command("git pull --ff-only", cwd=dn_repo_dir, log_path=Path(config.logs_dir) / "dn_git_pull.log")
else:
    run_command(
        "git clone --depth 1 https://github.com/maturk/dn-splatter /content/dn-splatter",
        log_path=Path(config.logs_dir) / "dn_git_clone.log",
    )

# DN-Splatter's official pyproject declares vdbfusion as a required dependency.
# PyPI does not publish a CPython 3.12 vdbfusion wheel for current Colab runtimes.
# vdbfusion is used by triangle-mesh TSDF export, not by DN-Splatter training or
# Nerfstudio Gaussian PLY export. Install the pinned training core explicitly,
# then register the official repository without resolving mesh-only dependencies.
run_command(
    (
        f"{shlex.quote(sys.executable)} -m pip install "
        "\"nerfstudio==1.1.3\" \"gsplat==1.0.0\" "
        "\"black==22.3.0\" natsort pytest geffnet"
    ),
    log_path=Path(config.logs_dir) / "pip_install_dn_splatter_training_core.log",
)
run_command(
    f"{shlex.quote(sys.executable)} -m pip install -e . --no-deps",
    cwd=dn_repo_dir,
    log_path=Path(config.logs_dir) / "pip_install_dn_splatter.log",
)

# DN-Splatter imports Omnidata predictor modules while registering dataparsers,
# even when sensor-depth normal supervision is used and no mono normals are generated.
omnidata_repo_dir = Path("/content/omnidata")
if (omnidata_repo_dir / ".git").exists():
    run_command("git pull --ff-only", cwd=omnidata_repo_dir, log_path=Path(config.logs_dir) / "omnidata_git_pull.log")
else:
    run_command(
        "git clone --depth 1 https://github.com/EPFL-VILAB/omnidata /content/omnidata",
        log_path=Path(config.logs_dir) / "omnidata_git_clone.log",
    )
site_packages_dir = Path(
    capture_command(
        f"{shlex.quote(sys.executable)} -c \"import site; print(site.getsitepackages()[0])\""
    ).strip().splitlines()[-1]
)
omnidata_pth_path = site_packages_dir / "dn_splatter_omnidata.pth"
omnidata_pth_path.write_text(str(omnidata_repo_dir) + "\n", encoding="utf-8")
print("Registered Omnidata Python path:", omnidata_pth_path)
print("Official pyproject pins: nerfstudio==1.1.3 and gsplat==1.0.0")
print("Skipped vdbfusion: unavailable for Colab CPython 3.12 and only needed for triangle-mesh TSDF export.")
print("Installed geffnet: imported by DN-Splatter DSINE registration even when depth normal supervision is selected.")
print("Registered EPFL-VILAB/omnidata: imported by DN-Splatter Omnidata registration even when mono normals are not generated.")
print("Pipeline artifact remains Gaussian PLY, not triangle mesh.")
"""
    ),
    md("## 16. DN-Splatter CLI Validation"),
    code(
        r"""
print("[Step 16] Validating installed CLI registrations and options")
ns_train_help = capture_command("ns-train --help", log_path=Path(config.logs_dir) / "ns_train_help.log")
dn_help = capture_command("ns-train dn-splatter --help", log_path=Path(config.logs_dir) / "ns_train_dn_splatter_help.log")
normal_parser_help = capture_command(
    "ns-train dn-splatter normal-nerfstudio --help",
    log_path=Path(config.logs_dir) / "ns_train_dn_splatter_normal_nerfstudio_help.log",
)
export_help = capture_command(
    "ns-export gaussian-splat --help",
    log_path=Path(config.logs_dir) / "ns_export_gaussian_splat_help.log",
)


def require_cli_flags(help_text, flags, context):
    missing = [flag for flag in flags if flag not in help_text]
    if missing:
        raise RuntimeError(f"Installed CLI help for {context} is missing expected flags: {missing}")


require_cli_flags(
    dn_help,
    [
        "--pipeline.model.use-depth-loss",
        "--pipeline.model.depth-lambda",
        "--pipeline.model.use-normal-loss",
        "--pipeline.model.use-normal-tv-loss",
        "--pipeline.model.normal-supervision",
        "--max-num-iterations",
        "--output-dir",
        "--steps-per-save",
        "--load-dir",
    ],
    "ns-train dn-splatter",
)
require_cli_flags(
    normal_parser_help,
    ["--data", "--eval-mode", "--depth-unit-scale-factor", "--load-normals", "--load-depths", "--load-pcd-normals"],
    "normal-nerfstudio",
)
require_cli_flags(export_help, ["--load-config", "--output-dir"], "ns-export gaussian-splat")
print("DN-Splatter CLI validation passed.")
"""
    ),
    md("## 17. DN-Splatter Training"),
    code(
        r"""
def latest_checkpoint_dir(output_dir):
    checkpoints = sorted(Path(output_dir).rglob("*.ckpt"), key=lambda path: path.stat().st_mtime)
    return checkpoints[-1].parent if checkpoints else None


def build_dn_train_command(config, dataset_dir):
    args = [
        "ns-train", "dn-splatter",
        "--pipeline.model.use-depth-loss", str(config.dn_splatter_use_depth_loss),
        "--pipeline.model.depth-lambda", str(config.dn_splatter_depth_lambda),
        "--pipeline.model.depth-loss-type", config.dn_splatter_depth_loss_type,
        "--pipeline.model.use-normal-loss", str(config.dn_splatter_use_normal_loss),
        "--pipeline.model.use-normal-tv-loss", str(config.dn_splatter_use_normal_tv_loss),
        "--pipeline.model.normal-supervision", config.dn_splatter_normal_supervision,
        "--max-num-iterations", str(config.dn_splatter_max_num_iterations),
        "--steps-per-save", str(config.checkpoint_every_iterations),
        "--output-dir", config.dn_output_dir,
        "--vis", "tensorboard",
    ]
    checkpoint_dir = latest_checkpoint_dir(config.dn_output_dir)
    if config.resume and checkpoint_dir is not None:
        args.extend(["--load-dir", str(checkpoint_dir)])
        print("Resume checkpoint directory:", checkpoint_dir)
    elif config.resume:
        print("Resume requested, but no prior checkpoint exists. Starting a fresh run.")
    args.extend([
        "normal-nerfstudio",
        "--data", str(dataset_dir),
        "--eval-mode", config.dn_splatter_eval_mode,
        "--depth-unit-scale-factor", str(config.depth_unit_scale_factor),
        "--load-normals", "False",
        "--load-depths", "True",
        "--load-pcd-normals", "True",
    ])
    return shlex.join(args)


print("[Step 17] Building DN-Splatter training command")
train_command = build_dn_train_command(config, dn_dataset_dir)
print(train_command)
if config.dry_run:
    print("dry_run=True: dataset conversion and CLI validation completed; training command was not executed.")
else:
    run_command(
        train_command,
        log_path=Path(config.logs_dir) / "dn_splatter_training.log",
        env={**os.environ, "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
    )
"""
    ),
    md("## 18. Gaussian PLY Export"),
    code(
        r"""
def latest_training_config(output_dir):
    configs = sorted(Path(output_dir).rglob("config.yml"), key=lambda path: path.stat().st_mtime)
    return configs[-1] if configs else None


def locate_gaussian_ply(export_dir):
    candidates = sorted(Path(export_dir).rglob("*.ply"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise RuntimeError(f"Gaussian PLY export did not create a .ply file under: {export_dir}")
    return candidates[-1]


print("[Step 18] Exporting Gaussian PLY")
training_config_path = latest_training_config(config.dn_output_dir)
gaussian_ply_path = None
if config.dry_run:
    print(
        "dry_run=True: export will run after training with:\n"
        f"ns-export gaussian-splat --load-config <latest config.yml under {config.dn_output_dir}> "
        f"--output-dir {config.gaussian_export_dir}"
    )
else:
    if training_config_path is None:
        raise RuntimeError(f"No DN-Splatter config.yml found under: {config.dn_output_dir}")
    export_command = shlex.join([
        "ns-export", "gaussian-splat",
        "--load-config", str(training_config_path),
        "--output-dir", config.gaussian_export_dir,
    ])
    run_command(
        export_command,
        log_path=Path(config.logs_dir) / "ns_export_gaussian_splat.log",
        env={**os.environ, "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
    )
    gaussian_ply_path = locate_gaussian_ply(config.gaussian_export_dir)
    print("Gaussian PLY:", gaussian_ply_path)
    print("Artifact type: Gaussian splat PLY, not triangle mesh.")
"""
    ),
    md("## 19. Training Validation"),
    code(
        r"""
print("[Step 19] Writing training validation report")
checkpoint_paths = sorted(str(path) for path in Path(config.dn_output_dir).rglob("*.ckpt"))
training_validation_report = {
    "dry_run": config.dry_run,
    "training_executed": not config.dry_run,
    "training_command": train_command,
    "training_config_path": str(training_config_path) if training_config_path else None,
    "checkpoint_count": len(checkpoint_paths),
    "checkpoints": checkpoint_paths,
    "gaussian_ply_path": str(gaussian_ply_path) if gaussian_ply_path else None,
    "gaussian_ply_exists": bool(gaussian_ply_path and Path(gaussian_ply_path).exists()),
    "final_artifact": "Gaussian PLY, not triangle mesh",
}
if not config.dry_run and not training_validation_report["gaussian_ply_exists"]:
    raise RuntimeError("Training completed but Gaussian PLY export is missing.")
write_json(Path(config.reports_dir) / "training_validation_report.json", training_validation_report)
print(json.dumps(training_validation_report, indent=2))
"""
    ),
    md("## 20-22. Result Packaging, Drive Storage, And Optional Download"),
    code(
        r"""
REPORT_NAMES = [
    "stray_input_manifest.json",
    "frame_alignment_report.json",
    "frame_filter_report.json",
    "pose_depth_validation_report.json",
    "dn_splatter_dataset_manifest.json",
    "training_validation_report.json",
    "config_snapshot.json",
]


def package_results(config, gaussian_ply_path, transforms_path):
    print("[Steps 20-22] Packaging result ZIP in persistent project storage")
    result_dir = Path(config.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = result_dir / "staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    for name in REPORT_NAMES:
        source = Path(config.reports_dir) / name
        if not source.exists():
            raise RuntimeError(f"Missing required report before packaging: {source}")
        shutil.copy2(source, staging_dir / name)
    shutil.copy2(transforms_path, staging_dir / "transforms.json")
    logs_target = staging_dir / "logs"
    if Path(config.logs_dir).exists():
        shutil.copytree(config.logs_dir, logs_target, dirs_exist_ok=True)
    if gaussian_ply_path:
        shutil.copy2(gaussian_ply_path, staging_dir / "gaussians.ply")
    result_manifest = {
        "job_id": config.job_id,
        "scene_name": config.scene_name,
        "created_at": datetime.now().isoformat(),
        "dry_run": config.dry_run,
        "final_artifact": "Gaussian PLY, not triangle mesh",
        "gaussian_ply_included": bool(gaussian_ply_path),
        "package_files": sorted(
            str(path.relative_to(staging_dir)).replace("\\", "/")
            for path in staging_dir.rglob("*")
            if path.is_file()
        ),
    }
    write_json(staging_dir / "result_manifest.json", result_manifest)
    write_json(Path(config.reports_dir) / "result_manifest.json", result_manifest)
    zip_path = result_dir / f"{config.job_id}_stray_dn_splatter_result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(staging_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(staging_dir))
    print("Result ZIP:", zip_path)
    print("Google Drive backed:", config.use_drive)
    if config.optional_colab_download:
        try:
            from google.colab import files
            files.download(str(zip_path))
        except Exception as exc:
            print("Automatic download skipped or failed:", repr(exc))
            print("Download manually from:", zip_path)
    return zip_path


result_zip_path = package_results(config, gaussian_ply_path, transforms_path)
print("Done:", result_zip_path)
if config.dry_run:
    print("This is a dry-run diagnostics package. Set config.dry_run=False and rerun from the top for training and Gaussian PLY export.")
"""
    ),
    md(
        r"""
## Output Summary

Persistent project outputs:

```text
capstone_3dgs_project/
  data/dn_splatter/<job_id>/
    images/
    depth/
    confidence/
    transforms.json
    stray_depth_init_points.ply
  outputs/dn_splatter/<job_id>/
    ... DN-Splatter checkpoints and config.yml ...
  exports/gaussian_ply/<job_id>/
    ... exported Gaussian PLY ...
  runs/<job_id>/
    logs/
    reports/
    result_package/
      <job_id>_stray_dn_splatter_result.zip
```

The result ZIP contains `gaussians.ply` after a real training run. With `dry_run=True`, the ZIP is a diagnostics package and intentionally has no trained PLY.

The notebook never runs COLMAP feature extraction, COLMAP matching, COLMAP mapping, `ns-process-data video`, or `ns-train splatfacto`.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": NOTEBOOK_PATH.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH} with {len(cells)} cells.")
