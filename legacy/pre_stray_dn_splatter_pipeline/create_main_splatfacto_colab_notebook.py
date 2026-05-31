import json
from pathlib import Path


PROJECT_ROOT = Path("capstone_3dgs_project")
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_main_splatfacto_pipeline.ipynb"


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
# Main Splatfacto Pipeline

Smartphone RGB video -> Nerfstudio COLMAP preprocessing -> Splatfacto 3DGS training -> Gaussian PLY export.

This notebook is intentionally limited to the stable Nerfstudio path. Experimental reconstruction, pose-estimation, SLAM comparison, and custom trainer routes are kept in separate notebooks.

Run cells from top to bottom after a fresh Colab runtime.
"""
    ),
    md("## 1. Environment Check"),
    code(
        r"""
import os
import sys
import json
import time
import shutil
import zipfile
import platform
import subprocess
import shlex
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional


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
@dataclass
class PipelineConfig:
    use_drive: bool = True
    video_source: Literal["drive_path", "upload"] = "drive_path"
    input_video_path: str = "/content/drive/MyDrive/capstone_3dgs_project/input/videos/input_video.mp4"
    job_id: str = "smartphone_room_001"
    scene_name: str = "smartphone_room_001"
    max_video_seconds: int = 300

    frame_fps: float = 2.0
    max_preview_frames: int = 240
    preview_resize_long_side: Optional[int] = 1280

    ns_process_downscale_factor: int = 2
    ns_process_matching_method: Literal["vocab_tree", "exhaustive", "sequential"] = "vocab_tree"
    training_input_source: Literal["ns_process_data", "colmap_undistorted"] = "ns_process_data"

    colmap_frame_fps: float = 2.0
    colmap_max_frames: Optional[int] = 300
    colmap_resize_long_side: Optional[int] = 1600
    colmap_sequential_overlap: int = 15
    colmap_use_gpu: int = 1

    splatfacto_max_num_iterations: int = 30000
    splatfacto_viewer_quit_on_train_completion: bool = True

    overwrite: bool = False
    project_root: Optional[str] = None

    def resolve(self):
        if self.project_root is None:
            self.project_root = (
                "/content/drive/MyDrive/capstone_3dgs_project"
                if self.use_drive
                else "/content/capstone_3dgs_project"
            )
        root = Path(self.project_root)
        self.input_videos_dir = str(root / "input" / "videos")
        self.nerfstudio_data_root = str(root / "data" / "nerfstudio")
        self.splatfacto_outputs_root = str(root / "outputs" / "splatfacto")
        self.gaussian_export_root = str(root / "exports" / "gaussian_ply")
        self.mesh_export_root = str(root / "exports" / "mesh")
        self.colmap_work_root = str(root / "work")
        self.colmap_scene_root = str(Path(self.colmap_work_root) / self.scene_name)
        self.job_root = str(root / "runs" / self.job_id)
        self.logs_dir = str(Path(self.job_root) / "logs")
        self.prepared_video_path = str(Path(self.input_videos_dir) / f"{self.job_id}_input_video.mp4")
        self.preview_frames_dir = str(Path(self.job_root) / "frame_quality_preview")
        self.ns_data_dir = str(Path(self.nerfstudio_data_root) / self.job_id)
        self.colmap_undistorted_data_dir = str(Path(self.colmap_scene_root) / "colmap" / "undistorted")
        self.ns_output_dir = str(Path(self.splatfacto_outputs_root) / self.job_id)
        self.gaussian_export_dir = str(Path(self.gaussian_export_root) / self.job_id)
        self.mesh_export_dir = str(Path(self.mesh_export_root) / self.job_id)
        self.result_dir = str(Path(self.job_root) / "result_package")
        return self


config = PipelineConfig(
    use_drive=True,
    video_source="drive_path",
    input_video_path="/content/drive/MyDrive/capstone_3dgs_project/input/videos/input_video.mp4",
    job_id="smartphone_room_001",
    scene_name="smartphone_room_001",
    frame_fps=2.0,
    max_preview_frames=240,
    ns_process_downscale_factor=2,
    ns_process_matching_method="vocab_tree",
    training_input_source="ns_process_data",  # "ns_process_data" or "colmap_undistorted"
    colmap_frame_fps=2.0,
    colmap_max_frames=300,
    colmap_resize_long_side=1600,
    colmap_sequential_overlap=15,
    colmap_use_gpu=1,
    splatfacto_max_num_iterations=30000,
).resolve()

if config.use_drive:
    from google.colab import drive
    drive.mount("/content/drive")
    config.project_root = "/content/drive/MyDrive/capstone_3dgs_project"
else:
    config.project_root = "/content/capstone_3dgs_project"

config.resolve()
print(json.dumps(asdict(config), indent=2, ensure_ascii=False))
"""
    ),
    md("## 3. Input Video Path Setup"),
    code(
        r"""
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}

for path in [
    config.input_videos_dir,
    config.nerfstudio_data_root,
    config.splatfacto_outputs_root,
    config.gaussian_export_root,
    config.mesh_export_root,
    config.colmap_work_root,
    config.colmap_scene_root,
    config.job_root,
    config.logs_dir,
    config.result_dir,
]:
    Path(path).mkdir(parents=True, exist_ok=True)


def prepare_input_video(config):
    if config.video_source == "upload":
        from google.colab import files
        uploaded = files.upload()
        if not uploaded or len(uploaded) != 1:
            raise ValueError("Upload exactly one video file.")
        source_path = Path("/content") / next(iter(uploaded.keys()))
    elif config.video_source == "drive_path":
        source_path = Path(config.input_video_path)
    else:
        raise ValueError("config.video_source must be 'drive_path' or 'upload'.")

    if not source_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {source_path}")
    if source_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {source_path.suffix}")

    prepared_path = Path(config.prepared_video_path)
    prepared_path = prepared_path.with_suffix(source_path.suffix.lower())
    config.prepared_video_path = str(prepared_path)
    if source_path.resolve() != prepared_path.resolve():
        if prepared_path.exists() and not config.overwrite:
            print(f"Using existing prepared video: {prepared_path}")
        else:
            shutil.copy2(source_path, prepared_path)
            print(f"Copied input video to: {prepared_path}")

    metadata = {
        "source_path": str(source_path),
        "prepared_video_path": str(prepared_path),
        "extension": source_path.suffix.lower(),
        "size_mb": prepared_path.stat().st_size / (1024 * 1024),
    }
    metadata_path = Path(config.job_root) / "input_video_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return prepared_path


input_video_path = prepare_input_video(config)
"""
    ),
    md("## 4. FFmpeg / COLMAP / Nerfstudio Install"),
    code(
        r"""
run_command("apt-get update -y", log_path=Path(config.logs_dir) / "apt_update.log")
run_command("apt-get install -y ffmpeg colmap libgl1-mesa-glx xvfb", log_path=Path(config.logs_dir) / "apt_install.log")

run_command(f"{sys.executable} -m pip install --upgrade pip setuptools wheel", log_path=Path(config.logs_dir) / "pip_upgrade.log")
run_command(f"{sys.executable} -m pip install nerfstudio", log_path=Path(config.logs_dir) / "pip_install_nerfstudio.log")

run_command("ffmpeg -version", check=False)
run_command("colmap -h", check=False)
run_command("ns-process-data --help", check=False)
run_command("ns-train splatfacto --help", check=False)
run_command("ns-export gaussian-splat --help", check=False)

# COLMAP uses Qt internally. Colab has no display server, so ns-process-data is run through xvfb-run later.
os.environ.pop("QT_QPA_PLATFORM", None)
print("xvfb-run is available for headless COLMAP execution.")
"""
    ),
    md("## 5. Video Frame Quality Check"),
    code(
        r"""
import cv2
import math
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from IPython.display import display


def video_metadata(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or frame_count <= 0:
        raise ValueError(f"Invalid video metadata: fps={fps}, frame_count={frame_count}")
    duration_sec = frame_count / fps
    if duration_sec > config.max_video_seconds:
        raise ValueError(f"Video duration {duration_sec:.2f}s exceeds {config.max_video_seconds}s.")
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "width": width,
        "height": height,
    }


def extract_quality_preview_frames(config, input_video_path):
    meta = video_metadata(input_video_path)
    preview_dir = Path(config.preview_frames_dir)
    if preview_dir.exists() and config.overwrite:
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video_path))
    requested = max(1, int(math.floor(meta["duration_sec"] * config.frame_fps)))
    indices = np.linspace(0, meta["frame_count"] - 1, num=requested, dtype=np.int64)
    if config.max_preview_frames and len(indices) > config.max_preview_frames:
        keep = np.linspace(0, len(indices) - 1, num=config.max_preview_frames, dtype=np.int64)
        indices = indices[keep]
    indices = np.unique(indices)

    frames = []
    blur_scores = []
    brightness_scores = []
    for out_idx, frame_idx in enumerate(tqdm(indices, desc="Extracting preview frames"), start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise ValueError(f"Failed to read frame {frame_idx}")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if config.preview_resize_long_side:
            h, w = rgb.shape[:2]
            scale = min(1.0, float(config.preview_resize_long_side) / max(h, w))
            if scale < 1.0:
                rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        image_path = preview_dir / f"frame_{out_idx:06d}.jpg"
        Image.fromarray(rgb).save(image_path, quality=95)
        frames.append({
            "image_path": str(image_path),
            "source_frame_index": int(frame_idx),
            "blur_laplacian_var": blur,
            "brightness_mean": brightness,
        })
        blur_scores.append(blur)
        brightness_scores.append(brightness)
    cap.release()

    report = {
        "video": meta,
        "preview_frame_count": len(frames),
        "blur_laplacian_var_min": min(blur_scores) if blur_scores else None,
        "blur_laplacian_var_median": float(np.median(blur_scores)) if blur_scores else None,
        "brightness_mean_min": min(brightness_scores) if brightness_scores else None,
        "brightness_mean_max": max(brightness_scores) if brightness_scores else None,
        "frames": frames,
    }
    report_path = Path(config.job_root) / "frame_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2, ensure_ascii=False))
    return report


quality_report = extract_quality_preview_frames(config, input_video_path)

sample_paths = [item["image_path"] for item in quality_report["frames"][:3]]
for path in sample_paths:
    display(Image.open(path))
"""
    ),
    md("## 6. COLMAP preprocessing baseline based on Nannigalaxy repo"),
    code(
        r"""
# This section reconstructs only the preprocessing logic inspired by
# Nannigalaxy/video-3d-reconstruction-gsplat:
# FFmpeg frames -> COLMAP feature_extractor -> sequential_matcher -> mapper
# -> image_undistorter -> model_converter.
#
# It intentionally does not clone the repository, initialize submodules, install
# Speedy-Splat, or run train_speedy_splat.sh.

SCENE_NAME = config.scene_name
USE_GPU = config.colmap_use_gpu  # Set to 0 and rerun this section if COLMAP reports CUDA/GPU errors.
SEQUENTIAL_OVERLAP = config.colmap_sequential_overlap

scene_root = Path(config.colmap_scene_root)
baseline_images_dir = scene_root / "images"
baseline_colmap_dir = scene_root / "colmap"
baseline_database_path = baseline_colmap_dir / "database.db"
baseline_sparse_root = baseline_colmap_dir / "sparse"
baseline_sparse_dir = baseline_sparse_root / "0"
baseline_undistorted_dir = baseline_colmap_dir / "undistorted"
baseline_custom_export_dir = baseline_colmap_dir / "custom_export"
baseline_logs_dir = scene_root / "logs"

for path in [
    baseline_images_dir,
    baseline_colmap_dir,
    baseline_sparse_root,
    baseline_undistorted_dir,
    baseline_custom_export_dir,
    baseline_logs_dir,
]:
    path.mkdir(parents=True, exist_ok=True)

baseline_layout = {
    "scene_root": str(scene_root),
    "images": str(baseline_images_dir),
    "database": str(baseline_database_path),
    "sparse_0": str(baseline_sparse_dir),
    "undistorted": str(baseline_undistorted_dir),
    "custom_export": str(baseline_custom_export_dir),
    "logs": str(baseline_logs_dir),
    "use_gpu": USE_GPU,
    "sequential_overlap": SEQUENTIAL_OVERLAP,
}
print(json.dumps(baseline_layout, indent=2, ensure_ascii=False))
"""
    ),
    code(
        r"""
def extract_colmap_frames_with_ffmpeg(config, input_video_path, images_dir):
    images_dir = Path(images_dir)
    if images_dir.exists() and config.overwrite:
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(images_dir.glob("*.jpg"))
    if existing and not config.overwrite:
        print(f"Using {len(existing)} existing extracted frames in {images_dir}")
        return existing

    vf_parts = [f"fps={config.colmap_frame_fps}"]
    if config.colmap_resize_long_side:
        long_side = int(config.colmap_resize_long_side)
        vf_parts.append(
            "scale="
            f"'if(gt(iw,ih),min({long_side},iw),-2)':"
            f"'if(gt(ih,iw),min({long_side},ih),-2)'"
        )
    frame_pattern = images_dir / "frame_%06d.jpg"
    extract_cmd = (
        "ffmpeg -y "
        f"-i {shlex.quote(str(input_video_path))} "
        f"-vf {shlex.quote(','.join(vf_parts))} "
        "-q:v 2 "
        f"{shlex.quote(str(frame_pattern))}"
    )
    run_command(extract_cmd, log_path=baseline_logs_dir / "ffmpeg_frame_extraction.log")

    frames = sorted(images_dir.glob("*.jpg"))
    if config.colmap_max_frames and len(frames) > config.colmap_max_frames:
        keep_indices = set(np.linspace(0, len(frames) - 1, num=config.colmap_max_frames, dtype=np.int64).tolist())
        for idx, frame_path in enumerate(frames):
            if idx not in keep_indices:
                frame_path.unlink()
        frames = sorted(images_dir.glob("*.jpg"))
        for new_idx, frame_path in enumerate(frames, start=1):
            target = images_dir / f"frame_{new_idx:06d}.jpg"
            if frame_path != target:
                frame_path.rename(target)
        frames = sorted(images_dir.glob("*.jpg"))

    if len(frames) < 8:
        raise ValueError(f"Only {len(frames)} frames were extracted. Increase colmap_frame_fps or use a longer video.")
    print(f"Extracted COLMAP frames: {len(frames)}")
    return frames


baseline_frames = extract_colmap_frames_with_ffmpeg(config, input_video_path, baseline_images_dir)
"""
    ),
    code(
        r"""
def run_colmap_preprocessing_baseline():
    if baseline_database_path.exists() and config.overwrite:
        baseline_database_path.unlink()
    if baseline_sparse_dir.exists() and config.overwrite:
        shutil.rmtree(baseline_sparse_dir)
    if baseline_undistorted_dir.exists() and config.overwrite:
        shutil.rmtree(baseline_undistorted_dir)
    if baseline_custom_export_dir.exists() and config.overwrite:
        shutil.rmtree(baseline_custom_export_dir)

    baseline_sparse_dir.mkdir(parents=True, exist_ok=True)
    baseline_undistorted_dir.mkdir(parents=True, exist_ok=True)
    baseline_custom_export_dir.mkdir(parents=True, exist_ok=True)

    feature_cmd = (
        "xvfb-run -a colmap feature_extractor "
        f"--database_path {shlex.quote(str(baseline_database_path))} "
        f"--image_path {shlex.quote(str(baseline_images_dir))} "
        "--ImageReader.single_camera 1 "
        f"--SiftExtraction.use_gpu {USE_GPU}"
    )
    run_command(feature_cmd, log_path=baseline_logs_dir / "colmap_feature_extractor.log")

    matcher_cmd = (
        "xvfb-run -a colmap sequential_matcher "
        f"--database_path {shlex.quote(str(baseline_database_path))} "
        f"--SequentialMatching.overlap {SEQUENTIAL_OVERLAP} "
        f"--SiftMatching.use_gpu {USE_GPU}"
    )
    run_command(matcher_cmd, log_path=baseline_logs_dir / "colmap_sequential_matcher.log")

    mapper_cmd = (
        "xvfb-run -a colmap mapper "
        f"--database_path {shlex.quote(str(baseline_database_path))} "
        f"--image_path {shlex.quote(str(baseline_images_dir))} "
        f"--output_path {shlex.quote(str(baseline_sparse_root))}"
    )
    run_command(mapper_cmd, log_path=baseline_logs_dir / "colmap_mapper.log")

    if not baseline_sparse_dir.exists():
        sparse_candidates = sorted(p for p in baseline_sparse_root.iterdir() if p.is_dir())
        if not sparse_candidates:
            raise FileNotFoundError(f"COLMAP mapper did not create a sparse model under {baseline_sparse_root}")
        print(f"Using first mapper output as sparse/0: {sparse_candidates[0]}")
        if sparse_candidates[0] != baseline_sparse_dir:
            sparse_candidates[0].rename(baseline_sparse_dir)

    undistort_cmd = (
        "xvfb-run -a colmap image_undistorter "
        f"--image_path {shlex.quote(str(baseline_images_dir))} "
        f"--input_path {shlex.quote(str(baseline_sparse_dir))} "
        f"--output_path {shlex.quote(str(baseline_undistorted_dir))} "
        "--output_type COLMAP"
    )
    run_command(undistort_cmd, log_path=baseline_logs_dir / "colmap_image_undistorter.log")

    undistorted_sparse_root = baseline_undistorted_dir / "sparse"
    undistorted_sparse_dir = undistorted_sparse_root / "0"
    if not undistorted_sparse_dir.exists():
        root_bin_files = [undistorted_sparse_root / name for name in ["cameras.bin", "images.bin", "points3D.bin"]]
        if all(path.exists() for path in root_bin_files):
            undistorted_sparse_dir.mkdir(parents=True, exist_ok=True)
            for path in root_bin_files:
                shutil.copy2(path, undistorted_sparse_dir / path.name)
        else:
            sparse_candidates = sorted(p for p in undistorted_sparse_root.iterdir() if p.is_dir()) if undistorted_sparse_root.exists() else []
            if sparse_candidates:
                sparse_candidates[0].rename(undistorted_sparse_dir)

    ply_cmd = (
        "xvfb-run -a colmap model_converter "
        f"--input_path {shlex.quote(str(baseline_sparse_dir))} "
        f"--output_path {shlex.quote(str(baseline_custom_export_dir / 'scene.ply'))} "
        "--output_type PLY"
    )
    run_command(ply_cmd, log_path=baseline_logs_dir / "colmap_model_converter_ply.log")

    txt_cmd = (
        "xvfb-run -a colmap model_converter "
        f"--input_path {shlex.quote(str(baseline_sparse_dir))} "
        f"--output_path {shlex.quote(str(baseline_custom_export_dir))} "
        "--output_type TXT"
    )
    run_command(txt_cmd, log_path=baseline_logs_dir / "colmap_model_converter_txt.log")

    return {
        "sparse_dir": str(baseline_sparse_dir),
        "undistorted_dir": str(baseline_undistorted_dir),
        "custom_export_dir": str(baseline_custom_export_dir),
    }


baseline_colmap_outputs = run_colmap_preprocessing_baseline()
print(json.dumps(baseline_colmap_outputs, indent=2, ensure_ascii=False))
"""
    ),
    md("## 7. Validate COLMAP preprocessing baseline"),
    code(
        r"""
def count_registered_images_from_colmap_txt(images_txt):
    images_txt = Path(images_txt)
    if not images_txt.exists():
        return 0
    count = 0
    for line in images_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 10:
            count += 1
    return count


def validate_colmap_preprocessing_baseline():
    sparse_required = [baseline_sparse_dir / name for name in ["cameras.bin", "images.bin", "points3D.bin"]]
    undistorted_sparse_dir = baseline_undistorted_dir / "sparse" / "0"
    export_required = [
        baseline_custom_export_dir / "scene.ply",
        baseline_custom_export_dir / "cameras.txt",
        baseline_custom_export_dir / "images.txt",
        baseline_custom_export_dir / "points3D.txt",
    ]
    report = {
        "scene_root": str(scene_root),
        "sparse_0_exists": baseline_sparse_dir.exists(),
        "sparse_bin_files": {path.name: path.exists() for path in sparse_required},
        "undistorted_images_exists": (baseline_undistorted_dir / "images").exists(),
        "undistorted_sparse_0_exists": undistorted_sparse_dir.exists(),
        "registered_image_count": count_registered_images_from_colmap_txt(baseline_custom_export_dir / "images.txt"),
        "scene_ply_exists": (baseline_custom_export_dir / "scene.ply").exists(),
        "custom_export_files": {path.name: path.exists() for path in export_required},
    }
    report_path = baseline_logs_dir / "colmap_preprocessing_baseline_validation.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if not report["sparse_0_exists"]:
        raise FileNotFoundError(baseline_sparse_dir)
    missing_sparse = [name for name, exists in report["sparse_bin_files"].items() if not exists]
    if missing_sparse:
        raise FileNotFoundError(f"Missing sparse/0 files: {missing_sparse}")
    if report["registered_image_count"] < 8:
        raise ValueError("Too few registered images for robust Splatfacto training.")
    if not report["scene_ply_exists"]:
        raise FileNotFoundError(baseline_custom_export_dir / "scene.ply")
    return report


baseline_validation_report = validate_colmap_preprocessing_baseline()
"""
    ),
    md("## 8. Run ns-process-data video"),
    code(
        r"""
ns_data_dir = Path(config.ns_data_dir)
if ns_data_dir.exists() and config.overwrite:
    shutil.rmtree(ns_data_dir)
ns_data_dir.mkdir(parents=True, exist_ok=True)

process_cmd = (
    "xvfb-run -a ns-process-data video "
    f"--data {shlex.quote(str(input_video_path))} "
    f"--output-dir {shlex.quote(str(ns_data_dir))} "
    f"--matching-method {config.ns_process_matching_method}"
)
help_text = subprocess.run(
    "ns-process-data video --help",
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
).stdout
if "--num-downscales" in help_text:
    process_cmd += f" --num-downscales {config.ns_process_downscale_factor}"
elif "--downscale-factor" in help_text:
    process_cmd += f" --downscale-factor {config.ns_process_downscale_factor}"
else:
    print("No supported downscale option found in this Nerfstudio version; running without a downscale flag.")

run_command(process_cmd, log_path=Path(config.logs_dir) / "ns_process_data.log")
"""
    ),
    md("## 9. Validate ns-process-data COLMAP Results"),
    code(
        r"""
def count_images_in_transforms(transforms_path):
    data = json.loads(Path(transforms_path).read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    existing = []
    missing = []
    base = Path(transforms_path).parent
    for frame in frames:
        rel = frame.get("file_path")
        path = base / rel if rel else None
        if path and path.exists():
            existing.append(str(path))
        else:
            missing.append(rel)
    return data, existing, missing


def validate_colmap_scene(config):
    data_dir = Path(config.ns_data_dir)
    transforms_path = data_dir / "transforms.json"
    colmap_dir = data_dir / "colmap"
    sparse_root = colmap_dir / "sparse"

    required = [transforms_path, colmap_dir]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    transforms, existing_images, missing_images = count_images_in_transforms(transforms_path)
    sparse_candidates = sorted(sparse_root.glob("*/")) if sparse_root.exists() else []
    sparse_files = []
    for sparse_dir in sparse_candidates:
        sparse_files.extend(str(p) for p in sparse_dir.glob("*"))

    report = {
        "transforms_path": str(transforms_path),
        "frames_in_transforms": len(transforms.get("frames", [])),
        "existing_referenced_images": len(existing_images),
        "missing_referenced_images": missing_images,
        "camera_model": transforms.get("camera_model"),
        "fl_x": transforms.get("fl_x"),
        "fl_y": transforms.get("fl_y"),
        "cx": transforms.get("cx"),
        "cy": transforms.get("cy"),
        "w": transforms.get("w"),
        "h": transforms.get("h"),
        "sparse_dirs": [str(p) for p in sparse_candidates],
        "sparse_files": sparse_files,
    }
    report_path = Path(config.job_root) / "colmap_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if report["frames_in_transforms"] < 8:
        raise ValueError("Too few registered frames for robust 3DGS training.")
    if missing_images:
        raise FileNotFoundError(f"transforms.json references missing images: {missing_images[:5]}")
    return report


colmap_report = validate_colmap_scene(config)
"""
    ),
    md("## 10. Select Splatfacto input dataset"),
    code(
        r"""
# Choose the input for Nerfstudio Splatfacto:
# A. "ns_process_data" uses the Nerfstudio ns-process-data video result.
# B. "colmap_undistorted" uses the COLMAP undistorted dataset generated above.
TRAINING_INPUT_SOURCE = config.training_input_source

if TRAINING_INPUT_SOURCE == "ns_process_data":
    splatfacto_data_dir = Path(config.ns_data_dir)
    required = [splatfacto_data_dir / "transforms.json"]
elif TRAINING_INPUT_SOURCE == "colmap_undistorted":
    splatfacto_data_dir = Path(config.colmap_undistorted_data_dir)
    required = [
        splatfacto_data_dir / "images",
        splatfacto_data_dir / "sparse" / "0" / "cameras.bin",
        splatfacto_data_dir / "sparse" / "0" / "images.bin",
        splatfacto_data_dir / "sparse" / "0" / "points3D.bin",
    ]
else:
    raise ValueError("TRAINING_INPUT_SOURCE must be 'ns_process_data' or 'colmap_undistorted'.")

missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError(f"Selected Splatfacto input is incomplete: {missing}")

selection_report = {
    "training_input_source": TRAINING_INPUT_SOURCE,
    "splatfacto_data_dir": str(splatfacto_data_dir),
    "required_paths": [str(path) for path in required],
}
selection_path = Path(config.job_root) / "splatfacto_input_selection.json"
selection_path.write_text(json.dumps(selection_report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(selection_report, indent=2, ensure_ascii=False))
"""
    ),
    md("## 11. Splatfacto Training"),
    code(
        r"""
ns_output_dir = Path(config.ns_output_dir)
ns_output_dir.mkdir(parents=True, exist_ok=True)

train_cmd = (
    "ns-train splatfacto "
    f"--data {shlex.quote(str(splatfacto_data_dir))} "
    f"--output-dir {ns_output_dir} "
    f"--max-num-iterations {config.splatfacto_max_num_iterations} "
    f"--viewer.quit-on-train-completion {str(config.splatfacto_viewer_quit_on_train_completion)}"
)
if TRAINING_INPUT_SOURCE == "colmap_undistorted":
    train_cmd += " colmap"
run_command(train_cmd, log_path=Path(config.logs_dir) / "ns_train_splatfacto.log")
"""
    ),
    md("## 12. Viewer / Checkpoint Check"),
    code(
        r"""
def find_latest_config(output_root):
    candidates = sorted(Path(output_root).rglob("config.yml"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No Nerfstudio config.yml found under {output_root}")
    return candidates[0]


def validate_training_outputs(config):
    latest_config = find_latest_config(config.ns_output_dir)
    run_dir = latest_config.parent
    checkpoint_dir = run_dir / "nerfstudio_models"
    checkpoints = sorted(checkpoint_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    report = {
        "latest_config": str(latest_config),
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_count": len(checkpoints),
        "latest_checkpoint": str(checkpoints[0]) if checkpoints else None,
        "viewer_command": f"ns-viewer --load-config {latest_config}",
    }
    report_path = Path(config.job_root) / "training_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not checkpoints:
        raise FileNotFoundError("No checkpoint files were found after training.")
    return report


training_report = validate_training_outputs(config)
"""
    ),
    md("## 13. Gaussian PLY Export"),
    code(
        r"""
gaussian_export_dir = Path(config.gaussian_export_dir)
if gaussian_export_dir.exists() and config.overwrite:
    shutil.rmtree(gaussian_export_dir)
gaussian_export_dir.mkdir(parents=True, exist_ok=True)

load_config = Path(training_report["latest_config"])
export_cmd = (
    "ns-export gaussian-splat "
    f"--load-config {load_config} "
    f"--output-dir {gaussian_export_dir}"
)
export_env = os.environ.copy()
# Nerfstudio checkpoints may include non-tensor metadata. PyTorch 2.6+ defaults
# torch.load to weights_only=True, so allow full loading for this trusted local checkpoint.
export_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
run_command(export_cmd, log_path=Path(config.logs_dir) / "ns_export_gaussian_splat.log", env=export_env)

gaussian_ply_candidates = sorted(gaussian_export_dir.rglob("*.ply"), key=lambda p: p.stat().st_mtime, reverse=True)
if not gaussian_ply_candidates:
    raise FileNotFoundError(f"No Gaussian PLY exported under {gaussian_export_dir}")

gaussian_ply_path = gaussian_ply_candidates[0]
print(f"Gaussian PLY: {gaussian_ply_path}")
"""
    ),
    md("## 14. Prepare Export for Mesh Extraction"),
    code(
        r"""
mesh_export_dir = Path(config.mesh_export_dir)
mesh_export_dir.mkdir(parents=True, exist_ok=True)

mesh_manifest = {
    "note": "This directory collects inputs useful for later mesh extraction. The PLY is a Gaussian Splat PLY, not a triangle mesh.",
    "gaussian_ply": str(gaussian_ply_path),
    "nerfstudio_config": training_report["latest_config"],
    "splatfacto_data_dir": str(splatfacto_data_dir),
    "training_input_source": TRAINING_INPUT_SOURCE,
    "colmap_validation_report": str(Path(config.job_root) / "colmap_validation_report.json"),
    "baseline_validation_report": str(Path(config.colmap_scene_root) / "logs" / "colmap_preprocessing_baseline_validation.json"),
}

shutil.copy2(gaussian_ply_path, mesh_export_dir / "gaussians_for_mesh_extraction.ply")
if (splatfacto_data_dir / "transforms.json").exists():
    shutil.copy2(splatfacto_data_dir / "transforms.json", mesh_export_dir / "transforms.json")
Path(mesh_export_dir / "mesh_extraction_manifest.json").write_text(
    json.dumps(mesh_manifest, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(json.dumps(mesh_manifest, indent=2, ensure_ascii=False))
"""
    ),
    md("## 15. Save Results to Drive"),
    code(
        r"""
def collect_failure_logs(config):
    logs_dir = Path(config.logs_dir)
    summaries = {}
    for log_path in sorted(logs_dir.glob("*.log")):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        suspicious = [
            line for line in lines
            if any(token in line.lower() for token in ["error", "failed", "traceback", "cuda out of memory", "exception"])
        ]
        summaries[log_path.name] = {
            "line_count": len(lines),
            "suspicious_tail": suspicious[-30:],
            "tail": lines[-40:],
        }
    path = Path(config.result_dir) / "failure_log_summary.json"
    path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def package_results(config, gaussian_ply_path, training_report):
    result_dir = Path(config.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = {
        "gaussians.ply": gaussian_ply_path,
        "input_video_metadata.json": Path(config.job_root) / "input_video_metadata.json",
        "frame_quality_report.json": Path(config.job_root) / "frame_quality_report.json",
        "colmap_validation_report.json": Path(config.job_root) / "colmap_validation_report.json",
        "training_validation_report.json": Path(config.job_root) / "training_validation_report.json",
        "transforms.json": Path(config.ns_data_dir) / "transforms.json",
        "nerfstudio_config.yml": Path(training_report["latest_config"]),
    }

    copied = {}
    for name, src in files_to_copy.items():
        src = Path(src)
        if src.exists():
            dst = result_dir / name
            shutil.copy2(src, dst)
            copied[name] = str(dst)

    config_snapshot = result_dir / "config_snapshot.json"
    config_snapshot.write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8")
    copied["config_snapshot.json"] = str(config_snapshot)

    failure_summary = collect_failure_logs(config)
    copied["failure_log_summary.json"] = str(failure_summary)

    zip_path = result_dir / f"{config.job_id}_main_splatfacto_result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, path in copied.items():
            zf.write(path, name)
        for log_path in sorted(Path(config.logs_dir).glob("*.log")):
            zf.write(log_path, f"logs/{log_path.name}")

    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "job_id": config.job_id,
        "result_dir": str(result_dir),
        "zip_path": str(zip_path),
        "gaussian_ply": str(gaussian_ply_path),
        "copied_files": copied,
    }
    manifest_path = result_dir / "result_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    try:
        from google.colab import files
        files.download(str(zip_path))
    except Exception as exc:
        print(f"Automatic download skipped or failed: {exc!r}")
        print(f"Download manually from: {zip_path}")
    return zip_path


result_zip_path = package_results(config, gaussian_ply_path, training_report)
print(f"Done. Main pipeline result package: {result_zip_path}")
"""
    ),
]


def minimal_notebook(title: str) -> dict:
    return {
        "cells": [md(f"# {title}\n\nReserved for experiment work. The main execution path is `01_main_splatfacto_pipeline.ipynb`.")],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main():
    for directory in [
        PROJECT_ROOT / "notebooks",
        PROJECT_ROOT / "input" / "videos",
        PROJECT_ROOT / "data" / "nerfstudio",
        PROJECT_ROOT / "outputs" / "splatfacto",
        PROJECT_ROOT / "exports" / "gaussian_ply",
        PROJECT_ROOT / "exports" / "mesh",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")

    placeholders = {
        "02_experiment_da3_gsplat.ipynb": "Experiment DA3 gsplat",
        "03_experiment_vggt_pose.ipynb": "Experiment VGGT Pose",
        "04_compare_results.ipynb": "Compare Results",
    }
    for filename, title in placeholders.items():
        path = PROJECT_ROOT / "notebooks" / filename
        if not path.exists():
            path.write_text(json.dumps(minimal_notebook(title), indent=2, ensure_ascii=False), encoding="utf-8")

    readme = PROJECT_ROOT / "README.md"
    if not readme.exists():
        readme.write_text(
            """# Capstone 3DGS Project

Main execution notebook:

- `notebooks/01_main_splatfacto_pipeline.ipynb`

This main notebook runs the stable Nerfstudio path only:

1. RGB smartphone video input
2. `ns-process-data video` with COLMAP
3. `ns-train splatfacto`
4. `ns-export gaussian-splat`
5. Drive result packaging

Experiment notebooks are kept separate from the main pipeline.
""",
            encoding="utf-8",
        )

    for keep_dir in [
        PROJECT_ROOT / "input" / "videos",
        PROJECT_ROOT / "data" / "nerfstudio",
        PROJECT_ROOT / "outputs" / "splatfacto",
        PROJECT_ROOT / "exports" / "gaussian_ply",
        PROJECT_ROOT / "exports" / "mesh",
    ]:
        keep = keep_dir / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")

    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
