"""
FastAPI backend for MLX Video Processor Web Interface.

Provides REST API and WebSocket endpoints for video processing
with real-time progress updates.
"""

import asyncio
import json
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# MLX Video Processor imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_video_processor.models.base import ModelRegistry
from mlx_video_processor.pipelines.orchestrator import ModelOrchestrator
from mlx_video_processor.pipelines.video_processor import VideoProcessor
from mlx_video_processor.visualizers import (
    DetectionVisualizer,
    SegmentationVisualizer,
    DepthVisualizer,
    ClassificationVisualizer,
    CompositeVisualizer,
)
from mlx_video_processor.types import ProcessingConfig, PrecisionType
from mlx_video_processor.utils.video_io import VideoReader


# Configuration
UPLOAD_DIR = Path(__file__).parent / "uploads"
OUTPUT_DIR = Path(__file__).parent / "outputs"
STATIC_DIR = Path(__file__).parent / "static"
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Job tracking
jobs: dict[str, dict] = {}
active_websockets: dict[str, WebSocket] = {}


# Pydantic models
class ProcessingRequest(BaseModel):
    video_id: str
    models: list[str]
    frame_interval: int = 1
    confidence_threshold: float = 0.5
    max_frames: Optional[int] = None
    target_objects: Optional[str] = None  # Comma-separated list of objects to detect


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, error
    progress: float
    current_frame: int
    total_frames: int
    current_model: Optional[str]
    fps: float
    elapsed_time: float
    output_video: Optional[str]
    error: Optional[str]
    detection_results: Optional[dict] = None  # Summary of detected objects


class ModelInfo(BaseModel):
    name: str
    display_name: str
    description: str
    type: str
    icon: str


# Lifespan handler for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Clean old files
    cleanup_old_files()
    yield
    # Shutdown: nothing special needed


# FastAPI app
app = FastAPI(
    title="MLX Video Processor",
    description="Process videos with Apple ML models on Apple Silicon",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


# Model information
MODEL_INFO = {
    "fastvit": ModelInfo(
        name="fastvit",
        display_name="FastViT T8",
        description="Fast hybrid vision transformer for image classification (1000 ImageNet classes)",
        type="Classification",
        icon="tag",
    ),
    "yolo": ModelInfo(
        name="yolo",
        display_name="YOLOv3",
        description="Real-time multi-object detection with 80 COCO classes",
        type="Detection",
        icon="box",
    ),
    "deeplab": ModelInfo(
        name="deeplab",
        display_name="DeepLabV3",
        description="Pixel-level semantic segmentation with 21 VOC classes",
        type="Segmentation",
        icon="layers",
    ),
    "depth": ModelInfo(
        name="depth",
        display_name="Depth Anything V2",
        description="Monocular depth estimation from single images",
        type="Depth",
        icon="box",
    ),
    "detr": ModelInfo(
        name="detr",
        display_name="DETR ResNet50",
        description="Transformer-based object detection without NMS",
        type="Detection",
        icon="scan",
    ),
}


def cleanup_old_files(max_age_hours: int = 24):
    """Remove files older than max_age_hours."""
    cutoff = time.time() - (max_age_hours * 3600)

    for directory in [UPLOAD_DIR, OUTPUT_DIR]:
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff:
                try:
                    file_path.unlink()
                except Exception:
                    pass


def get_visualizer_for_models(model_names: list[str]):
    """Get appropriate visualizer based on selected models."""
    if len(model_names) > 1:
        return CompositeVisualizer(layout="overlay", show_fps=True)

    model = model_names[0].lower()
    if model in ["fastvit", "classification"]:
        return ClassificationVisualizer()
    elif model in ["yolo", "yolov3", "detr"]:
        return DetectionVisualizer()
    elif model in ["deeplab", "deeplabv3"]:
        return SegmentationVisualizer()
    elif model in ["depth", "depth_anything"]:
        return DepthVisualizer(colormap="turbo")
    return CompositeVisualizer()


async def process_video_task(
    job_id: str,
    video_path: Path,
    output_path: Path,
    models: list[str],
    config: ProcessingConfig,
):
    """Background task for video processing with progress updates."""
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["start_time"] = time.time()

        # Get video info
        with VideoReader(str(video_path)) as reader:
            total_frames = reader.metadata.total_frames
            if config.frame_interval > 1:
                total_frames = total_frames // config.frame_interval
            if config.max_frames:
                total_frames = min(total_frames, config.max_frames)

        jobs[job_id]["total_frames"] = total_frames

        # Create orchestrator or single processor
        if len(models) > 1:
            processor = ModelOrchestrator(config)
            for model_name in models:
                processor.add_model(model_name)
            visualizer = CompositeVisualizer(layout="overlay", show_fps=True)
            processor._visualizer = visualizer
        else:
            model = ModelRegistry.create(models[0], precision=config.precision)
            visualizer = get_visualizer_for_models(models)
            processor = VideoProcessor(model, visualizer, config)

        # Track progress
        frame_count = 0
        last_update = time.time()

        def progress_callback(result, vis_frame):
            nonlocal frame_count, last_update
            frame_count += 1

            current_time = time.time()
            elapsed = current_time - jobs[job_id]["start_time"]
            fps = frame_count / elapsed if elapsed > 0 else 0

            jobs[job_id]["current_frame"] = frame_count
            jobs[job_id]["progress"] = (frame_count / total_frames) * 100
            jobs[job_id]["fps"] = fps
            jobs[job_id]["elapsed_time"] = elapsed
            jobs[job_id]["current_model"] = models[-1] if len(models) == 1 else "multi"

            # Send WebSocket update (throttled to avoid flooding)
            if current_time - last_update >= 0.1:  # Max 10 updates/sec
                last_update = current_time
                asyncio.create_task(send_progress_update(job_id))

        # Process video
        if isinstance(processor, ModelOrchestrator):
            processor.process_video(
                str(video_path),
                output_path=str(output_path),
                progress=False,
                callback=progress_callback,
            )
        else:
            processor.process_video(
                str(video_path),
                output_path=str(output_path),
                progress=False,
                callback=progress_callback,
            )

        # Mark complete
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["output_video"] = output_path.name

        await send_progress_update(job_id)

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        await send_progress_update(job_id)
        raise


async def send_progress_update(job_id: str):
    """Send progress update via WebSocket."""
    if job_id in active_websockets:
        ws = active_websockets[job_id]
        try:
            await ws.send_json(jobs[job_id])
        except Exception:
            pass


# Routes

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h1>MLX Video Processor</h1><p>Static files not found.</p>")


@app.get("/api/models")
async def list_models():
    """List available models with descriptions."""
    return {"models": list(MODEL_INFO.values())}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file."""
    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Generate unique ID
    video_id = str(uuid.uuid4())[:8]
    filename = f"{video_id}{ext}"
    file_path = UPLOAD_DIR / filename

    # Save file
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="File too large (max 500MB)")
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Get video info
    try:
        with VideoReader(str(file_path)) as reader:
            metadata = reader.metadata
            info = {
                "video_id": video_id,
                "filename": file.filename,
                "width": metadata.width,
                "height": metadata.height,
                "fps": metadata.fps,
                "total_frames": metadata.total_frames,
                "duration": metadata.duration_seconds,
            }
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid video file: {e}")

    return info


@app.post("/api/process")
async def start_processing(request: ProcessingRequest, background_tasks: BackgroundTasks):
    """Start video processing job."""
    # Validate video exists
    video_files = list(UPLOAD_DIR.glob(f"{request.video_id}.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = video_files[0]

    # Validate models
    for model in request.models:
        if model.lower() not in MODEL_INFO:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    # Create job
    job_id = str(uuid.uuid4())[:8]
    output_filename = f"{job_id}_processed.mp4"
    output_path = OUTPUT_DIR / output_filename

    # Processing config
    config = ProcessingConfig(
        frame_interval=request.frame_interval,
        confidence_threshold=request.confidence_threshold,
        max_frames=request.max_frames,
        precision=PrecisionType.FLOAT16,
        use_compile=True,
    )

    # Initialize job
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "current_frame": 0,
        "total_frames": 0,
        "current_model": None,
        "fps": 0,
        "elapsed_time": 0,
        "output_video": None,
        "error": None,
        "models": request.models,
        "video_id": request.video_id,
        "start_time": None,
    }

    # Start background processing
    background_tasks.add_task(
        process_video_task,
        job_id,
        video_path,
        output_path,
        request.models,
        config,
    )

    return {"job_id": job_id, "status": "pending"}


@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    """Get job status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket for real-time progress updates."""
    await websocket.accept()
    active_websockets[job_id] = websocket

    try:
        # Send initial status
        if job_id in jobs:
            await websocket.send_json(jobs[job_id])

        # Keep connection alive
        while True:
            try:
                # Wait for messages (like ping)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_text("keepalive")
            except WebSocketDisconnect:
                break

    except Exception:
        pass
    finally:
        if job_id in active_websockets:
            del active_websockets[job_id]


@app.get("/api/video/{video_id}")
async def get_video(video_id: str):
    """Serve uploaded video."""
    video_files = list(UPLOAD_DIR.glob(f"{video_id}.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_files[0], media_type="video/mp4")


@app.get("/api/output/{filename}")
async def get_output(filename: str):
    """Serve processed video."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output not found")
    return FileResponse(file_path, media_type="video/mp4")


@app.delete("/api/video/{video_id}")
async def delete_video(video_id: str):
    """Delete uploaded video."""
    video_files = list(UPLOAD_DIR.glob(f"{video_id}.*"))
    for f in video_files:
        f.unlink(missing_ok=True)
    return {"status": "deleted"}


@app.get("/api/gallery")
async def get_gallery():
    """Get list of recent processed videos."""
    outputs = []
    for file_path in sorted(OUTPUT_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
        if file_path.suffix == ".mp4":
            job_id = file_path.stem.replace("_processed", "")
            outputs.append({
                "filename": file_path.name,
                "job_id": job_id,
                "created": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                "size_mb": file_path.stat().st_size / (1024 * 1024),
            })
    return {"videos": outputs}


@app.post("/api/cleanup")
async def cleanup_files():
    """Manually trigger file cleanup."""
    cleanup_old_files(max_age_hours=1)
    return {"status": "cleaned"}


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
