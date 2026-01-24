#!/usr/bin/env python3
"""
Command-line interface for MLX Video Processor.

Provides a CLI tool for processing videos with various Apple ML models.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from mlx_video_processor import __version__
from mlx_video_processor.models.base import ModelRegistry
from mlx_video_processor.types import ProcessingConfig, PrecisionType
from mlx_video_processor.pipelines.video_processor import VideoProcessor, BatchVideoProcessor
from mlx_video_processor.pipelines.orchestrator import ModelOrchestrator, create_default_orchestrator
from mlx_video_processor.visualizers import (
    DetectionVisualizer,
    SegmentationVisualizer,
    DepthVisualizer,
    ClassificationVisualizer,
    CompositeVisualizer,
)


def get_visualizer_for_model(model_name: str):
    """Get appropriate visualizer for a model type."""
    model_name = model_name.lower()

    if model_name in ["fastvit", "classification"]:
        return ClassificationVisualizer()
    elif model_name in ["yolo", "yolov3", "detr", "detection"]:
        return DetectionVisualizer()
    elif model_name in ["deeplab", "deeplabv3", "segmentation"]:
        return SegmentationVisualizer()
    elif model_name in ["depth", "depth_anything", "depth_anything_v2"]:
        return DepthVisualizer()
    else:
        return CompositeVisualizer()


def process_single_model(args):
    """Process video with a single model."""
    # Create config
    config = ProcessingConfig(
        frame_interval=args.frame_interval,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        max_frames=args.max_frames,
        confidence_threshold=args.confidence,
        precision=PrecisionType(args.precision),
        use_compile=not args.no_compile,
    )

    # Create model
    model = ModelRegistry.create(
        args.model,
        precision=config.precision,
        compile_model=config.use_compile,
    )

    # Create visualizer
    if args.no_visualize:
        visualizer = None
    else:
        visualizer = get_visualizer_for_model(args.model)

    # Create processor
    if args.batch_size > 1 and model.supports_batching:
        processor = BatchVideoProcessor(model, visualizer, config, batch_size=args.batch_size)
    else:
        processor = VideoProcessor(model, visualizer, config)

    # Process
    print(f"\nProcessing: {args.input}")
    print(f"Model: {model.model_name}")
    print(f"Precision: {args.precision}")
    if args.output:
        print(f"Output: {args.output}")
    print()

    results = processor.process_video(
        args.input,
        output_path=args.output,
        progress=not args.quiet,
    )

    # Print summary
    if not args.quiet:
        print("\n" + "=" * 50)
        print(processor.get_performance_summary())
        print("\nResults Summary:")
        summary = processor.get_results_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")


def process_multiple_models(args):
    """Process video with multiple models."""
    config = ProcessingConfig(
        frame_interval=args.frame_interval,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        max_frames=args.max_frames,
        confidence_threshold=args.confidence,
        precision=PrecisionType(args.precision),
        use_compile=not args.no_compile,
    )

    # Create orchestrator
    orchestrator = ModelOrchestrator(config)

    for model_name in args.models:
        orchestrator.add_model(model_name)

    print(f"\nProcessing: {args.input}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Precision: {args.precision}")
    if args.output:
        print(f"Output: {args.output}")
    print()

    results = orchestrator.process_video(
        args.input,
        output_path=args.output,
        progress=not args.quiet,
    )

    if not args.quiet:
        print("\n" + "=" * 50)
        print(orchestrator.get_performance_summary())


def list_models(args):
    """List available models."""
    print("\nAvailable Models:")
    print("=" * 50)

    models = ModelRegistry.list_models()
    for name in sorted(set(models)):
        try:
            model_cls = ModelRegistry.get(name)
            print(f"  {name:20s} - {model_cls.model_name}")
        except Exception as e:
            print(f"  {name:20s} - (error loading)")


def show_info(args):
    """Show model information."""
    model = ModelRegistry.create(args.model, compile_model=False)
    info = model.get_info()

    print(f"\nModel: {args.model}")
    print("=" * 50)
    for key, value in info.items():
        print(f"  {key}: {value}")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="mlx-video",
        description="Process videos with Apple ML models using MLX on Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video with YOLOv3 object detection
  mlx-video process -m yolo -i input.mp4 -o output.mp4

  # Process with FastViT classification
  mlx-video process -m fastvit -i input.mp4 -o output.mp4

  # Process with depth estimation
  mlx-video process -m depth -i input.mp4 -o depth_output.mp4

  # Process every 5th frame with high confidence threshold
  mlx-video process -m yolo -i input.mp4 -o output.mp4 --frame-interval 5 --confidence 0.7

  # Run multiple models
  mlx-video multi -m yolo deeplab depth -i input.mp4 -o multi_output.mp4

  # List available models
  mlx-video list

  # Show model information
  mlx-video info -m yolo
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command (single model)
    process_parser = subparsers.add_parser(
        "process",
        help="Process video with a single model",
    )
    process_parser.add_argument(
        "-m", "--model",
        required=True,
        help="Model to use (fastvit, yolo, deeplab, depth, detr)",
    )
    process_parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input video path",
    )
    process_parser.add_argument(
        "-o", "--output",
        help="Output video path (optional)",
    )
    process_parser.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1)",
    )
    process_parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Starting frame index (default: 0)",
    )
    process_parser.add_argument(
        "--end-frame",
        type=int,
        help="Ending frame index (default: end of video)",
    )
    process_parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum frames to process",
    )
    process_parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)",
    )
    process_parser.add_argument(
        "--precision",
        choices=["float32", "float16", "bfloat16"],
        default="float16",
        help="Model precision (default: float16)",
    )
    process_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for models that support batching",
    )
    process_parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable model compilation",
    )
    process_parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable visualization (faster processing)",
    )
    process_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    process_parser.set_defaults(func=process_single_model)

    # Multi-model command
    multi_parser = subparsers.add_parser(
        "multi",
        help="Process video with multiple models",
    )
    multi_parser.add_argument(
        "-m", "--models",
        nargs="+",
        required=True,
        help="Models to use",
    )
    multi_parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input video path",
    )
    multi_parser.add_argument(
        "-o", "--output",
        help="Output video path",
    )
    multi_parser.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="Process every Nth frame",
    )
    multi_parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Starting frame index",
    )
    multi_parser.add_argument(
        "--end-frame",
        type=int,
        help="Ending frame index",
    )
    multi_parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum frames to process",
    )
    multi_parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold",
    )
    multi_parser.add_argument(
        "--precision",
        choices=["float32", "float16", "bfloat16"],
        default="float16",
        help="Model precision",
    )
    multi_parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable model compilation",
    )
    multi_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    multi_parser.set_defaults(func=process_multiple_models)

    # List command
    list_parser = subparsers.add_parser(
        "list",
        help="List available models",
    )
    list_parser.set_defaults(func=list_models)

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show model information",
    )
    info_parser.add_argument(
        "-m", "--model",
        required=True,
        help="Model name",
    )
    info_parser.set_defaults(func=show_info)

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if "--debug" in sys.argv:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
