#!/usr/bin/env python3
"""
Multi-model processing examples for MLX Video Processor.

This script demonstrates how to use the ModelOrchestrator
to run multiple models on the same video.
"""

import numpy as np
from pathlib import Path

from mlx_video_processor.pipelines.orchestrator import (
    ModelOrchestrator,
    create_default_orchestrator,
    ParallelOrchestrator,
)
from mlx_video_processor.models import (
    FastViTProcessor,
    YOLOv3Processor,
    DeepLabV3Processor,
    DepthAnythingV2Processor,
)
from mlx_video_processor.types import ProcessingConfig, PrecisionType


def example_basic_orchestration():
    """
    Example: Basic Multi-Model Processing.

    Run multiple models on the same video frames with
    combined visualization.
    """
    print("\n" + "=" * 50)
    print("Example: Basic Multi-Model Processing")
    print("=" * 50)

    config = ProcessingConfig(
        frame_interval=5,
        max_frames=50,
        precision=PrecisionType.FLOAT16,
    )

    # Create orchestrator
    orchestrator = ModelOrchestrator(config)

    # Add models by name (from registry)
    orchestrator.add_model("yolo")       # Object detection
    orchestrator.add_model("deeplab")    # Semantic segmentation
    orchestrator.add_model("depth")      # Depth estimation

    print(f"Loaded models: {orchestrator.list_models()}")

    # Process video (uncomment with real video)
    # results = orchestrator.process_video(
    #     "input.mp4",
    #     output_path="multi_model_output.mp4",
    # )
    # print(orchestrator.get_performance_summary())

    # Process single frame for demonstration
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = orchestrator.process_single_frame(dummy_frame)

    print(f"\nCombined results for single frame:")
    print(f"  Detections: {result.detection.num_detections if result.detection else 0}")
    print(f"  Segmentation classes: {len(result.segmentation.class_ids) if result.segmentation else 0}")
    print(f"  Depth range: {result.depth.min_depth:.2f}-{result.depth.max_depth:.2f}" if result.depth else "N/A")
    print(f"  Total processing time: {result.processing_time_ms:.1f}ms")


def example_default_orchestrator():
    """
    Example: Using Default Orchestrator Factory.

    Quickly create an orchestrator with common models.
    """
    print("\n" + "=" * 50)
    print("Example: Default Orchestrator Factory")
    print("=" * 50)

    # Create with default models (fastvit, yolo, deeplab, depth)
    orchestrator = create_default_orchestrator()

    print(f"Default models: {orchestrator.list_models()}")

    # Or specify which models to include
    orchestrator = create_default_orchestrator(
        models=["yolo", "depth"],
        config=ProcessingConfig(frame_interval=3),
    )

    print(f"Custom selection: {orchestrator.list_models()}")


def example_chained_model_setup():
    """
    Example: Fluent Model Configuration.

    Use method chaining for clean model setup.
    """
    print("\n" + "=" * 50)
    print("Example: Fluent Model Configuration")
    print("=" * 50)

    config = ProcessingConfig(
        precision=PrecisionType.FLOAT16,
        confidence_threshold=0.6,
    )

    # Fluent configuration
    orchestrator = (
        ModelOrchestrator(config)
        .add_model("fastvit")
        .add_model("yolo", confidence_threshold=0.5)
        .add_model("depth")
    )

    print(f"Configured models: {orchestrator.list_models()}")

    # Get model information
    info = orchestrator.get_model_info()
    for name, model_info in info.items():
        print(f"\n{name}:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")


def example_custom_model_instances():
    """
    Example: Using Custom Model Instances.

    Create and configure model instances before adding
    them to the orchestrator.
    """
    print("\n" + "=" * 50)
    print("Example: Custom Model Instances")
    print("=" * 50)

    # Create custom-configured models
    yolo = YOLOv3Processor(
        precision=PrecisionType.FLOAT16,
        confidence_threshold=0.3,  # Lower threshold for more detections
        nms_threshold=0.4,
    )

    depth = DepthAnythingV2Processor(
        precision=PrecisionType.FLOAT16,
        normalize_output=True,
    )

    # Add custom instances
    orchestrator = ModelOrchestrator()
    orchestrator.add_model("detection", model=yolo)
    orchestrator.add_model("depth_map", model=depth)

    print(f"Custom models: {orchestrator.list_models()}")


def example_selective_processing():
    """
    Example: Selective Model Processing.

    Add and remove models dynamically based on needs.
    """
    print("\n" + "=" * 50)
    print("Example: Selective Model Processing")
    print("=" * 50)

    orchestrator = ModelOrchestrator()

    # Start with all models
    orchestrator.add_models(["fastvit", "yolo", "deeplab", "depth"])
    print(f"All models: {orchestrator.list_models()}")

    # Remove some for faster processing
    orchestrator.remove_model("deeplab")
    orchestrator.remove_model("fastvit")
    print(f"After removal: {orchestrator.list_models()}")


def example_grid_visualization():
    """
    Example: Grid Layout Visualization.

    Display results in a 2x2 grid showing different outputs.
    """
    print("\n" + "=" * 50)
    print("Example: Grid Layout Visualization")
    print("=" * 50)

    orchestrator = ModelOrchestrator()
    orchestrator.add_models(["yolo", "deeplab", "depth"])

    # Set grid layout
    orchestrator._visualizer.layout = "grid"

    # Process frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = orchestrator.process_single_frame(dummy_frame)

    # Get visualization
    vis_frame = orchestrator.visualize_frame(dummy_frame, result, layout="grid")

    print(f"Grid visualization shape: {vis_frame.shape}")
    print("Grid panels: Detection, Segmentation, Depth, Original")


def example_parallel_processing():
    """
    Example: Parallel Model Execution.

    Use ParallelOrchestrator for overlapped computation.
    """
    print("\n" + "=" * 50)
    print("Example: Parallel Model Execution")
    print("=" * 50)

    config = ProcessingConfig(
        frame_interval=2,
        max_frames=30,
    )

    # Create parallel orchestrator
    parallel_orchestrator = ParallelOrchestrator(
        config=config,
        num_streams=2,
    )

    parallel_orchestrator.add_models(["yolo", "depth"])

    print("Parallel orchestrator configured")
    print(f"Models: {parallel_orchestrator.list_models()}")
    print("Note: Full parallelism requires proper MLX stream management")


def example_performance_comparison():
    """
    Example: Compare Single vs Multi-Model Performance.

    Benchmark different configurations.
    """
    print("\n" + "=" * 50)
    print("Example: Performance Comparison")
    print("=" * 50)

    import time

    # Single model
    single = ModelOrchestrator()
    single.add_model("yolo")

    # Multi model
    multi = ModelOrchestrator()
    multi.add_models(["yolo", "depth"])

    # Benchmark with dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Warm up
    single.process_single_frame(dummy_frame)
    multi.process_single_frame(dummy_frame)

    # Time single model
    start = time.perf_counter()
    for _ in range(10):
        single.process_single_frame(dummy_frame)
    single_time = (time.perf_counter() - start) / 10 * 1000

    # Time multi model
    start = time.perf_counter()
    for _ in range(10):
        multi.process_single_frame(dummy_frame)
    multi_time = (time.perf_counter() - start) / 10 * 1000

    print(f"Single model (yolo): {single_time:.1f}ms per frame")
    print(f"Multi model (yolo+depth): {multi_time:.1f}ms per frame")
    print(f"Overhead: {multi_time - single_time:.1f}ms ({(multi_time/single_time - 1)*100:.1f}%)")


def example_result_aggregation():
    """
    Example: Aggregating Results Across Frames.

    Collect and analyze results over multiple frames.
    """
    print("\n" + "=" * 50)
    print("Example: Result Aggregation")
    print("=" * 50)

    orchestrator = ModelOrchestrator()
    orchestrator.add_model("yolo")
    orchestrator.add_model("fastvit")

    # Simulate processing multiple frames
    all_detections = []
    all_classifications = []

    for i in range(5):
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = orchestrator.process_single_frame(dummy_frame)

        if result.detection:
            all_detections.extend([
                (i, box.class_name, box.confidence)
                for box in result.detection.boxes
            ])

        if result.classification:
            all_classifications.append(
                (i, result.classification.class_name, result.classification.confidence)
            )

    print(f"Total detections across {5} frames: {len(all_detections)}")
    print(f"Total classifications: {len(all_classifications)}")

    # Analyze detection classes
    if all_detections:
        from collections import Counter
        class_counts = Counter(d[1] for d in all_detections)
        print("\nMost common detected classes:")
        for cls, count in class_counts.most_common(5):
            print(f"  {cls}: {count}")


if __name__ == "__main__":
    print("MLX Video Processor - Multi-Model Examples")
    print("==========================================")
    print("\nThese examples demonstrate multi-model processing.")
    print("Replace dummy frames with real video for full functionality.\n")

    example_basic_orchestration()
    example_default_orchestrator()
    example_chained_model_setup()
    example_custom_model_instances()
    example_selective_processing()
    example_grid_visualization()
    example_parallel_processing()
    example_performance_comparison()
    example_result_aggregation()

    print("\n" + "=" * 50)
    print("All multi-model examples completed!")
    print("=" * 50)
