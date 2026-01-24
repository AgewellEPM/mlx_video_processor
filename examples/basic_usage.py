#!/usr/bin/env python3
"""
Basic usage examples for MLX Video Processor.

This script demonstrates how to use the video processor
with individual models for common computer vision tasks.
"""

import numpy as np
import cv2
from pathlib import Path

# Import MLX Video Processor components
from mlx_video_processor.models import (
    FastViTProcessor,
    YOLOv3Processor,
    DeepLabV3Processor,
    DepthAnythingV2Processor,
    DETRProcessor,
)
from mlx_video_processor.pipelines import VideoProcessor
from mlx_video_processor.visualizers import (
    ClassificationVisualizer,
    DetectionVisualizer,
    SegmentationVisualizer,
    DepthVisualizer,
)
from mlx_video_processor.types import ProcessingConfig, PrecisionType


def example_image_classification():
    """
    Example: Image Classification with FastViT.

    FastViT is a fast hybrid vision transformer that classifies
    images into 1000 ImageNet categories.
    """
    print("\n" + "=" * 50)
    print("Example: Image Classification with FastViT")
    print("=" * 50)

    # Create the model
    model = FastViTProcessor(
        precision=PrecisionType.FLOAT16,
        compile_model=True,
        top_k=5,  # Return top 5 predictions
    )

    # Create visualizer
    visualizer = ClassificationVisualizer(
        show_top_k=5,
        bar_width=200,
        position="top_left",
    )

    # Configure processing
    config = ProcessingConfig(
        frame_interval=15,  # Process every 15th frame
        max_frames=100,     # Process up to 100 frames
    )

    # Create processor
    processor = VideoProcessor(model, visualizer, config)

    # Process video (replace with your video path)
    # results = processor.process_video(
    #     "input.mp4",
    #     output_path="classification_output.mp4",
    # )

    # Or process a single image
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = processor.process_single_frame(dummy_frame)

    print(f"Top prediction: {result.classification.class_name}")
    print(f"Confidence: {result.classification.confidence:.2%}")
    print("Top 5 predictions:")
    for class_id, class_name, conf in result.classification.top_k:
        print(f"  {class_name}: {conf:.2%}")


def example_object_detection():
    """
    Example: Object Detection with YOLOv3.

    YOLOv3 detects 80 different object classes in real-time
    with bounding boxes and confidence scores.
    """
    print("\n" + "=" * 50)
    print("Example: Object Detection with YOLOv3")
    print("=" * 50)

    # Create the model
    model = YOLOv3Processor(
        precision=PrecisionType.FLOAT16,
        compile_model=True,
        confidence_threshold=0.5,
        nms_threshold=0.45,
    )

    # Create visualizer with custom colors
    visualizer = DetectionVisualizer(
        show_labels=True,
        show_confidence=True,
        line_thickness=2,
        class_colors={0: (0, 255, 0)},  # Green for person
    )

    config = ProcessingConfig(
        frame_interval=1,
        confidence_threshold=0.5,
    )

    processor = VideoProcessor(model, visualizer, config)

    # Process video
    # results = processor.process_video(
    #     "input.mp4",
    #     output_path="detection_output.mp4",
    # )

    # Example with dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = processor.process_single_frame(dummy_frame)

    print(f"Detected {result.detection.num_detections} objects")
    for box in result.detection.boxes[:5]:
        print(f"  {box.class_name}: {box.confidence:.2%} at ({box.x1:.2f}, {box.y1:.2f})")


def example_semantic_segmentation():
    """
    Example: Semantic Segmentation with DeepLabV3.

    DeepLabV3 assigns class labels to each pixel in the image,
    enabling understanding of scene composition.
    """
    print("\n" + "=" * 50)
    print("Example: Semantic Segmentation with DeepLabV3")
    print("=" * 50)

    model = DeepLabV3Processor(
        precision=PrecisionType.FLOAT16,
        compile_model=True,
        output_stride=16,
    )

    visualizer = SegmentationVisualizer(
        opacity=0.5,
        show_legend=True,
        show_boundaries=True,
    )

    config = ProcessingConfig(
        frame_interval=5,
    )

    processor = VideoProcessor(model, visualizer, config)

    # Example processing
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = processor.process_single_frame(dummy_frame)

    print(f"Found {result.segmentation.num_classes} classes in image")
    print("Classes present:")
    for class_name in result.segmentation.class_names:
        print(f"  - {class_name}")


def example_depth_estimation():
    """
    Example: Depth Estimation with Depth Anything V2.

    Depth Anything V2 estimates relative depth from single images,
    useful for 3D understanding and video effects.
    """
    print("\n" + "=" * 50)
    print("Example: Depth Estimation with Depth Anything V2")
    print("=" * 50)

    model = DepthAnythingV2Processor(
        precision=PrecisionType.FLOAT16,
        compile_model=True,
        normalize_output=True,
    )

    visualizer = DepthVisualizer(
        colormap="magma",
        invert=False,
        show_colorbar=True,
        show_stats=True,
    )

    config = ProcessingConfig(frame_interval=1)
    processor = VideoProcessor(model, visualizer, config)

    # Example processing
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = processor.process_single_frame(dummy_frame)

    print(f"Depth range: {result.depth.min_depth:.3f} - {result.depth.max_depth:.3f}")
    print(f"Depth map shape: {result.depth.depth_map.shape}")

    # Get colored depth visualization
    colored_depth = model.get_colored_depth(result.depth, colormap="turbo")
    print(f"Colored depth shape: {colored_depth.shape}")


def example_detr_detection():
    """
    Example: Object Detection with DETR.

    DETR uses a transformer architecture for end-to-end
    object detection without hand-crafted components like NMS.
    """
    print("\n" + "=" * 50)
    print("Example: Object Detection with DETR")
    print("=" * 50)

    model = DETRProcessor(
        precision=PrecisionType.FLOAT16,
        compile_model=True,
        confidence_threshold=0.7,
        num_queries=100,
    )

    visualizer = DetectionVisualizer(
        show_labels=True,
        show_confidence=True,
    )

    config = ProcessingConfig(frame_interval=1)
    processor = VideoProcessor(model, visualizer, config)

    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = processor.process_single_frame(dummy_frame)

    print(f"DETR detected {result.detection.num_detections} objects")


def example_streaming_process():
    """
    Example: Streaming Video Processing.

    Process video frame by frame with real-time access to results.
    """
    print("\n" + "=" * 50)
    print("Example: Streaming Video Processing")
    print("=" * 50)

    model = YOLOv3Processor(precision=PrecisionType.FLOAT16)
    visualizer = DetectionVisualizer()
    config = ProcessingConfig(frame_interval=5)

    processor = VideoProcessor(model, visualizer, config)

    # Stream process (uncomment with real video)
    # for frame, result in processor.stream_process("input.mp4"):
    #     # Process each frame as it's ready
    #     print(f"Frame {result.frame_index}: {result.detection.num_detections} objects")
    #
    #     # Display frame (requires display)
    #     cv2.imshow("Detection", frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    print("Streaming example configured (provide video path to run)")


def example_custom_callback():
    """
    Example: Processing with Custom Callback.

    Use callbacks to process results in real-time or save
    intermediate outputs.
    """
    print("\n" + "=" * 50)
    print("Example: Custom Callback Processing")
    print("=" * 50)

    model = FastViTProcessor(precision=PrecisionType.FLOAT16)
    visualizer = ClassificationVisualizer()
    config = ProcessingConfig(frame_interval=10)

    processor = VideoProcessor(model, visualizer, config)

    # Track classifications
    classifications = []

    def my_callback(result, vis_frame):
        """Custom callback for each processed frame."""
        if result.classification:
            classifications.append({
                'frame': result.frame_index,
                'class': result.classification.class_name,
                'confidence': result.classification.confidence,
            })
            print(f"Frame {result.frame_index}: {result.classification.class_name}")

    # Process with callback (uncomment with real video)
    # processor.process_video("input.mp4", callback=my_callback)
    # print(f"\nTotal frames classified: {len(classifications)}")

    print("Callback example configured (provide video path to run)")


if __name__ == "__main__":
    print("MLX Video Processor - Basic Usage Examples")
    print("==========================================")
    print("\nNote: These examples use dummy frames for demonstration.")
    print("Replace with actual video paths for real processing.\n")

    # Run examples
    example_image_classification()
    example_object_detection()
    example_semantic_segmentation()
    example_depth_estimation()
    example_detr_detection()
    example_streaming_process()
    example_custom_callback()

    print("\n" + "=" * 50)
    print("All examples completed!")
    print("=" * 50)
