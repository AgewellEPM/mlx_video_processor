"""
Tests for video processing pipeline.
"""

import pytest
import numpy as np
import tempfile
import os

pytest.importorskip("mlx")
pytest.importorskip("cv2")

import cv2

from mlx_video_processor.pipelines import VideoProcessor, ModelOrchestrator
from mlx_video_processor.models import FastViTProcessor, YOLOv3Processor
from mlx_video_processor.visualizers import ClassificationVisualizer, DetectionVisualizer
from mlx_video_processor.types import ProcessingConfig, PrecisionType, FrameResult


class TestVideoProcessor:
    """Tests for VideoProcessor."""

    @pytest.fixture
    def model(self):
        return FastViTProcessor(precision=PrecisionType.FLOAT32, compile_model=False)

    @pytest.fixture
    def visualizer(self):
        return ClassificationVisualizer()

    @pytest.fixture
    def config(self):
        return ProcessingConfig(frame_interval=1, max_frames=5)

    @pytest.fixture
    def processor(self, model, visualizer, config):
        return VideoProcessor(model, visualizer, config)

    @pytest.fixture
    def dummy_frames(self):
        return [np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(5)]

    @pytest.fixture
    def temp_video(self, dummy_frames):
        """Create a temporary test video."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_path, fourcc, 30.0, (320, 240))

        for frame in dummy_frames:
            writer.write(frame)
        writer.release()

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_process_single_frame(self, processor, dummy_frames):
        """Test single frame processing."""
        result = processor.process_single_frame(dummy_frames[0])
        assert isinstance(result, FrameResult)
        assert result.classification is not None

    def test_process_frames_list(self, processor, dummy_frames):
        """Test processing list of frames."""
        results = processor.process_frames(dummy_frames[:3])
        assert len(results) == 3
        for result in results:
            assert result.classification is not None

    def test_results_summary(self, processor, dummy_frames):
        """Test results summary generation."""
        processor.process_frames(dummy_frames)
        summary = processor.get_results_summary()
        assert "total_frames" in summary
        assert summary["total_frames"] == len(dummy_frames)


class TestModelOrchestrator:
    """Tests for ModelOrchestrator."""

    @pytest.fixture
    def orchestrator(self):
        return ModelOrchestrator()

    @pytest.fixture
    def dummy_frame(self):
        return np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

    def test_add_model_by_name(self, orchestrator):
        """Test adding model by name."""
        orchestrator.add_model("fastvit")
        assert "fastvit" in orchestrator.list_models()

    def test_add_model_instance(self, orchestrator):
        """Test adding model instance."""
        model = YOLOv3Processor(compile_model=False)
        orchestrator.add_model("detection", model=model)
        assert "detection" in orchestrator.list_models()

    def test_add_multiple_models(self, orchestrator):
        """Test adding multiple models."""
        orchestrator.add_models(["fastvit", "yolo"])
        models = orchestrator.list_models()
        assert "fastvit" in models
        assert "yolo" in models

    def test_remove_model(self, orchestrator):
        """Test removing model."""
        orchestrator.add_model("fastvit")
        orchestrator.add_model("yolo")
        orchestrator.remove_model("fastvit")
        assert "fastvit" not in orchestrator.list_models()
        assert "yolo" in orchestrator.list_models()

    def test_method_chaining(self, orchestrator):
        """Test fluent interface."""
        result = orchestrator.add_model("fastvit").add_model("yolo")
        assert result is orchestrator
        assert len(orchestrator.list_models()) == 2

    def test_process_single_frame(self, orchestrator, dummy_frame):
        """Test processing with multiple models."""
        orchestrator.add_model("fastvit")
        result = orchestrator.process_single_frame(dummy_frame)
        assert result.classification is not None

    def test_no_models_error(self, orchestrator, dummy_frame):
        """Test error when no models added."""
        with pytest.raises(ValueError, match="No models added"):
            orchestrator.process_single_frame(dummy_frame)

    def test_get_model_info(self, orchestrator):
        """Test getting model information."""
        orchestrator.add_model("fastvit")
        info = orchestrator.get_model_info()
        assert "fastvit" in info
        assert "name" in info["fastvit"]


class TestProcessingConfig:
    """Tests for ProcessingConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ProcessingConfig()
        assert config.frame_interval == 1
        assert config.batch_size == 1
        assert config.precision == PrecisionType.FLOAT16

    def test_validation_success(self):
        """Test valid configuration."""
        config = ProcessingConfig(frame_interval=5, confidence_threshold=0.7)
        config.validate()  # Should not raise

    def test_validation_frame_interval(self):
        """Test invalid frame interval."""
        config = ProcessingConfig(frame_interval=0)
        with pytest.raises(ValueError, match="frame_interval"):
            config.validate()

    def test_validation_confidence(self):
        """Test invalid confidence threshold."""
        config = ProcessingConfig(confidence_threshold=1.5)
        with pytest.raises(ValueError, match="confidence_threshold"):
            config.validate()
