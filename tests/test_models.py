"""
Tests for model implementations.
"""

import pytest
import numpy as np

# Skip all tests if MLX is not available
pytest.importorskip("mlx")

import mlx.core as mx

from mlx_video_processor.models import (
    FastViTProcessor,
    YOLOv3Processor,
    DeepLabV3Processor,
    DepthAnythingV2Processor,
    DETRProcessor,
    ModelRegistry,
)
from mlx_video_processor.types import PrecisionType


class TestModelRegistry:
    """Tests for model registry."""

    def test_list_models(self):
        """Test listing available models."""
        models = ModelRegistry.list_models()
        assert len(models) > 0
        assert "fastvit" in models
        assert "yolo" in models

    def test_get_model_class(self):
        """Test getting model class by name."""
        model_cls = ModelRegistry.get("fastvit")
        assert model_cls == FastViTProcessor

    def test_create_model(self):
        """Test creating model instance."""
        model = ModelRegistry.create("fastvit")
        assert isinstance(model, FastViTProcessor)

    def test_case_insensitive(self):
        """Test case-insensitive model lookup."""
        model1 = ModelRegistry.get("FASTVIT")
        model2 = ModelRegistry.get("FastVit")
        model3 = ModelRegistry.get("fastvit")
        assert model1 == model2 == model3

    def test_invalid_model(self):
        """Test error on invalid model name."""
        with pytest.raises(KeyError):
            ModelRegistry.get("nonexistent_model")


class TestFastViTProcessor:
    """Tests for FastViT classifier."""

    @pytest.fixture
    def model(self):
        return FastViTProcessor(precision=PrecisionType.FLOAT32, compile_model=False)

    @pytest.fixture
    def dummy_frame(self):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_initialization(self, model):
        """Test model initialization."""
        assert model.model_name == "FastViT-T8"
        assert model.default_input_size == (256, 256)
        assert model.supports_batching is True

    def test_preprocess(self, model, dummy_frame):
        """Test preprocessing."""
        model.load_model()
        x = model.preprocess(dummy_frame)
        assert x.shape == (1, 256, 256, 3)
        assert x.dtype == mx.float32

    def test_process_frame(self, model, dummy_frame):
        """Test full frame processing."""
        result = model.process_frame(dummy_frame, 0, 0.0)
        assert result.classification is not None
        assert result.classification.class_id >= 0
        assert 0 <= result.classification.confidence <= 1
        assert len(result.classification.top_k) == 5

    def test_model_info(self, model):
        """Test model info retrieval."""
        info = model.get_info()
        assert info["name"] == "FastViT-T8"
        assert "precision" in info
        assert "input_size" in info


class TestYOLOv3Processor:
    """Tests for YOLOv3 detector."""

    @pytest.fixture
    def model(self):
        return YOLOv3Processor(precision=PrecisionType.FLOAT32, compile_model=False)

    @pytest.fixture
    def dummy_frame(self):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_initialization(self, model):
        """Test model initialization."""
        assert model.model_name == "YOLOv3"
        assert model.default_input_size == (416, 416)
        assert model.num_classes == 80

    def test_preprocess(self, model, dummy_frame):
        """Test preprocessing with letterboxing."""
        model.load_model()
        x = model.preprocess(dummy_frame)
        assert x.shape == (1, 3, 416, 416)

    def test_process_frame(self, model, dummy_frame):
        """Test full frame processing."""
        result = model.process_frame(dummy_frame, 0, 0.0)
        assert result.detection is not None
        assert result.detection.num_detections >= 0
        assert isinstance(result.detection.boxes, list)


class TestDeepLabV3Processor:
    """Tests for DeepLabV3 segmentation."""

    @pytest.fixture
    def model(self):
        return DeepLabV3Processor(precision=PrecisionType.FLOAT32, compile_model=False)

    @pytest.fixture
    def dummy_frame(self):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_initialization(self, model):
        """Test model initialization."""
        assert model.model_name == "DeepLabV3"
        assert model.num_classes == 21

    def test_process_frame(self, model, dummy_frame):
        """Test full frame processing."""
        result = model.process_frame(dummy_frame, 0, 0.0)
        assert result.segmentation is not None
        assert result.segmentation.mask.shape == (480, 640)
        assert len(result.segmentation.class_ids) > 0


class TestDepthAnythingV2Processor:
    """Tests for Depth Anything V2."""

    @pytest.fixture
    def model(self):
        return DepthAnythingV2Processor(precision=PrecisionType.FLOAT32, compile_model=False)

    @pytest.fixture
    def dummy_frame(self):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_initialization(self, model):
        """Test model initialization."""
        assert model.model_name == "Depth-Anything-V2-Small"
        assert model.default_input_size == (518, 518)

    def test_process_frame(self, model, dummy_frame):
        """Test full frame processing."""
        result = model.process_frame(dummy_frame, 0, 0.0)
        assert result.depth is not None
        assert result.depth.depth_map.shape == (480, 640)
        assert result.depth.min_depth <= result.depth.max_depth


class TestDETRProcessor:
    """Tests for DETR detector."""

    @pytest.fixture
    def model(self):
        return DETRProcessor(precision=PrecisionType.FLOAT32, compile_model=False)

    @pytest.fixture
    def dummy_frame(self):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_initialization(self, model):
        """Test model initialization."""
        assert model.model_name == "DETR-ResNet50"
        assert model.num_queries == 100

    def test_process_frame(self, model, dummy_frame):
        """Test full frame processing."""
        result = model.process_frame(dummy_frame, 0, 0.0)
        assert result.detection is not None
        assert result.detection.num_detections >= 0
