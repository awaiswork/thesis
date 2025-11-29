"""Base class for object detection model evaluators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import time
import platform

import torch


@dataclass
class TimingBreakdown:
    """Breakdown of inference timing components."""
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms

    def to_dict(self) -> Dict[str, float]:
        return {
            "preprocess_ms": self.preprocess_ms,
            "inference_ms": self.inference_ms,
            "postprocess_ms": self.postprocess_ms,
            "total_ms": self.total_ms,
        }


@dataclass
class APBySize:
    """Average Precision broken down by object size."""
    small: float = 0.0   # area < 32^2
    medium: float = 0.0  # 32^2 < area < 96^2
    large: float = 0.0   # area > 96^2

    def to_dict(self) -> Dict[str, float]:
        return {
            "small": self.small,
            "medium": self.medium,
            "large": self.large,
        }


@dataclass
class ModelInfo:
    """Model architecture information."""
    name: str = ""
    parameters: int = 0  # Number of parameters
    gflops: float = 0.0  # Computational cost
    input_size: int = 640  # Input image size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parameters": self.parameters,
            "gflops": self.gflops,
            "input_size": self.input_size,
        }


@dataclass
class HardwareInfo:
    """Hardware and environment information."""
    device: str = "cpu"
    device_name: str = ""
    pytorch_version: str = ""
    python_version: str = ""
    platform: str = ""

    @classmethod
    def collect(cls, device: torch.device) -> "HardwareInfo":
        """Collect current hardware information."""
        device_name = ""
        if device.type == "cuda":
            device_name = torch.cuda.get_device_name(0)
        elif device.type == "mps":
            device_name = "Apple Silicon GPU"

        return cls(
            device=str(device),
            device_name=device_name,
            pytorch_version=torch.__version__,
            python_version=platform.python_version(),
            platform=platform.platform(),
        )

    def to_dict(self) -> Dict[str, str]:
        return {
            "device": self.device,
            "device_name": self.device_name,
            "pytorch_version": self.pytorch_version,
            "python_version": self.python_version,
            "platform": self.platform,
        }


@dataclass
class EvaluationResults:
    """Container for comprehensive evaluation metrics."""

    # Model identification
    model_name: str

    # Primary metrics (percentages, 0-100)
    map50: float  # mAP@0.5
    map50_95: float  # mAP@[0.5:0.95]
    precision: Optional[float] = None
    recall: Optional[float] = None

    # Extended AP metrics
    map75: Optional[float] = None  # mAP@0.75
    ap_by_size: Optional[APBySize] = None

    # Derived metrics
    f1_score: Optional[float] = None

    # Timing metrics
    time_per_image_ms: float = 0.0
    fps: float = 0.0
    timing_breakdown: Optional[TimingBreakdown] = None

    # Evaluation info
    num_images: int = 0
    total_time_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Model and hardware info
    model_info: Optional[ModelInfo] = None
    hardware_info: Optional[HardwareInfo] = None

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        # Calculate F1 score if precision and recall are available
        if self.precision is not None and self.recall is not None:
            if self.precision + self.recall > 0:
                self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
            else:
                self.f1_score = 0.0

    def __str__(self) -> str:
        parts = [
            f"{self.model_name}:",
            f"mAP@0.5: {self.map50:.1f}%",
            f"mAP@[0.5:0.95]: {self.map50_95:.1f}%",
        ]
        if self.precision is not None:
            parts.append(f"Precision: {self.precision:.1f}%")
        if self.recall is not None:
            parts.append(f"Recall: {self.recall:.1f}%")
        if self.f1_score is not None:
            parts.append(f"F1: {self.f1_score:.1f}%")
        parts.extend([
            f"Time/image: {self.time_per_image_ms:.1f}ms",
            f"FPS: {self.fps:.1f}",
        ])
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "model_name": self.model_name,
            "map50": self.map50,
            "map50_95": self.map50_95,
            "map75": self.map75,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "time_per_image_ms": self.time_per_image_ms,
            "fps": self.fps,
            "num_images": self.num_images,
            "total_time_seconds": self.total_time_seconds,
            "timestamp": self.timestamp,
        }

        if self.ap_by_size is not None:
            result["ap_by_size"] = self.ap_by_size.to_dict()

        if self.timing_breakdown is not None:
            result["timing_breakdown"] = self.timing_breakdown.to_dict()

        if self.model_info is not None:
            result["model_info"] = self.model_info.to_dict()

        if self.hardware_info is not None:
            result["hardware_info"] = self.hardware_info.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResults":
        """Create from dictionary."""
        # Parse nested objects
        ap_by_size = None
        if "ap_by_size" in data and data["ap_by_size"]:
            ap_by_size = APBySize(**data["ap_by_size"])

        timing_breakdown = None
        if "timing_breakdown" in data and data["timing_breakdown"]:
            tb = data["timing_breakdown"]
            timing_breakdown = TimingBreakdown(
                preprocess_ms=tb.get("preprocess_ms", 0),
                inference_ms=tb.get("inference_ms", 0),
                postprocess_ms=tb.get("postprocess_ms", 0),
            )

        model_info = None
        if "model_info" in data and data["model_info"]:
            model_info = ModelInfo(**data["model_info"])

        hardware_info = None
        if "hardware_info" in data and data["hardware_info"]:
            hardware_info = HardwareInfo(**data["hardware_info"])

        return cls(
            model_name=data["model_name"],
            map50=data["map50"],
            map50_95=data["map50_95"],
            map75=data.get("map75"),
            precision=data.get("precision"),
            recall=data.get("recall"),
            f1_score=data.get("f1_score"),
            time_per_image_ms=data.get("time_per_image_ms", 0),
            fps=data.get("fps", 0),
            num_images=data.get("num_images", 0),
            total_time_seconds=data.get("total_time_seconds", 0),
            timestamp=data.get("timestamp", ""),
            ap_by_size=ap_by_size,
            timing_breakdown=timing_breakdown,
            model_info=model_info,
            hardware_info=hardware_info,
        )

    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for CSV export."""
        flat = {
            "model_name": self.model_name,
            "map50": self.map50,
            "map50_95": self.map50_95,
            "map75": self.map75,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "time_per_image_ms": self.time_per_image_ms,
            "fps": self.fps,
            "num_images": self.num_images,
            "total_time_seconds": self.total_time_seconds,
            "timestamp": self.timestamp,
        }

        # Flatten AP by size
        if self.ap_by_size:
            flat["ap_small"] = self.ap_by_size.small
            flat["ap_medium"] = self.ap_by_size.medium
            flat["ap_large"] = self.ap_by_size.large

        # Flatten timing breakdown
        if self.timing_breakdown:
            flat["preprocess_ms"] = self.timing_breakdown.preprocess_ms
            flat["inference_ms"] = self.timing_breakdown.inference_ms
            flat["postprocess_ms"] = self.timing_breakdown.postprocess_ms

        # Flatten model info
        if self.model_info:
            flat["parameters"] = self.model_info.parameters
            flat["gflops"] = self.model_info.gflops
            flat["input_size"] = self.model_info.input_size

        # Flatten hardware info
        if self.hardware_info:
            flat["device"] = self.hardware_info.device
            flat["device_name"] = self.hardware_info.device_name

        return flat


class BaseEvaluator(ABC):
    """Abstract base class for model evaluators."""

    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the evaluator.

        Args:
            device: PyTorch device to use. If None, auto-detects best available.
        """
        if device is None:
            from src.utils.device import get_device
            device = get_device()
        self.device = device
        self.model = None
        self._start_time: Optional[float] = None

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model."""
        pass

    @abstractmethod
    def load_model(self) -> None:
        """Load the model weights."""
        pass

    @abstractmethod
    def evaluate(self) -> EvaluationResults:
        """
        Run evaluation on the dataset.

        Returns:
            EvaluationResults containing all metrics.
        """
        pass

    def get_model_info(self) -> Optional[ModelInfo]:
        """
        Get model architecture information.
        Override in subclasses to provide model-specific info.
        """
        return None

    def get_hardware_info(self) -> HardwareInfo:
        """Get current hardware information."""
        return HardwareInfo.collect(self.device)

    def _start_timer(self) -> None:
        """Start the evaluation timer."""
        self._start_time = time.time()

    def _get_elapsed_time(self) -> float:
        """Get elapsed time since timer started."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def _calculate_timing_metrics(self, num_images: int) -> tuple[float, float, float]:
        """
        Calculate timing metrics.

        Args:
            num_images: Number of images processed.

        Returns:
            Tuple of (total_time_seconds, time_per_image_ms, fps).
        """
        elapsed = self._get_elapsed_time()
        time_per_image_ms = (elapsed / num_images) * 1000 if num_images > 0 else 0
        fps = 1000 / time_per_image_ms if time_per_image_ms > 0 else 0
        return elapsed, time_per_image_ms, fps

    def run(self) -> EvaluationResults:
        """
        Load model and run evaluation.

        Returns:
            EvaluationResults containing all metrics.
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {self.model_name}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")

        self.load_model()
        results = self.evaluate()

        print(f"\nResults: {results}")
        return results
