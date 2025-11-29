"""YOLOv8 model evaluator."""

from typing import Optional

import torch
from ultralytics import YOLO

from src.config import YOLOV8_WEIGHTS, YOLO_CONFIG_FILE, NUM_SAMPLES
from src.models.base import (
    BaseEvaluator,
    EvaluationResults,
    TimingBreakdown,
    ModelInfo,
    APBySize,
)


class YOLOv8Evaluator(BaseEvaluator):
    """Evaluator for YOLOv8 object detection model."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize YOLOv8 evaluator.

        Args:
            weights_path: Path to YOLOv8 weights file. Defaults to config value.
            config_path: Path to YOLO config YAML. Defaults to config value.
            device: PyTorch device to use.
        """
        super().__init__(device)
        self.weights_path = str(weights_path or YOLOV8_WEIGHTS)
        self.config_path = str(config_path or YOLO_CONFIG_FILE)

    @property
    def model_name(self) -> str:
        return "YOLOv8n"

    def load_model(self) -> None:
        """Load YOLOv8 model."""
        print(f"Loading YOLOv8 from: {self.weights_path}")
        self.model = YOLO(self.weights_path)

    def get_model_info(self) -> ModelInfo:
        """Get YOLOv8 model information."""
        if self.model is None:
            return ModelInfo(name=self.model_name)

        # Get model info from ultralytics
        try:
            info = self.model.info(verbose=False)
            # info returns (layers, parameters, gradients, gflops)
            if isinstance(info, tuple) and len(info) >= 4:
                return ModelInfo(
                    name=self.model_name,
                    parameters=int(info[1]),
                    gflops=float(info[3]),
                    input_size=640,
                )
        except Exception:
            pass

        return ModelInfo(
            name=self.model_name,
            parameters=3_151_904,  # YOLOv8n default
            gflops=8.7,
            input_size=640,
        )

    def evaluate(self) -> EvaluationResults:
        """
        Run YOLOv8 validation on COCO subset.

        Returns:
            EvaluationResults with comprehensive metrics.
        """
        if self.model is None:
            self.load_model()

        print(f"Validating on: {self.config_path}")

        self._start_timer()

        # Run YOLO validation
        metrics = self.model.val(
            data=self.config_path,
            device=str(self.device),
            verbose=False
        )

        elapsed, time_per_img, fps = self._calculate_timing_metrics(NUM_SAMPLES)

        # Extract timing breakdown from YOLO speed dict
        timing = TimingBreakdown()
        if hasattr(metrics, 'speed') and metrics.speed:
            timing = TimingBreakdown(
                preprocess_ms=metrics.speed.get('preprocess', 0),
                inference_ms=metrics.speed.get('inference', 0),
                postprocess_ms=metrics.speed.get('postprocess', 0),
            )

        # Extract AP by size if available
        ap_by_size = None
        try:
            # YOLO provides these in the results
            if hasattr(metrics, 'results_dict'):
                rd = metrics.results_dict
                ap_by_size = APBySize(
                    small=rd.get('metrics/APsmall(B)', 0) * 100,
                    medium=rd.get('metrics/APmedium(B)', 0) * 100,
                    large=rd.get('metrics/APlarge(B)', 0) * 100,
                )
        except Exception:
            pass

        # Extract mAP@0.75 if available
        map75 = None
        try:
            if hasattr(metrics.box, 'maps') and len(metrics.box.maps) > 5:
                # maps contains AP at different IoU thresholds
                map75 = metrics.box.maps[5] * 100  # Index 5 is IoU=0.75
        except Exception:
            pass

        return EvaluationResults(
            model_name=self.model_name,
            map50=metrics.box.map50 * 100,
            map50_95=metrics.box.map * 100,
            map75=map75,
            precision=metrics.box.mp * 100,
            recall=metrics.box.mr * 100,
            ap_by_size=ap_by_size,
            time_per_image_ms=time_per_img,
            fps=fps,
            timing_breakdown=timing,
            num_images=NUM_SAMPLES,
            total_time_seconds=elapsed,
            model_info=self.get_model_info(),
            hardware_info=self.get_hardware_info(),
        )
