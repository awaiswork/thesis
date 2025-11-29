"""Faster R-CNN model evaluator."""

import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fcos_resnet50_fpn,
    FCOS_ResNet50_FPN_Weights,
)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.config import (
    COCO_SUBSET_IMAGES,
    COCO_SUBSET_ANN_FILE,
    SCORE_THRESHOLD,
    MAX_DETECTIONS,
)
from src.models.base import (
    BaseEvaluator,
    EvaluationResults,
    TimingBreakdown,
    ModelInfo,
    APBySize,
)


class FasterRCNNEvaluator(BaseEvaluator):
    """Evaluator for Faster R-CNN object detection model."""

    # Model configurations
    MODEL_CONFIGS = {
        "resnet50": {
            "name": "Faster R-CNN",
            "parameters": 43_712_278,
            "gflops": 134.4,
        },
        "fcos": {
            "name": "FCOS",
            "parameters": 32_269_600,
            "gflops": 128.2,
        },
    }

    def __init__(
        self,
        images_dir: Optional[str] = None,
        ann_file: Optional[str] = None,
        score_threshold: float = SCORE_THRESHOLD,
        max_detections: int = MAX_DETECTIONS,
        device: Optional[torch.device] = None,
        backbone: str = "resnet50",
    ):
        """
        Initialize Faster R-CNN evaluator.

        Args:
            images_dir: Path to images directory.
            ann_file: Path to COCO annotation file.
            score_threshold: Minimum score for detections.
            max_detections: Maximum detections per image.
            device: PyTorch device to use.
            backbone: Backbone type ('resnet50' or 'fcos' for anchor-free detection).
        """
        super().__init__(device)
        self.images_dir = Path(images_dir or COCO_SUBSET_IMAGES)
        self.ann_file = str(ann_file or COCO_SUBSET_ANN_FILE)
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.backbone = backbone

    @property
    def model_name(self) -> str:
        return self.MODEL_CONFIGS.get(self.backbone, self.MODEL_CONFIGS["resnet50"])["name"]

    def load_model(self) -> None:
        """Load Faster R-CNN or FCOS model with COCO pretrained weights."""
        if self.backbone == "fcos":
            print("Loading FCOS (ResNet50-FPN, COCO pretrained)...")
            weights = FCOS_ResNet50_FPN_Weights.COCO_V1
            self.model = fcos_resnet50_fpn(weights=weights)
        else:
            print("Loading Faster R-CNN (ResNet50-FPN v2, COCO pretrained)...")
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        self.model.to(self.device)
        self.model.eval()

    def get_model_info(self) -> ModelInfo:
        """Get model information."""
        config = self.MODEL_CONFIGS.get(self.backbone, self.MODEL_CONFIGS["resnet50"])
        return ModelInfo(
            name=self.model_name,
            parameters=config["parameters"],
            gflops=config["gflops"],
            input_size=800,  # Default min size
        )

    def _xyxy_to_xywh(self, box: List[float]) -> List[float]:
        """Convert box from [x1, y1, x2, y2] to [x, y, w, h] format."""
        x1, y1, x2, y2 = box
        return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

    def _load_image_as_tensor(self, img_path: Path) -> torch.Tensor:
        """Load image and convert to tensor."""
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img).astype("float32") / 255.0
        tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        return tensor

    def _run_inference(
        self,
        file_list: List[str],
        id_by_filename: Dict[str, int]
    ) -> tuple[List[Dict[str, Any]], TimingBreakdown]:
        """
        Run inference on image files.

        Args:
            file_list: List of image filenames.
            id_by_filename: Mapping from filename to image ID.

        Returns:
            Tuple of (detection results, timing breakdown).
        """
        detections = []
        num_images = len(file_list)

        total_preprocess = 0.0
        total_inference = 0.0
        total_postprocess = 0.0

        print(f"Running Faster R-CNN inference on {num_images} images...")

        with torch.no_grad():
            for fname in tqdm(file_list, desc="Inference"):
                # Preprocess timing
                t0 = time.time()
                img_path = self.images_dir / fname
                inp = self._load_image_as_tensor(img_path).to(self.device)
                t1 = time.time()
                total_preprocess += (t1 - t0) * 1000

                # Inference timing
                pred = self.model([inp])[0]
                t2 = time.time()
                total_inference += (t2 - t1) * 1000

                # Postprocess timing
                scores = pred["scores"].detach().cpu().numpy().tolist()
                labels = pred["labels"].detach().cpu().numpy().tolist()
                boxes = pred["boxes"].detach().cpu().numpy().tolist()

                # Filter by score and cap detections
                keep = [
                    (s, l, b)
                    for s, l, b in zip(scores, labels, boxes)
                    if s >= self.score_threshold
                ][:self.max_detections]

                img_id = id_by_filename[fname]
                for s, l, b in keep:
                    detections.append({
                        "image_id": int(img_id),
                        "category_id": int(l),
                        "bbox": self._xyxy_to_xywh(b),
                        "score": float(s)
                    })

                t3 = time.time()
                total_postprocess += (t3 - t2) * 1000

        timing = TimingBreakdown(
            preprocess_ms=total_preprocess / num_images,
            inference_ms=total_inference / num_images,
            postprocess_ms=total_postprocess / num_images,
        )

        print(f"Collected {len(detections)} detections")
        return detections, timing

    def evaluate(self) -> EvaluationResults:
        """
        Run Faster R-CNN evaluation on COCO subset.

        Returns:
            EvaluationResults with comprehensive metrics.
        """
        if self.model is None:
            self.load_model()

        # Load COCO ground truth
        coco_gt = COCO(self.ann_file)
        img_ids = coco_gt.getImgIds()
        imgs = coco_gt.loadImgs(img_ids)
        id_by_filename = {img["file_name"]: img["id"] for img in imgs}

        # Get image file list
        file_list = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        file_list = [f for f in file_list if f in id_by_filename]
        num_images = len(file_list)

        print(f"Found {num_images} images for evaluation")

        self._start_timer()

        # Run inference
        detections, timing = self._run_inference(file_list, id_by_filename)

        elapsed, time_per_img, fps = self._calculate_timing_metrics(num_images)

        if len(detections) == 0:
            print("WARNING: No detections produced")
            return EvaluationResults(
                model_name=self.model_name,
                map50=0.0,
                map50_95=0.0,
                time_per_image_ms=time_per_img,
                fps=fps,
                timing_breakdown=timing,
                num_images=num_images,
                total_time_seconds=elapsed,
                model_info=self.get_model_info(),
                hardware_info=self.get_hardware_info(),
            )

        # Run COCO evaluation
        print("\nRunning COCOeval...")
        coco_dt = coco_gt.loadRes(detections)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.maxDets = [1, 10, self.max_detections]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics from COCOeval stats
        stats = coco_eval.stats

        map50_95 = stats[0] * 100.0
        map50 = stats[1] * 100.0
        map75 = stats[2] * 100.0

        ap_by_size = APBySize(
            small=stats[3] * 100.0,
            medium=stats[4] * 100.0,
            large=stats[5] * 100.0,
        )

        return EvaluationResults(
            model_name=self.model_name,
            map50=map50,
            map50_95=map50_95,
            map75=map75,
            ap_by_size=ap_by_size,
            time_per_image_ms=time_per_img,
            fps=fps,
            timing_breakdown=timing,
            num_images=num_images,
            total_time_seconds=elapsed,
            model_info=self.get_model_info(),
            hardware_info=self.get_hardware_info(),
        )
