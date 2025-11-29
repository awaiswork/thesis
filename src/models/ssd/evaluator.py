"""SSD model evaluator."""

from typing import Optional, List, Dict, Any

import torch
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.config import (
    COCO_SUBSET_IMAGES,
    COCO_SUBSET_ANN_FILE,
    SCORE_THRESHOLD,
)
from src.models.base import (
    BaseEvaluator,
    EvaluationResults,
    TimingBreakdown,
    ModelInfo,
    APBySize,
)


class SSDEvaluator(BaseEvaluator):
    """Evaluator for SSD300 object detection model."""

    def __init__(
        self,
        images_dir: Optional[str] = None,
        ann_file: Optional[str] = None,
        score_threshold: float = SCORE_THRESHOLD,
        device: Optional[torch.device] = None
    ):
        """
        Initialize SSD evaluator.

        Args:
            images_dir: Path to images directory.
            ann_file: Path to COCO annotation file.
            score_threshold: Minimum score for detections.
            device: PyTorch device to use.
        """
        super().__init__(device)
        self.images_dir = str(images_dir or COCO_SUBSET_IMAGES)
        self.ann_file = str(ann_file or COCO_SUBSET_ANN_FILE)
        self.score_threshold = score_threshold

    @property
    def model_name(self) -> str:
        return "SSD300"

    def load_model(self) -> None:
        """Load SSD300 model with COCO pretrained weights."""
        print("Loading SSD300_VGG16 (COCO pretrained)...")
        weights = SSD300_VGG16_Weights.COCO_V1
        self.model = ssd300_vgg16(weights=weights)
        self.model.to(self.device)
        self.model.eval()

    def get_model_info(self) -> ModelInfo:
        """Get SSD model information."""
        return ModelInfo(
            name=self.model_name,
            parameters=35_641_826,  # SSD300 VGG16 parameters
            gflops=34.1,  # Approximate GFLOPs for SSD300
            input_size=300,
        )

    def _convert_boxes_to_xywh(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from [x1, y1, x2, y2] to [x, y, w, h] format."""
        boxes_xywh = boxes.clone()
        boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # w = x2 - x1
        boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # h = y2 - y1
        return boxes_xywh

    def _run_inference(self, dataset: CocoDetection) -> tuple[List[Dict[str, Any]], TimingBreakdown]:
        """
        Run inference on the dataset.

        Args:
            dataset: COCO detection dataset.

        Returns:
            Tuple of (detection results, timing breakdown).
        """
        import time

        coco_results = []
        num_images = len(dataset)

        total_preprocess = 0.0
        total_inference = 0.0
        total_postprocess = 0.0

        print(f"Running SSD inference on {num_images} images...")

        with torch.no_grad():
            for idx in range(num_images):
                # Preprocess timing
                t0 = time.time()
                img, _ = dataset[idx]
                img_id = dataset.ids[idx]
                img_tensor = img.to(self.device)
                t1 = time.time()
                total_preprocess += (t1 - t0) * 1000

                # Inference timing
                outputs = self.model([img_tensor])[0]
                t2 = time.time()
                total_inference += (t2 - t1) * 1000

                # Postprocess timing
                boxes = outputs["boxes"].cpu()
                scores = outputs["scores"].cpu()
                labels = outputs["labels"].cpu()

                # Filter by score threshold
                keep = scores >= self.score_threshold
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                if boxes.numel() > 0:
                    boxes_xywh = self._convert_boxes_to_xywh(boxes)

                    for box, score, label in zip(boxes_xywh, scores, labels):
                        coco_results.append({
                            "image_id": int(img_id),
                            "category_id": int(label),
                            "bbox": [
                                float(box[0]),
                                float(box[1]),
                                float(box[2]),
                                float(box[3]),
                            ],
                            "score": float(score),
                        })

                t3 = time.time()
                total_postprocess += (t3 - t2) * 1000

        timing = TimingBreakdown(
            preprocess_ms=total_preprocess / num_images,
            inference_ms=total_inference / num_images,
            postprocess_ms=total_postprocess / num_images,
        )

        print(f"Collected {len(coco_results)} detections")
        return coco_results, timing

    def evaluate(self) -> EvaluationResults:
        """
        Run SSD evaluation on COCO subset.

        Returns:
            EvaluationResults with comprehensive metrics.
        """
        if self.model is None:
            self.load_model()

        # Load dataset
        dataset = CocoDetection(
            root=self.images_dir,
            annFile=self.ann_file,
            transform=ToTensor()
        )
        num_images = len(dataset)
        print(f"Loaded {num_images} images for evaluation")

        # Load COCO ground truth
        coco_gt = COCO(self.ann_file)

        self._start_timer()

        # Run inference
        coco_results, timing = self._run_inference(dataset)

        elapsed, time_per_img, fps = self._calculate_timing_metrics(num_images)

        if len(coco_results) == 0:
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
        print("Running COCOeval...")
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.imgIds = dataset.ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics from COCOeval stats
        # stats[0] = AP @[0.5:0.95]
        # stats[1] = AP @0.5
        # stats[2] = AP @0.75
        # stats[3] = AP small
        # stats[4] = AP medium
        # stats[5] = AP large
        # stats[8] = AR @100
        stats = coco_eval.stats

        map50_95 = stats[0] * 100.0
        map50 = stats[1] * 100.0
        map75 = stats[2] * 100.0
        recall = stats[8] * 100.0

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
            recall=recall,
            ap_by_size=ap_by_size,
            time_per_image_ms=time_per_img,
            fps=fps,
            timing_breakdown=timing,
            num_images=num_images,
            total_time_seconds=elapsed,
            model_info=self.get_model_info(),
            hardware_info=self.get_hardware_info(),
        )
