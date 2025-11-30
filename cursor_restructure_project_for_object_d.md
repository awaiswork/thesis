# Restructure project for object detection models
_Exported on 29/11/2025 at 16:35:33 EET from Cursor (2.1.26)_

---

**User**

Go through the whole project. Read files. I have written the code to do performance analysis of 4 object detection models. The whole project was running on PyCharm before in a notebook @code.ipynb  but I want to change the noteboook format and divide code into mdules in files and dirs. Plan the restructure of the project and also to divide the code in different directories for different models. 

---

**Cursor**

I'll explore your project to understand the current structure and the code you've written for performance analysis of the 4 object detection models.

I've explored your project and found the following:

**Current project structure:**
- `code.ipynb` - Contains only a "Hello World" template, not the actual performance analysis code
- Model weights: `yolov8n.pt`, `yolov10n.pt`, and `models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`
- COCO dataset subset with 5000 images and annotations
- Results from multiple validation runs in `runs/detect/`

However, I don't see the actual performance analysis code for the 4 object detection models in the notebook. The notebook only contains a sample template with `print("Hello World!")`.

Great! So you're analyzing **YOLOv8, YOLOv10, Faster R-CNN, and SSD** models.

Please paste your performance analysis code in the next message, and I'll create a plan to restructure it into a modular project with separate directories for each model.

---

**User**

@thesis.ipynb 

---

**Cursor**

Now I can see your complete performance analysis code in `thesis.ipynb`. You have evaluation code for **YOLOv8n**, **YOLOv10n**, **SSD300**, and **Faster R-CNN** on a COCO subset.

Let me create a restructuring plan for you:

---

**User**

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

---

**Cursor**

I'll implement the plan step by step. Let me start with the first todo - creating the directory structure.

```python
"""Object Detection Performance Analysis Package."""

```

```python
"""Data loading and preprocessing utilities."""

from .coco_loader import load_coco_annotations
from .subset_creator import create_subset
from .yolo_converter import convert_to_yolo_format

__all__ = ["load_coco_annotations", "create_subset", "convert_to_yolo_format"]

```

```python
"""Object detection model evaluators."""

from .base import BaseEvaluator

__all__ = ["BaseEvaluator"]

```

```python
"""YOLOv8 model evaluator."""

from .evaluator import YOLOv8Evaluator

__all__ = ["YOLOv8Evaluator"]

```

```python
"""YOLOv10 model evaluator."""

from .evaluator import YOLOv10Evaluator

__all__ = ["YOLOv10Evaluator"]

```

```python
"""SSD model evaluator."""

from .evaluator import SSDEvaluator

__all__ = ["SSDEvaluator"]

```

```python
"""Faster R-CNN model evaluator."""

from .evaluator import FasterRCNNEvaluator

__all__ = ["FasterRCNNEvaluator"]

```

```python
"""Utility functions for object detection evaluation."""

from .device import get_device
from .metrics import extract_metrics, format_results
from .visualization import plot_comparison

__all__ = ["get_device", "extract_metrics", "format_results", "plot_comparison"]

```

```python
"""Scripts for running object detection evaluation pipeline."""

```

```python
"""Global configuration for object detection evaluation."""

from pathlib import Path

# =============================================================================
# Path Configuration
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATAFILES_DIR = PROJECT_ROOT / "datafiles"
ANNOTATIONS_ZIP = DATAFILES_DIR / "annotations_trainval2017.zip"
IMAGES_ZIP = DATAFILES_DIR / "val2017.zip"

# Dataset paths
COCO_SUBSET_DIR = PROJECT_ROOT / "coco_subset"
COCO_SUBSET_IMAGES = COCO_SUBSET_DIR / "images"
COCO_SUBSET_LABELS = COCO_SUBSET_DIR / "labels"
COCO_SUBSET_ANNOTATIONS = COCO_SUBSET_DIR / "annotations"
COCO_SUBSET_ANN_FILE = COCO_SUBSET_ANNOTATIONS / "instances_val2017_subset.json"

# YOLO config
YOLO_CONFIG_FILE = PROJECT_ROOT / "coco_subset.yaml"

# Model weights
WEIGHTS_DIR = PROJECT_ROOT / "models"
YOLOV8_WEIGHTS = PROJECT_ROOT / "yolov8n.pt"
YOLOV10_WEIGHTS = PROJECT_ROOT / "yolov10n.pt"
FASTER_RCNN_WEIGHTS = WEIGHTS_DIR / "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"

# Output paths
RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"

# =============================================================================
# Dataset Configuration
# =============================================================================

# Number of images to sample for evaluation
NUM_SAMPLES = 5000

# Random seed for reproducibility
RANDOM_SEED = 42

# =============================================================================
# Model Evaluation Configuration
# =============================================================================

# Score threshold for detections (low for fair COCO eval)
SCORE_THRESHOLD = 0.05

# Maximum detections per image
MAX_DETECTIONS = 100

# Batch size for evaluation
BATCH_SIZE = 16

# =============================================================================
# COCO Class Names (80 classes)
# =============================================================================

COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
    38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup", 42: "fork",
    43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple",
    48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog",
    53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch",
    58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet", 62: "tv",
    63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
    68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator",
    73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}

```

```python
"""Load COCO annotations from ZIP file."""

import json
import zipfile
from pathlib import Path
from typing import Dict, Any

from src.config import ANNOTATIONS_ZIP


def load_coco_annotations(
    zip_path: Path = ANNOTATIONS_ZIP,
    annotation_file: str = "annotations/instances_val2017.json"
) -> Dict[str, Any]:
    """
    Load COCO annotations directly from a ZIP file.

    Args:
        zip_path: Path to the annotations ZIP file.
        annotation_file: Path within the ZIP to the annotation JSON.

    Returns:
        Dictionary containing COCO annotations with keys:
        - info: Dataset info
        - licenses: License information
        - images: List of image metadata
        - annotations: List of object annotations
        - categories: List of category definitions

    Raises:
        FileNotFoundError: If the ZIP file doesn't exist.
        KeyError: If the annotation file isn't found in the ZIP.
    """
    if not zip_path.exists():
        raise FileNotFoundError(
            f"ZIP file not found at: {zip_path}\n"
            "Please download COCO annotations and place the ZIP in the datafiles folder."
        )

    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(annotation_file) as f:
            coco = json.load(f)

    print(f"Loaded COCO annotations from: {zip_path}")
    print(f"  Total images: {len(coco['images'])}")
    print(f"  Total annotations: {len(coco['annotations'])}")

    return coco


def get_category_mapping(coco: Dict[str, Any]) -> Dict[int, int]:
    """
    Create a mapping from COCO category IDs to 0-indexed class indices.

    Args:
        coco: COCO annotation dictionary.

    Returns:
        Dictionary mapping category_id -> class_index (0-79).
    """
    return {cat['id']: idx for idx, cat in enumerate(coco['categories'])}


def get_image_info(coco: Dict[str, Any], image_id: int) -> Dict[str, Any]:
    """
    Get image metadata by image ID.

    Args:
        coco: COCO annotation dictionary.
        image_id: The image ID to look up.

    Returns:
        Image metadata dictionary with keys like 'file_name', 'width', 'height'.

    Raises:
        StopIteration: If image_id is not found.
    """
    return next(img for img in coco['images'] if img['id'] == image_id)

```

```python
"""Create a subset of COCO dataset for evaluation."""

import json
import random
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Set

from src.config import (
    IMAGES_ZIP,
    COCO_SUBSET_DIR,
    COCO_SUBSET_IMAGES,
    COCO_SUBSET_ANNOTATIONS,
    COCO_SUBSET_ANN_FILE,
    NUM_SAMPLES,
    RANDOM_SEED,
)


def sample_images(
    coco: Dict[str, Any],
    num_samples: int = NUM_SAMPLES,
    seed: int = RANDOM_SEED
) -> tuple[List[Dict], Set[int]]:
    """
    Randomly sample images from COCO dataset.

    Args:
        coco: COCO annotation dictionary.
        num_samples: Number of images to sample.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (sampled_images, sampled_image_ids).
    """
    random.seed(seed)
    sampled_images = random.sample(coco['images'], min(num_samples, len(coco['images'])))
    sampled_image_ids = {img['id'] for img in sampled_images}

    print(f"Sampled {len(sampled_images)} images with seed {seed}")
    return sampled_images, sampled_image_ids


def filter_annotations(
    coco: Dict[str, Any],
    image_ids: Set[int]
) -> List[Dict]:
    """
    Filter annotations to only include those for specified image IDs.

    Args:
        coco: COCO annotation dictionary.
        image_ids: Set of image IDs to keep.

    Returns:
        List of filtered annotations.
    """
    filtered = [ann for ann in coco['annotations'] if ann['image_id'] in image_ids]
    print(f"Filtered to {len(filtered)} annotations")
    return filtered


def extract_images(
    sampled_images: List[Dict],
    images_zip: Path = IMAGES_ZIP,
    output_dir: Path = COCO_SUBSET_IMAGES,
    zip_prefix: str = "val2017"
) -> int:
    """
    Extract sampled images from ZIP to output directory.

    Args:
        sampled_images: List of image metadata dictionaries.
        images_zip: Path to the images ZIP file.
        output_dir: Directory to extract images to.
        zip_prefix: Prefix path within the ZIP file.

    Returns:
        Number of images extracted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(images_zip, 'r') as z:
        for img in sampled_images:
            file_name = img['file_name']
            zip_path = f"{zip_prefix}/{file_name}"
            output_path = output_dir / file_name

            with z.open(zip_path) as source:
                output_path.write_bytes(source.read())

    extracted = len(list(output_dir.glob('*.jpg')))
    print(f"Extracted {extracted} images to: {output_dir}")
    return extracted


def save_subset_annotations(
    coco: Dict[str, Any],
    sampled_images: List[Dict],
    sampled_annotations: List[Dict],
    output_path: Path = COCO_SUBSET_ANN_FILE
) -> Path:
    """
    Save subset annotations to a new JSON file.

    Args:
        coco: Original COCO annotation dictionary (for info, licenses, categories).
        sampled_images: List of sampled image metadata.
        sampled_annotations: List of annotations for sampled images.
        output_path: Path to save the subset annotations.

    Returns:
        Path to the saved annotation file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subset_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": sampled_images,
        "annotations": sampled_annotations,
        "categories": coco.get("categories", []),
    }

    with open(output_path, 'w') as f:
        json.dump(subset_coco, f)

    print(f"Saved subset annotations to: {output_path}")
    return output_path


def create_subset(
    coco: Dict[str, Any],
    num_samples: int = NUM_SAMPLES,
    seed: int = RANDOM_SEED,
    images_zip: Path = IMAGES_ZIP,
    output_dir: Path = COCO_SUBSET_DIR
) -> Dict[str, Any]:
    """
    Create a complete COCO subset with images and annotations.

    This is the main entry point that orchestrates the subset creation.

    Args:
        coco: COCO annotation dictionary.
        num_samples: Number of images to sample.
        seed: Random seed for reproducibility.
        images_zip: Path to the images ZIP file.
        output_dir: Base directory for the subset.

    Returns:
        Dictionary with subset info including paths and counts.
    """
    print(f"\nCreating COCO subset with {num_samples} images...")

    # Sample images
    sampled_images, sampled_ids = sample_images(coco, num_samples, seed)

    # Filter annotations
    sampled_annotations = filter_annotations(coco, sampled_ids)

    # Extract images
    images_dir = output_dir / "images"
    num_extracted = extract_images(sampled_images, images_zip, images_dir)

    # Save annotations
    ann_path = output_dir / "annotations" / "instances_val2017_subset.json"
    save_subset_annotations(coco, sampled_images, sampled_annotations, ann_path)

    return {
        "images_dir": images_dir,
        "annotations_file": ann_path,
        "num_images": num_extracted,
        "num_annotations": len(sampled_annotations),
    }

```

```python
"""Convert COCO annotations to YOLO format."""

from pathlib import Path
from typing import Dict, Any, List

from src.config import COCO_SUBSET_LABELS
from src.data.coco_loader import get_category_mapping


def coco_bbox_to_yolo(
    bbox: List[float],
    img_width: int,
    img_height: int
) -> tuple[float, float, float, float]:
    """
    Convert COCO bbox format to YOLO format.

    COCO format: [x_min, y_min, width, height] in pixels
    YOLO format: [x_center, y_center, width, height] normalized to [0, 1]

    Args:
        bbox: COCO bounding box [x_min, y_min, width, height].
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        Tuple of (x_center, y_center, width, height) normalized.
    """
    x_min, y_min, bw, bh = bbox

    # Calculate center coordinates
    x_center = x_min + bw / 2
    y_center = y_min + bh / 2

    # Normalize to [0, 1]
    norm_x = x_center / img_width
    norm_y = y_center / img_height
    norm_w = bw / img_width
    norm_h = bh / img_height

    return norm_x, norm_y, norm_w, norm_h


def build_yolo_labels(
    sampled_images: List[Dict],
    sampled_annotations: List[Dict],
    category_mapping: Dict[int, int]
) -> Dict[int, List[str]]:
    """
    Build YOLO format label strings for each image.

    Args:
        sampled_images: List of image metadata dictionaries.
        sampled_annotations: List of COCO annotations.
        category_mapping: Mapping from COCO category_id to class index.

    Returns:
        Dictionary mapping image_id -> list of YOLO label lines.
    """
    # Create lookup for image dimensions
    img_info = {img['id']: img for img in sampled_images}

    # Initialize labels dictionary
    yolo_labels: Dict[int, List[str]] = {img['id']: [] for img in sampled_images}

    for ann in sampled_annotations:
        img_id = ann['image_id']
        cat_id = ann['category_id']

        # Get class index (0-79)
        class_idx = category_mapping[cat_id]

        # Get image dimensions
        img = img_info[img_id]
        img_w, img_h = img['width'], img['height']

        # Convert bbox to YOLO format
        norm_x, norm_y, norm_w, norm_h = coco_bbox_to_yolo(ann['bbox'], img_w, img_h)

        # Create YOLO label line
        label_line = f"{class_idx} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}"
        yolo_labels[img_id].append(label_line)

    return yolo_labels


def write_yolo_labels(
    sampled_images: List[Dict],
    yolo_labels: Dict[int, List[str]],
    output_dir: Path = COCO_SUBSET_LABELS
) -> int:
    """
    Write YOLO label files to disk.

    Args:
        sampled_images: List of image metadata dictionaries.
        yolo_labels: Dictionary mapping image_id -> list of label lines.
        output_dir: Directory to write label files.

    Returns:
        Number of label files written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for img in sampled_images:
        img_id = img['id']
        file_name = img['file_name']
        label_lines = yolo_labels.get(img_id, [])

        # Replace .jpg with .txt
        label_path = output_dir / file_name.replace('.jpg', '.txt')

        with open(label_path, 'w') as f:
            f.write("\n".join(label_lines))

    num_files = len(list(output_dir.glob('*.txt')))
    print(f"Created {num_files} YOLO label files in: {output_dir}")
    return num_files


def convert_to_yolo_format(
    coco: Dict[str, Any],
    sampled_images: List[Dict],
    sampled_annotations: List[Dict],
    output_dir: Path = COCO_SUBSET_LABELS
) -> int:
    """
    Convert COCO annotations to YOLO format and write to disk.

    This is the main entry point for YOLO format conversion.

    Args:
        coco: Original COCO annotation dictionary (for category mapping).
        sampled_images: List of sampled image metadata.
        sampled_annotations: List of annotations for sampled images.
        output_dir: Directory to write label files.

    Returns:
        Number of label files created.
    """
    print("\nConverting annotations to YOLO format...")

    # Get category mapping
    category_mapping = get_category_mapping(coco)

    # Build YOLO labels
    yolo_labels = build_yolo_labels(sampled_images, sampled_annotations, category_mapping)

    # Write label files
    num_files = write_yolo_labels(sampled_images, yolo_labels, output_dir)

    return num_files

```

```python
"""Base class for object detection model evaluators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import time

import torch


@dataclass
class EvaluationResults:
    """Container for evaluation metrics."""

    model_name: str
    map50: float  # mAP@0.5
    map50_95: float  # mAP@[0.5:0.95]
    precision: Optional[float] = None
    recall: Optional[float] = None
    time_per_image_ms: float = 0.0
    fps: float = 0.0
    num_images: int = 0
    total_time_seconds: float = 0.0

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
        parts.extend([
            f"Time/image: {self.time_per_image_ms:.1f}ms",
            f"FPS: {self.fps:.1f}",
        ])
        return " | ".join(parts)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "map50": self.map50,
            "map50_95": self.map50_95,
            "precision": self.precision,
            "recall": self.recall,
            "time_per_image_ms": self.time_per_image_ms,
            "fps": self.fps,
            "num_images": self.num_images,
            "total_time_seconds": self.total_time_seconds,
        }


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

```

```python
"""YOLOv8 model evaluator."""

from typing import Optional

import torch
from ultralytics import YOLO

from src.config import YOLOV8_WEIGHTS, YOLO_CONFIG_FILE, NUM_SAMPLES
from src.models.base import BaseEvaluator, EvaluationResults


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

    def evaluate(self) -> EvaluationResults:
        """
        Run YOLOv8 validation on COCO subset.

        Returns:
            EvaluationResults with metrics.
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

        # Extract metrics from YOLO results
        return EvaluationResults(
            model_name=self.model_name,
            map50=metrics.box.map50 * 100,
            map50_95=metrics.box.map * 100,
            precision=metrics.box.mp * 100,
            recall=metrics.box.mr * 100,
            time_per_image_ms=time_per_img,
            fps=fps,
            num_images=NUM_SAMPLES,
            total_time_seconds=elapsed,
        )

```

```python
"""YOLOv10 model evaluator."""

from typing import Optional

import torch
from ultralytics import YOLO

from src.config import YOLOV10_WEIGHTS, YOLO_CONFIG_FILE, NUM_SAMPLES
from src.models.base import BaseEvaluator, EvaluationResults


class YOLOv10Evaluator(BaseEvaluator):
    """Evaluator for YOLOv10 object detection model."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize YOLOv10 evaluator.

        Args:
            weights_path: Path to YOLOv10 weights file. Defaults to config value.
            config_path: Path to YOLO config YAML. Defaults to config value.
            device: PyTorch device to use.
        """
        super().__init__(device)
        self.weights_path = str(weights_path or YOLOV10_WEIGHTS)
        self.config_path = str(config_path or YOLO_CONFIG_FILE)

    @property
    def model_name(self) -> str:
        return "YOLOv10n"

    def load_model(self) -> None:
        """Load YOLOv10 model."""
        print(f"Loading YOLOv10 from: {self.weights_path}")
        self.model = YOLO(self.weights_path)

    def evaluate(self) -> EvaluationResults:
        """
        Run YOLOv10 validation on COCO subset.

        Returns:
            EvaluationResults with metrics.
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

        # Extract metrics from YOLO results
        return EvaluationResults(
            model_name=self.model_name,
            map50=metrics.box.map50 * 100,
            map50_95=metrics.box.map * 100,
            precision=metrics.box.mp * 100,
            recall=metrics.box.mr * 100,
            time_per_image_ms=time_per_img,
            fps=fps,
            num_images=NUM_SAMPLES,
            total_time_seconds=elapsed,
        )

```

```python
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
from src.models.base import BaseEvaluator, EvaluationResults


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

    def _convert_boxes_to_xywh(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from [x1, y1, x2, y2] to [x, y, w, h] format."""
        boxes_xywh = boxes.clone()
        boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # w = x2 - x1
        boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # h = y2 - y1
        return boxes_xywh

    def _run_inference(self, dataset: CocoDetection) -> List[Dict[str, Any]]:
        """
        Run inference on the dataset.

        Args:
            dataset: COCO detection dataset.

        Returns:
            List of detection results in COCO format.
        """
        coco_results = []
        num_images = len(dataset)

        print(f"Running SSD inference on {num_images} images...")

        with torch.no_grad():
            for idx in range(num_images):
                img, _ = dataset[idx]
                img_id = dataset.ids[idx]

                # Run model
                outputs = self.model([img.to(self.device)])[0]

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

        print(f"Collected {len(coco_results)} detections")
        return coco_results

    def evaluate(self) -> EvaluationResults:
        """
        Run SSD evaluation on COCO subset.

        Returns:
            EvaluationResults with metrics.
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
        coco_results = self._run_inference(dataset)

        elapsed, time_per_img, fps = self._calculate_timing_metrics(num_images)

        if len(coco_results) == 0:
            print("WARNING: No detections produced")
            return EvaluationResults(
                model_name=self.model_name,
                map50=0.0,
                map50_95=0.0,
                time_per_image_ms=time_per_img,
                fps=fps,
                num_images=num_images,
                total_time_seconds=elapsed,
            )

        # Run COCO evaluation
        print("Running COCOeval...")
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.imgIds = dataset.ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics
        # stats[0] = AP @[0.5:0.95], stats[1] = AP @0.5, stats[8] = AR @100
        map50_95 = coco_eval.stats[0] * 100.0
        map50 = coco_eval.stats[1] * 100.0
        recall = coco_eval.stats[8] * 100.0

        return EvaluationResults(
            model_name=self.model_name,
            map50=map50,
            map50_95=map50_95,
            recall=recall,
            time_per_image_ms=time_per_img,
            fps=fps,
            num_images=num_images,
            total_time_seconds=elapsed,
        )

```

```python
"""Faster R-CNN model evaluator."""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from PIL import Image
from tqdm import tqdm
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights
)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.config import (
    COCO_SUBSET_IMAGES,
    COCO_SUBSET_ANN_FILE,
    SCORE_THRESHOLD,
    MAX_DETECTIONS,
)
from src.models.base import BaseEvaluator, EvaluationResults


class FasterRCNNEvaluator(BaseEvaluator):
    """Evaluator for Faster R-CNN object detection model."""

    def __init__(
        self,
        images_dir: Optional[str] = None,
        ann_file: Optional[str] = None,
        score_threshold: float = SCORE_THRESHOLD,
        max_detections: int = MAX_DETECTIONS,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Faster R-CNN evaluator.

        Args:
            images_dir: Path to images directory.
            ann_file: Path to COCO annotation file.
            score_threshold: Minimum score for detections.
            max_detections: Maximum detections per image.
            device: PyTorch device to use.
        """
        super().__init__(device)
        self.images_dir = Path(images_dir or COCO_SUBSET_IMAGES)
        self.ann_file = str(ann_file or COCO_SUBSET_ANN_FILE)
        self.score_threshold = score_threshold
        self.max_detections = max_detections

    @property
    def model_name(self) -> str:
        return "Faster R-CNN"

    def load_model(self) -> None:
        """Load Faster R-CNN model with COCO pretrained weights."""
        print("Loading Faster R-CNN (ResNet50-FPN v2, COCO pretrained)...")
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        self.model.to(self.device)
        self.model.eval()

    def _xyxy_to_xywh(self, box: List[float]) -> List[float]:
        """Convert box from [x1, y1, x2, y2] to [x, y, w, h] format."""
        x1, y1, x2, y2 = box
        return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

    def _load_image_as_tensor(self, img_path: Path) -> torch.Tensor:
        """Load image and convert to tensor."""
        img = Image.open(img_path).convert("RGB")
        # Convert to tensor: HxWxC [0..255] -> CxHxW [0..1]
        import numpy as np
        img_array = np.array(img).astype("float32") / 255.0
        tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        return tensor

    def _run_inference(
        self,
        file_list: List[str],
        id_by_filename: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """
        Run inference on image files.

        Args:
            file_list: List of image filenames.
            id_by_filename: Mapping from filename to image ID.

        Returns:
            List of detection results in COCO format.
        """
        detections = []

        print(f"Running Faster R-CNN inference on {len(file_list)} images...")

        with torch.no_grad():
            for fname in tqdm(file_list, desc="Inference"):
                img_path = self.images_dir / fname
                inp = self._load_image_as_tensor(img_path).to(self.device)

                pred = self.model([inp])[0]

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

        print(f"Collected {len(detections)} detections")
        return detections

    def evaluate(self) -> EvaluationResults:
        """
        Run Faster R-CNN evaluation on COCO subset.

        Returns:
            EvaluationResults with metrics.
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
        detections = self._run_inference(file_list, id_by_filename)

        elapsed, time_per_img, fps = self._calculate_timing_metrics(num_images)

        if len(detections) == 0:
            print("WARNING: No detections produced")
            return EvaluationResults(
                model_name=self.model_name,
                map50=0.0,
                map50_95=0.0,
                time_per_image_ms=time_per_img,
                fps=fps,
                num_images=num_images,
                total_time_seconds=elapsed,
            )

        # Run COCO evaluation
        print("\nRunning COCOeval...")
        coco_dt = coco_gt.loadRes(detections)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.maxDets = [1, 10, self.max_detections]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics
        map50_95 = coco_eval.stats[0] * 100.0
        map50 = coco_eval.stats[1] * 100.0

        return EvaluationResults(
            model_name=self.model_name,
            map50=map50,
            map50_95=map50_95,
            time_per_image_ms=time_per_img,
            fps=fps,
            num_images=num_images,
            total_time_seconds=elapsed,
        )

```

```python
"""Device detection utilities."""

import torch


def get_device() -> torch.device:
    """
    Detect and return the best available device.

    Priority: CUDA > MPS (Apple Silicon) > CPU

    Returns:
        torch.device for the best available hardware.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS GPU backend.")
    else:
        device = torch.device("cpu")
        print("No GPU available. Using CPU.")

    print(f"PyTorch version: {torch.__version__}")
    return device


def get_device_string() -> str:
    """
    Get device as a string (for YOLO compatibility).

    Returns:
        Device name as string: "cuda", "mps", or "cpu".
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

```

```python
"""Metrics extraction and formatting utilities."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.models.base import EvaluationResults


def extract_metrics(results: EvaluationResults) -> Dict[str, Any]:
    """
    Extract metrics from evaluation results as a dictionary.

    Args:
        results: EvaluationResults object.

    Returns:
        Dictionary of metrics.
    """
    return results.to_dict()


def format_results(results: EvaluationResults) -> str:
    """
    Format evaluation results as a human-readable string.

    Args:
        results: EvaluationResults object.

    Returns:
        Formatted string.
    """
    return str(results)


def format_results_table(results_list: List[EvaluationResults]) -> str:
    """
    Format multiple evaluation results as a comparison table.

    Args:
        results_list: List of EvaluationResults objects.

    Returns:
        Formatted table string.
    """
    if not results_list:
        return "No results to display."

    # Header
    header = (
        f"{'Model':<15} {'mAP@0.5':>10} {'mAP@0.5:0.95':>14} "
        f"{'Precision':>10} {'Recall':>10} {'ms/img':>10} {'FPS':>8}"
    )
    separator = "-" * len(header)

    lines = [separator, header, separator]

    for r in results_list:
        prec = f"{r.precision:.1f}%" if r.precision is not None else "N/A"
        rec = f"{r.recall:.1f}%" if r.recall is not None else "N/A"
        line = (
            f"{r.model_name:<15} {r.map50:>9.1f}% {r.map50_95:>13.1f}% "
            f"{prec:>10} {rec:>10} {r.time_per_image_ms:>9.1f} {r.fps:>8.1f}"
        )
        lines.append(line)

    lines.append(separator)
    return "\n".join(lines)


def save_results_json(
    results_list: List[EvaluationResults],
    output_path: Path
) -> None:
    """
    Save evaluation results to a JSON file.

    Args:
        results_list: List of EvaluationResults objects.
        output_path: Path to save the JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "results": [r.to_dict() for r in results_list]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to: {output_path}")


def load_results_json(input_path: Path) -> List[Dict[str, Any]]:
    """
    Load evaluation results from a JSON file.

    Args:
        input_path: Path to the JSON file.

    Returns:
        List of result dictionaries.
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    return data.get("results", [])


def calculate_speedup(
    baseline_fps: float,
    comparison_fps: float
) -> float:
    """
    Calculate speedup factor between two models.

    Args:
        baseline_fps: FPS of the baseline model.
        comparison_fps: FPS of the comparison model.

    Returns:
        Speedup factor (comparison_fps / baseline_fps).
    """
    if baseline_fps <= 0:
        return 0.0
    return comparison_fps / baseline_fps

```

```python
"""Visualization utilities for comparing model performance."""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.models.base import EvaluationResults
from src.config import RESULTS_DIR


def plot_comparison(
    results_list: List[EvaluationResults],
    output_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """
    Create bar charts comparing model performance.

    Args:
        results_list: List of EvaluationResults objects.
        output_path: Path to save the plot. If None, uses default.
        show: Whether to display the plot.
    """
    if not results_list:
        print("No results to plot.")
        return

    if output_path is None:
        output_path = RESULTS_DIR / "comparison.png"

    # Extract data
    models = [r.model_name for r in results_list]
    map50 = [r.map50 for r in results_list]
    map50_95 = [r.map50_95 for r in results_list]
    fps = [r.fps for r in results_list]

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Colors for each model
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    # mAP@0.5 comparison
    ax1 = axes[0]
    bars1 = ax1.bar(models, map50, color=colors)
    ax1.set_ylabel('mAP@0.5 (%)')
    ax1.set_title('mAP@0.5 Comparison')
    ax1.set_ylim(0, 100)
    for bar, val in zip(bars1, map50):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # mAP@[0.5:0.95] comparison
    ax2 = axes[1]
    bars2 = ax2.bar(models, map50_95, color=colors)
    ax2.set_ylabel('mAP@[0.5:0.95] (%)')
    ax2.set_title('mAP@[0.5:0.95] Comparison')
    ax2.set_ylim(0, 100)
    for bar, val in zip(bars2, map50_95):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # FPS comparison
    ax3 = axes[2]
    bars3 = ax3.bar(models, fps, color=colors)
    ax3.set_ylabel('FPS')
    ax3.set_title('Speed Comparison (FPS)')
    for bar, val in zip(bars3, fps):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # Rotate x labels if needed
    for ax in axes:
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_accuracy_vs_speed(
    results_list: List[EvaluationResults],
    output_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """
    Create scatter plot of accuracy vs speed trade-off.

    Args:
        results_list: List of EvaluationResults objects.
        output_path: Path to save the plot. If None, uses default.
        show: Whether to display the plot.
    """
    if not results_list:
        print("No results to plot.")
        return

    if output_path is None:
        output_path = RESULTS_DIR / "accuracy_vs_speed.png"

    # Extract data
    models = [r.model_name for r in results_list]
    map50_95 = [r.map50_95 for r in results_list]
    fps = [r.fps for r in results_list]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot with different colors
    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
    scatter = ax.scatter(fps, map50_95, c=colors, s=200, edgecolors='black', linewidth=1.5)

    # Add labels for each point
    for i, model in enumerate(models):
        ax.annotate(
            model,
            (fps[i], map50_95[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=10,
            fontweight='bold'
        )

    ax.set_xlabel('FPS (Frames Per Second)', fontsize=12)
    ax.set_ylabel('mAP@[0.5:0.95] (%)', fontsize=12)
    ax.set_title('Accuracy vs Speed Trade-off', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Accuracy vs speed plot saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

```

```python
#!/usr/bin/env python3
"""
Data Preparation Script

This script prepares the COCO subset dataset for evaluation:
1. Loads COCO annotations from ZIP file
2. Samples a subset of images
3. Extracts images from ZIP
4. Converts annotations to YOLO format
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    ANNOTATIONS_ZIP,
    IMAGES_ZIP,
    COCO_SUBSET_DIR,
    NUM_SAMPLES,
    RANDOM_SEED,
)
from src.data.coco_loader import load_coco_annotations
from src.data.subset_creator import (
    sample_images,
    filter_annotations,
    extract_images,
    save_subset_annotations,
)
from src.data.yolo_converter import convert_to_yolo_format


def main():
    """Run data preparation pipeline."""
    print("=" * 60)
    print("COCO Subset Data Preparation")
    print("=" * 60)

    # Check required files
    if not ANNOTATIONS_ZIP.exists():
        print(f"ERROR: Annotations ZIP not found: {ANNOTATIONS_ZIP}")
        print("Please download annotations_trainval2017.zip and place it in datafiles/")
        sys.exit(1)

    if not IMAGES_ZIP.exists():
        print(f"ERROR: Images ZIP not found: {IMAGES_ZIP}")
        print("Please download val2017.zip and place it in datafiles/")
        sys.exit(1)

    # Load COCO annotations
    print("\n[1/5] Loading COCO annotations...")
    coco = load_coco_annotations()

    # Sample images
    print(f"\n[2/5] Sampling {NUM_SAMPLES} images (seed={RANDOM_SEED})...")
    sampled_images, sampled_ids = sample_images(coco, NUM_SAMPLES, RANDOM_SEED)

    # Filter annotations
    print("\n[3/5] Filtering annotations...")
    sampled_annotations = filter_annotations(coco, sampled_ids)

    # Extract images
    print("\n[4/5] Extracting images...")
    images_dir = COCO_SUBSET_DIR / "images"
    extract_images(sampled_images, IMAGES_ZIP, images_dir)

    # Save subset annotations
    print("\n[5/5] Saving annotations and converting to YOLO format...")
    ann_path = COCO_SUBSET_DIR / "annotations" / "instances_val2017_subset.json"
    save_subset_annotations(coco, sampled_images, sampled_annotations, ann_path)

    # Convert to YOLO format
    labels_dir = COCO_SUBSET_DIR / "labels"
    convert_to_yolo_format(coco, sampled_images, sampled_annotations, labels_dir)

    # Summary
    print("\n" + "=" * 60)
    print("Data Preparation Complete!")
    print("=" * 60)
    print(f"Images:      {images_dir}")
    print(f"Labels:      {labels_dir}")
    print(f"Annotations: {ann_path}")
    print(f"Total images: {len(sampled_images)}")
    print(f"Total annotations: {len(sampled_annotations)}")


if __name__ == "__main__":
    main()

```

```python
#!/usr/bin/env python3
"""
Model Evaluation Script

This script runs evaluation for all object detection models:
- YOLOv8n
- YOLOv10n
- SSD300
- Faster R-CNN

Results are saved to the results directory.
"""

import sys
import argparse
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import RESULTS_DIR
from src.models.base import EvaluationResults
from src.models.yolov8 import YOLOv8Evaluator
from src.models.yolov10 import YOLOv10Evaluator
from src.models.ssd import SSDEvaluator
from src.models.faster_rcnn import FasterRCNNEvaluator
from src.utils.metrics import format_results_table, save_results_json


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate object detection models on COCO subset"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["yolov8", "yolov10", "ssd", "faster_rcnn", "all"],
        default=["all"],
        help="Models to evaluate (default: all)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "evaluation_results.json",
        help="Output JSON file for results"
    )
    return parser.parse_args()


def get_evaluators(model_names: List[str]) -> List:
    """
    Get evaluator instances for specified models.

    Args:
        model_names: List of model names to evaluate.

    Returns:
        List of evaluator instances.
    """
    evaluator_map = {
        "yolov8": YOLOv8Evaluator,
        "yolov10": YOLOv10Evaluator,
        "ssd": SSDEvaluator,
        "faster_rcnn": FasterRCNNEvaluator,
    }

    if "all" in model_names:
        model_names = list(evaluator_map.keys())

    evaluators = []
    for name in model_names:
        if name in evaluator_map:
            evaluators.append(evaluator_map[name]())
        else:
            print(f"WARNING: Unknown model '{name}', skipping.")

    return evaluators


def main():
    """Run model evaluations."""
    args = parse_args()

    print("=" * 60)
    print("Object Detection Model Evaluation")
    print("=" * 60)

    # Get evaluators
    evaluators = get_evaluators(args.models)

    if not evaluators:
        print("No models to evaluate.")
        sys.exit(1)

    print(f"\nModels to evaluate: {[e.model_name for e in evaluators]}")

    # Run evaluations
    results: List[EvaluationResults] = []

    for evaluator in evaluators:
        try:
            result = evaluator.run()
            results.append(result)
        except Exception as e:
            print(f"\nERROR evaluating {evaluator.model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison table
    if results:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(format_results_table(results))

        # Save results
        save_results_json(results, args.output)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

```

```python
#!/usr/bin/env python3
"""
Results Comparison Script

This script loads saved evaluation results and generates:
- Comparison tables
- Visualization charts
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import RESULTS_DIR
from src.models.base import EvaluationResults
from src.utils.metrics import load_results_json, format_results_table
from src.utils.visualization import plot_comparison, plot_accuracy_vs_speed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare and visualize evaluation results"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_DIR / "evaluation_results.json",
        help="Input JSON file with evaluation results"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Output directory for plots"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (just save them)"
    )
    return parser.parse_args()


def dict_to_results(data: dict) -> EvaluationResults:
    """Convert dictionary to EvaluationResults object."""
    return EvaluationResults(
        model_name=data["model_name"],
        map50=data["map50"],
        map50_95=data["map50_95"],
        precision=data.get("precision"),
        recall=data.get("recall"),
        time_per_image_ms=data.get("time_per_image_ms", 0),
        fps=data.get("fps", 0),
        num_images=data.get("num_images", 0),
        total_time_seconds=data.get("total_time_seconds", 0),
    )


def main():
    """Generate comparison visualizations."""
    args = parse_args()

    print("=" * 60)
    print("Results Comparison and Visualization")
    print("=" * 60)

    # Check input file
    if not args.input.exists():
        print(f"ERROR: Results file not found: {args.input}")
        print("Run evaluate_all.py first to generate results.")
        sys.exit(1)

    # Load results
    print(f"\nLoading results from: {args.input}")
    results_data = load_results_json(args.input)

    if not results_data:
        print("No results found in file.")
        sys.exit(1)

    # Convert to EvaluationResults objects
    results = [dict_to_results(r) for r in results_data]
    print(f"Loaded {len(results)} model results")

    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(format_results_table(results))

    # Generate plots
    print("\nGenerating visualizations...")
    show = not args.no_show

    # Bar chart comparison
    plot_comparison(
        results,
        output_path=args.output_dir / "model_comparison.png",
        show=show
    )

    # Accuracy vs speed scatter plot
    plot_accuracy_vs_speed(
        results,
        output_path=args.output_dir / "accuracy_vs_speed.png",
        show=show
    )

    # Print summary insights
    print("\n" + "=" * 60)
    print("INSIGHTS")
    print("=" * 60)

    # Find best models
    best_accuracy = max(results, key=lambda r: r.map50_95)
    fastest = max(results, key=lambda r: r.fps)
    best_map50 = max(results, key=lambda r: r.map50)

    print(f"Highest mAP@[0.5:0.95]: {best_accuracy.model_name} ({best_accuracy.map50_95:.1f}%)")
    print(f"Highest mAP@0.5:        {best_map50.model_name} ({best_map50.map50:.1f}%)")
    print(f"Fastest (FPS):          {fastest.model_name} ({fastest.fps:.1f} FPS)")

    # Speed comparison
    if len(results) > 1:
        slowest = min(results, key=lambda r: r.fps if r.fps > 0 else float('inf'))
        if slowest.fps > 0:
            speedup = fastest.fps / slowest.fps
            print(f"Speed ratio:            {fastest.model_name} is {speedup:.1f}x faster than {slowest.model_name}")

    print("\nDone!")


if __name__ == "__main__":
    main()

```

```pip-requirements
# Object Detection Performance Analysis
# Python >= 3.9 required

# Deep Learning Frameworks
torch>=2.0.0
torchvision>=0.15.0

# YOLO Models
ultralytics>=8.0.0

# COCO Evaluation
pycocotools>=2.0.6

# Data Processing
numpy>=1.23.0
Pillow>=9.0.0

# Visualization
matplotlib>=3.5.0

# Progress Bar
tqdm>=4.64.0

# Configuration
PyYAML>=6.0

```

```markdown
# Object Detection Performance Analysis

This project evaluates and compares the performance of 4 object detection models on a COCO validation subset:

- **YOLOv8n** - Ultralytics YOLOv8 Nano
- **YOLOv10n** - Ultralytics YOLOv10 Nano
- **SSD300** - Single Shot Detector with VGG16 backbone
- **Faster R-CNN** - Faster R-CNN with ResNet50-FPN v2

## Project Structure

```
├── src/
│   ├── config.py                    # Global configuration
│   ├── data/
│   │   ├── coco_loader.py           # Load COCO annotations
│   │   ├── subset_creator.py        # Create dataset subset
│   │   └── yolo_converter.py        # Convert to YOLO format
│   ├── models/
│   │   ├── base.py                  # Base evaluator class
│   │   ├── yolov8/evaluator.py      # YOLOv8 evaluation
│   │   ├── yolov10/evaluator.py     # YOLOv10 evaluation
│   │   ├── ssd/evaluator.py         # SSD evaluation
│   │   └── faster_rcnn/evaluator.py # Faster R-CNN evaluation
│   └── utils/
│       ├── device.py                # Device detection
│       ├── metrics.py               # Metrics utilities
│       └── visualization.py         # Plotting functions
├── scripts/
│   ├── prepare_data.py              # Data preparation
│   ├── evaluate_all.py              # Run all evaluations
│   └── compare_results.py           # Generate comparisons
├── datafiles/                       # Source data (ZIP files)
├── coco_subset/                     # Prepared dataset
├── results/                         # Evaluation results
└── requirements.txt
```

## Installation

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download COCO data and place in `datafiles/`:
   - `annotations_trainval2017.zip`
   - `val2017.zip`

4. Download model weights:
   - `yolov8n.pt` (place in project root)
   - `yolov10n.pt` (place in project root)

## Usage

### 1. Prepare Data (one-time)

```bash
python scripts/prepare_data.py
```

This extracts a subset of COCO images and converts annotations to YOLO format.

### 2. Run Evaluations

Run all models:
```bash
python scripts/evaluate_all.py
```

Run specific models:
```bash
python scripts/evaluate_all.py --models yolov8 yolov10
python scripts/evaluate_all.py --models ssd faster_rcnn
```

### 3. Compare Results

```bash
python scripts/compare_results.py
```

This generates comparison tables and visualization plots in the `results/` directory.

## Metrics

The evaluation measures:

| Metric | Description |
|--------|-------------|
| mAP@0.5 | Mean Average Precision at IoU 0.5 |
| mAP@[0.5:0.95] | Mean AP averaged over IoU 0.5 to 0.95 |
| Precision | Detection precision |
| Recall | Detection recall |
| FPS | Frames per second (inference speed) |
| ms/image | Milliseconds per image |

## Configuration

Edit `src/config.py` to modify:

- Dataset paths
- Number of samples
- Score thresholds
- Batch sizes

## Hardware

The code automatically detects available hardware:
- **CUDA** - NVIDIA GPU
- **MPS** - Apple Silicon GPU
- **CPU** - Fallback

## License

This project is for educational and research purposes.

```

I've completed all the tasks. Here's a summary of the modular project structure I created:

## Created Files

### Core Package (`src/`)
- `config.py` - Centralized configuration (paths, thresholds, COCO classes)
- `data/` - Data preparation modules
  - `coco_loader.py` - Load COCO annotations from ZIP
  - `subset_creator.py` - Sample and extract subset
  - `yolo_converter.py` - Convert to YOLO label format
- `models/` - Model evaluators
  - `base.py` - Abstract base class with `EvaluationResults`
  - `yolov8/evaluator.py` - YOLOv8 validation
  - `yolov10/evaluator.py` - YOLOv10 validation
  - `ssd/evaluator.py` - SSD300 with COCOeval
  - `faster_rcnn/evaluator.py` - Faster R-CNN with COCOeval
- `utils/` - Shared utilities
  - `device.py` - Auto-detect CUDA/MPS/CPU
  - `metrics.py` - Format and save results
  - `visualization.py` - Comparison plots

### Scripts (`scripts/`)
- `prepare_data.py` - One-time dataset setup
- `evaluate_all.py` - Run model evaluations
- `compare_results.py` - Generate comparison charts

### Documentation
- `requirements.txt` - All dependencies
- `README.md` - Usage instructions

## Usage

```bash
# 1. Prepare data (if not already done)
python scripts/prepare_data.py

# 2. Evaluate all models
python scripts/evaluate_all.py

# 3. Generate comparison visualizations
python scripts/compare_results.py
```

Or evaluate specific models:
```bash
python scripts/evaluate_all.py --models yolov8 yolov10
```

```shellscript
# 1. Prepare data (if not already done)
python scripts/prepare_data.py

# 2. Evaluate all models
python scripts/evaluate_all.py

# 3. Generate comparison visualizations
python scripts/compare_results.py
```

```shellscript
python scripts/evaluate_all.py --models yolov8 yolov10
```

---

**User**

Remove all the old and unused code, create virtual environment and install all requirements. 

---

**Cursor**

Done! Here's what I completed:

## Removed Old Files
- ✅ `thesis.ipynb` - Converted to modular code
- ✅ `code.ipynb` - Sample notebook (unused)
- ✅ `models/__init__.py` - Old empty file in weights folder

## Virtual Environment
- ✅ Created fresh `.venv` with Python 3.12
- ✅ Upgraded pip to v25.3
- ✅ Installed all 37 packages from `requirements.txt`

## Installed Packages
Key packages installed:
- `torch` 2.9.1 (with MPS support for Apple Silicon)
- `torchvision` 0.24.1
- `ultralytics` 8.3.232
- `pycocotools` 2.0.10
- `matplotlib` 3.10.7
- `numpy` 2.2.6
- `tqdm` 4.67.1

## Ready to Use

Activate the environment and run:
```bash
source .venv/bin/activate

# Evaluate all models
python scripts/evaluate_all.py

# Or evaluate specific models
python scripts/evaluate_all.py --models yolov8 yolov10

# Generate comparison charts
python scripts/compare_results.py
```

```shellscript
source .venv/bin/activate

# Evaluate all models
python scripts/evaluate_all.py

# Or evaluate specific models
python scripts/evaluate_all.py --models yolov8 yolov10

# Generate comparison charts
python scripts/compare_results.py
```

---

**User**

You are an experienced ML engineer and Python refactorer working inside a benchmark project.

Now the evaluations are already implemented.

Your task



Extend the codebase so that after evaluating each model, the project:

Collects metrics programmatically (not just prints them).

Stores them in a central structure/file.

Generates all plots/tables/figures I need for a thesis Results & Discussion chapter.

---

**Cursor**

Let me first understand the current state of the metrics collection and visualization code.

Now I have a full picture. Let me create a plan to extend this for thesis-quality output.

---

**User**

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

---

**Cursor**

I'll implement the plan step by step. Starting with extending the `EvaluationResults` dataclass.

```python
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

```

```python
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

```

```python
"""YOLOv10 model evaluator."""

from typing import Optional

import torch
from ultralytics import YOLO

from src.config import YOLOV10_WEIGHTS, YOLO_CONFIG_FILE, NUM_SAMPLES
from src.models.base import (
    BaseEvaluator,
    EvaluationResults,
    TimingBreakdown,
    ModelInfo,
    APBySize,
)


class YOLOv10Evaluator(BaseEvaluator):
    """Evaluator for YOLOv10 object detection model."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize YOLOv10 evaluator.

        Args:
            weights_path: Path to YOLOv10 weights file. Defaults to config value.
            config_path: Path to YOLO config YAML. Defaults to config value.
            device: PyTorch device to use.
        """
        super().__init__(device)
        self.weights_path = str(weights_path or YOLOV10_WEIGHTS)
        self.config_path = str(config_path or YOLO_CONFIG_FILE)

    @property
    def model_name(self) -> str:
        return "YOLOv10n"

    def load_model(self) -> None:
        """Load YOLOv10 model."""
        print(f"Loading YOLOv10 from: {self.weights_path}")
        self.model = YOLO(self.weights_path)

    def get_model_info(self) -> ModelInfo:
        """Get YOLOv10 model information."""
        if self.model is None:
            return ModelInfo(name=self.model_name)

        # Get model info from ultralytics
        try:
            info = self.model.info(verbose=False)
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
            parameters=2_299_264,  # YOLOv10n default
            gflops=6.7,
            input_size=640,
        )

    def evaluate(self) -> EvaluationResults:
        """
        Run YOLOv10 validation on COCO subset.

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
                map75 = metrics.box.maps[5] * 100
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

```

```python
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

```

```python
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
    FasterRCNN_ResNet50_FPN_V2_Weights
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

    def __init__(
        self,
        images_dir: Optional[str] = None,
        ann_file: Optional[str] = None,
        score_threshold: float = SCORE_THRESHOLD,
        max_detections: int = MAX_DETECTIONS,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Faster R-CNN evaluator.

        Args:
            images_dir: Path to images directory.
            ann_file: Path to COCO annotation file.
            score_threshold: Minimum score for detections.
            max_detections: Maximum detections per image.
            device: PyTorch device to use.
        """
        super().__init__(device)
        self.images_dir = Path(images_dir or COCO_SUBSET_IMAGES)
        self.ann_file = str(ann_file or COCO_SUBSET_ANN_FILE)
        self.score_threshold = score_threshold
        self.max_detections = max_detections

    @property
    def model_name(self) -> str:
        return "Faster R-CNN"

    def load_model(self) -> None:
        """Load Faster R-CNN model with COCO pretrained weights."""
        print("Loading Faster R-CNN (ResNet50-FPN v2, COCO pretrained)...")
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        self.model.to(self.device)
        self.model.eval()

    def get_model_info(self) -> ModelInfo:
        """Get Faster R-CNN model information."""
        return ModelInfo(
            name=self.model_name,
            parameters=43_712_278,  # Faster R-CNN ResNet50-FPN v2
            gflops=134.4,  # Approximate GFLOPs
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

```

```python
"""Results management and thesis figure generation."""

from .results_manager import ResultsManager
from .thesis_figures import ThesisFigureGenerator
from .latex_tables import LaTeXTableGenerator

__all__ = ["ResultsManager", "ThesisFigureGenerator", "LaTeXTableGenerator"]

```

```python
"""Central results storage and I/O management."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import RESULTS_DIR, NUM_SAMPLES
from src.models.base import EvaluationResults


class ResultsManager:
    """
    Central manager for evaluation results.

    Handles saving, loading, and exporting results in multiple formats
    with full metadata for reproducibility.
    """

    def __init__(self, results_dir: Optional[Path] = None):
        """
        Initialize the results manager.

        Args:
            results_dir: Directory for storing results. Defaults to config value.
        """
        self.results_dir = Path(results_dir or RESULTS_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.figures_dir = self.results_dir / "figures"
        self.tables_dir = self.results_dir / "tables"
        self.figures_dir.mkdir(exist_ok=True)
        self.tables_dir.mkdir(exist_ok=True)

        # File paths
        self.json_path = self.results_dir / "evaluation_results.json"
        self.csv_path = self.results_dir / "evaluation_results.csv"

        # In-memory results storage
        self._results: List[EvaluationResults] = []
        self._metadata: Dict[str, Any] = {}

    @property
    def results(self) -> List[EvaluationResults]:
        """Get current results list."""
        return self._results

    def add_result(self, result: EvaluationResults) -> None:
        """
        Add a single evaluation result.

        Args:
            result: EvaluationResults to add.
        """
        self._results.append(result)

    def add_results(self, results: List[EvaluationResults]) -> None:
        """
        Add multiple evaluation results.

        Args:
            results: List of EvaluationResults to add.
        """
        self._results.extend(results)

    def clear(self) -> None:
        """Clear all stored results."""
        self._results = []
        self._metadata = {}

    def set_metadata(
        self,
        dataset_name: str = "COCO val2017 subset",
        num_images: int = NUM_SAMPLES,
        **kwargs
    ) -> None:
        """
        Set evaluation metadata.

        Args:
            dataset_name: Name of the dataset used.
            num_images: Number of images evaluated.
            **kwargs: Additional metadata fields.
        """
        self._metadata = {
            "dataset_name": dataset_name,
            "num_images": num_images,
            "evaluation_date": datetime.now().isoformat(),
            **kwargs
        }

    def save_json(self, path: Optional[Path] = None) -> Path:
        """
        Save results to JSON with full metadata.

        Args:
            path: Output path. Defaults to standard location.

        Returns:
            Path to saved file.
        """
        path = Path(path or self.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": self._metadata,
            "results": [r.to_dict() for r in self._results],
            "summary": self._generate_summary(),
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to JSON: {path}")
        return path

    def save_csv(self, path: Optional[Path] = None) -> Path:
        """
        Save results to CSV for easy import into LaTeX/Excel.

        Args:
            path: Output path. Defaults to standard location.

        Returns:
            Path to saved file.
        """
        path = Path(path or self.csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self._results:
            print("No results to save.")
            return path

        # Get all fields from flat dict
        flat_results = [r.to_flat_dict() for r in self._results]
        fieldnames = list(flat_results[0].keys())

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_results)

        print(f"Results saved to CSV: {path}")
        return path

    def save_all(self) -> Dict[str, Path]:
        """
        Save results in all formats.

        Returns:
            Dictionary mapping format name to saved path.
        """
        return {
            "json": self.save_json(),
            "csv": self.save_csv(),
        }

    def load_json(self, path: Optional[Path] = None) -> List[EvaluationResults]:
        """
        Load results from JSON file.

        Args:
            path: Input path. Defaults to standard location.

        Returns:
            List of loaded EvaluationResults.
        """
        path = Path(path or self.json_path)

        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        self._metadata = data.get("metadata", {})
        self._results = [
            EvaluationResults.from_dict(r)
            for r in data.get("results", [])
        ]

        print(f"Loaded {len(self._results)} results from: {path}")
        return self._results

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        if not self._results:
            return {}

        # Find best performers
        best_accuracy = max(self._results, key=lambda r: r.map50_95)
        best_map50 = max(self._results, key=lambda r: r.map50)
        fastest = max(self._results, key=lambda r: r.fps)
        slowest = min(self._results, key=lambda r: r.fps if r.fps > 0 else float('inf'))

        summary = {
            "num_models": len(self._results),
            "best_map50_95": {
                "model": best_accuracy.model_name,
                "value": best_accuracy.map50_95,
            },
            "best_map50": {
                "model": best_map50.model_name,
                "value": best_map50.map50,
            },
            "fastest": {
                "model": fastest.model_name,
                "fps": fastest.fps,
            },
            "slowest": {
                "model": slowest.model_name,
                "fps": slowest.fps,
            },
        }

        if slowest.fps > 0:
            summary["speed_ratio"] = fastest.fps / slowest.fps

        return summary

    def get_comparison_data(self) -> Dict[str, List[Any]]:
        """
        Get data structured for comparison plots.

        Returns:
            Dictionary with model names and metric lists.
        """
        return {
            "models": [r.model_name for r in self._results],
            "map50": [r.map50 for r in self._results],
            "map50_95": [r.map50_95 for r in self._results],
            "map75": [r.map75 for r in self._results],
            "precision": [r.precision for r in self._results],
            "recall": [r.recall for r in self._results],
            "f1_score": [r.f1_score for r in self._results],
            "fps": [r.fps for r in self._results],
            "time_per_image_ms": [r.time_per_image_ms for r in self._results],
        }

    def print_summary(self) -> None:
        """Print a formatted summary of results."""
        if not self._results:
            print("No results to summarize.")
            return

        print("\n" + "=" * 70)
        print("EVALUATION RESULTS SUMMARY")
        print("=" * 70)

        # Header
        header = (
            f"{'Model':<15} {'mAP@0.5':>10} {'mAP@0.5:0.95':>14} "
            f"{'Precision':>10} {'Recall':>10} {'F1':>8} {'FPS':>8}"
        )
        print("-" * 70)
        print(header)
        print("-" * 70)

        for r in self._results:
            prec = f"{r.precision:.1f}%" if r.precision else "N/A"
            rec = f"{r.recall:.1f}%" if r.recall else "N/A"
            f1 = f"{r.f1_score:.1f}%" if r.f1_score else "N/A"
            print(
                f"{r.model_name:<15} {r.map50:>9.1f}% {r.map50_95:>13.1f}% "
                f"{prec:>10} {rec:>10} {f1:>8} {r.fps:>7.1f}"
            )

        print("-" * 70)

        # Summary insights
        summary = self._generate_summary()
        print(f"\nBest mAP@[0.5:0.95]: {summary['best_map50_95']['model']} "
              f"({summary['best_map50_95']['value']:.1f}%)")
        print(f"Fastest model:       {summary['fastest']['model']} "
              f"({summary['fastest']['fps']:.1f} FPS)")
        if 'speed_ratio' in summary:
            print(f"Speed ratio:         {summary['speed_ratio']:.1f}x")
        print("=" * 70)

```

```python
"""Publication-ready figure generation for thesis."""

from pathlib import Path
from typing import List, Optional, Literal

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.models.base import EvaluationResults
from src.config import RESULTS_DIR

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Consistent color scheme for models
MODEL_COLORS = {
    'YOLOv8n': '#2ecc71',      # Green
    'YOLOv10n': '#3498db',     # Blue
    'SSD300': '#e74c3c',       # Red
    'Faster R-CNN': '#9b59b6', # Purple
}


def get_color(model_name: str) -> str:
    """Get consistent color for a model."""
    return MODEL_COLORS.get(model_name, '#95a5a6')


class ThesisFigureGenerator:
    """Generate publication-ready figures for thesis."""

    def __init__(
        self,
        results: List[EvaluationResults],
        output_dir: Optional[Path] = None,
        format: Literal['pdf', 'png', 'svg'] = 'pdf'
    ):
        """
        Initialize the figure generator.

        Args:
            results: List of evaluation results.
            output_dir: Directory to save figures.
            format: Output format for figures.
        """
        self.results = results
        self.output_dir = Path(output_dir or RESULTS_DIR / "figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format

    def _save_figure(self, fig: plt.Figure, name: str) -> Path:
        """Save figure to file."""
        path = self.output_dir / f"{name}.{self.format}"
        fig.savefig(path, format=self.format)
        plt.close(fig)
        print(f"Saved: {path}")
        return path

    def plot_map_comparison(self) -> Path:
        """
        Create grouped bar chart comparing mAP@0.5 and mAP@0.5:0.95.

        Returns:
            Path to saved figure.
        """
        models = [r.model_name for r in self.results]
        map50 = [r.map50 for r in self.results]
        map50_95 = [r.map50_95 for r in self.results]

        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))

        bars1 = ax.bar(x - width/2, map50, width, label='mAP@0.5',
                       color=[get_color(m) for m in models], alpha=0.9)
        bars2 = ax.bar(x + width/2, map50_95, width, label='mAP@[0.5:0.95]',
                       color=[get_color(m) for m in models], alpha=0.6,
                       hatch='//')

        ax.set_xlabel('Model')
        ax.set_ylabel('Average Precision (%)')
        ax.set_title('Object Detection Accuracy Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        return self._save_figure(fig, 'map_comparison')

    def plot_pareto_frontier(self) -> Path:
        """
        Create accuracy vs speed scatter plot with Pareto frontier.

        Returns:
            Path to saved figure.
        """
        models = [r.model_name for r in self.results]
        map50_95 = [r.map50_95 for r in self.results]
        fps = [r.fps for r in self.results]

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot each model
        for i, (model, acc, speed) in enumerate(zip(models, map50_95, fps)):
            ax.scatter(speed, acc, s=200, c=get_color(model),
                       edgecolors='black', linewidth=1.5, zorder=5,
                       label=model)

        # Find and plot Pareto frontier
        points = list(zip(fps, map50_95))
        pareto_points = self._get_pareto_frontier(points)
        if len(pareto_points) > 1:
            pareto_x, pareto_y = zip(*sorted(pareto_points))
            ax.plot(pareto_x, pareto_y, 'k--', alpha=0.5, linewidth=1.5,
                    label='Pareto Frontier', zorder=1)

        ax.set_xlabel('Inference Speed (FPS)')
        ax.set_ylabel('mAP@[0.5:0.95] (%)')
        ax.set_title('Accuracy vs Speed Trade-off')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Set reasonable axis limits
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=max(map50_95) * 1.15)

        plt.tight_layout()
        return self._save_figure(fig, 'pareto_frontier')

    def _get_pareto_frontier(self, points: List[tuple]) -> List[tuple]:
        """Find Pareto-optimal points (maximize both x and y)."""
        sorted_points = sorted(points, key=lambda p: p[0], reverse=True)
        pareto = []
        max_y = float('-inf')

        for x, y in sorted_points:
            if y >= max_y:
                pareto.append((x, y))
                max_y = y

        return pareto

    def plot_radar_chart(self) -> Path:
        """
        Create radar/spider chart for multi-metric comparison.

        Returns:
            Path to saved figure.
        """
        # Metrics to include (normalized to 0-100 scale)
        metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1 Score']
        num_metrics = len(metrics)

        # Extract and normalize data
        data = []
        for r in self.results:
            values = [
                r.map50,
                r.map50_95,
                r.precision if r.precision else 0,
                r.recall if r.recall else 0,
                r.f1_score if r.f1_score else 0,
            ]
            data.append(values)

        # Compute angles for radar chart
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for i, r in enumerate(self.results):
            values = data[i] + data[i][:1]  # Complete the loop
            ax.plot(angles, values, 'o-', linewidth=2, label=r.model_name,
                    color=get_color(r.model_name))
            ax.fill(angles, values, alpha=0.15, color=get_color(r.model_name))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 100)
        ax.set_title('Multi-Metric Model Comparison', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        plt.tight_layout()
        return self._save_figure(fig, 'radar_chart')

    def plot_inference_breakdown(self) -> Path:
        """
        Create stacked bar chart showing inference time breakdown.

        Returns:
            Path to saved figure.
        """
        models = [r.model_name for r in self.results]

        # Extract timing breakdown
        preprocess = []
        inference = []
        postprocess = []

        for r in self.results:
            if r.timing_breakdown:
                preprocess.append(r.timing_breakdown.preprocess_ms)
                inference.append(r.timing_breakdown.inference_ms)
                postprocess.append(r.timing_breakdown.postprocess_ms)
            else:
                # Fallback if no breakdown available
                preprocess.append(0)
                inference.append(r.time_per_image_ms)
                postprocess.append(0)

        fig, ax = plt.subplots(figsize=(8, 5))

        x = np.arange(len(models))
        width = 0.6

        # Stacked bars
        ax.bar(x, preprocess, width, label='Preprocessing', color='#3498db')
        ax.bar(x, inference, width, bottom=preprocess, label='Inference', color='#2ecc71')
        ax.bar(x, postprocess, width,
               bottom=np.array(preprocess) + np.array(inference),
               label='Postprocessing', color='#e74c3c')

        ax.set_xlabel('Model')
        ax.set_ylabel('Time per Image (ms)')
        ax.set_title('Inference Time Breakdown')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add total time labels
        totals = np.array(preprocess) + np.array(inference) + np.array(postprocess)
        for i, total in enumerate(totals):
            ax.annotate(f'{total:.1f}ms',
                        xy=(i, total),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        return self._save_figure(fig, 'inference_breakdown')

    def plot_ap_by_size(self) -> Path:
        """
        Create grouped bar chart for AP by object size.

        Returns:
            Path to saved figure.
        """
        models = [r.model_name for r in self.results]

        # Extract AP by size
        ap_small = []
        ap_medium = []
        ap_large = []

        for r in self.results:
            if r.ap_by_size:
                ap_small.append(r.ap_by_size.small)
                ap_medium.append(r.ap_by_size.medium)
                ap_large.append(r.ap_by_size.large)
            else:
                ap_small.append(0)
                ap_medium.append(0)
                ap_large.append(0)

        x = np.arange(len(models))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 5))

        bars1 = ax.bar(x - width, ap_small, width, label='Small (area < 32²)',
                       color='#3498db')
        bars2 = ax.bar(x, ap_medium, width, label='Medium (32² < area < 96²)',
                       color='#2ecc71')
        bars3 = ax.bar(x + width, ap_large, width, label='Large (area > 96²)',
                       color='#e74c3c')

        ax.set_xlabel('Model')
        ax.set_ylabel('Average Precision (%)')
        ax.set_title('Detection Performance by Object Size')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.1f}',
                                xy=(bar.get_x() + bar.get_width()/2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        return self._save_figure(fig, 'ap_by_size')

    def generate_all(self) -> List[Path]:
        """
        Generate all thesis figures.

        Returns:
            List of paths to generated figures.
        """
        print(f"\nGenerating thesis figures in: {self.output_dir}")
        print("-" * 50)

        paths = [
            self.plot_map_comparison(),
            self.plot_pareto_frontier(),
            self.plot_radar_chart(),
            self.plot_inference_breakdown(),
            self.plot_ap_by_size(),
        ]

        print("-" * 50)
        print(f"Generated {len(paths)} figures")

        return paths

```

```python
"""LaTeX table generation for thesis."""

from pathlib import Path
from typing import List, Optional

from src.models.base import EvaluationResults
from src.config import RESULTS_DIR


class LaTeXTableGenerator:
    """Generate LaTeX tables for thesis."""

    def __init__(
        self,
        results: List[EvaluationResults],
        output_dir: Optional[Path] = None
    ):
        """
        Initialize the LaTeX table generator.

        Args:
            results: List of evaluation results.
            output_dir: Directory to save tables.
        """
        self.results = results
        self.output_dir = Path(output_dir or RESULTS_DIR / "tables")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_table(self, content: str, name: str) -> Path:
        """Save LaTeX table to file."""
        path = self.output_dir / f"{name}.tex"
        with open(path, 'w') as f:
            f.write(content)
        print(f"Saved: {path}")
        return path

    def _format_value(self, value: Optional[float], precision: int = 1) -> str:
        """Format a value for LaTeX, handling None."""
        if value is None:
            return "---"
        return f"{value:.{precision}f}"

    def _bold_best(self, values: List[Optional[float]], maximize: bool = True) -> List[str]:
        """
        Format values with best one in bold.

        Args:
            values: List of values (may contain None).
            maximize: If True, bold the maximum; else bold the minimum.

        Returns:
            List of formatted strings with best in bold.
        """
        formatted = []
        valid_values = [v for v in values if v is not None]

        if not valid_values:
            return [self._format_value(v) for v in values]

        best = max(valid_values) if maximize else min(valid_values)

        for v in values:
            if v is None:
                formatted.append("---")
            elif abs(v - best) < 0.01:  # Account for float comparison
                formatted.append(f"\\textbf{{{v:.1f}}}")
            else:
                formatted.append(f"{v:.1f}")

        return formatted

    def generate_main_results_table(self) -> Path:
        """
        Generate the main results comparison table.

        Returns:
            Path to saved .tex file.
        """
        # Extract metrics
        models = [r.model_name for r in self.results]
        map50 = [r.map50 for r in self.results]
        map50_95 = [r.map50_95 for r in self.results]
        precision = [r.precision for r in self.results]
        recall = [r.recall for r in self.results]
        f1 = [r.f1_score for r in self.results]
        fps = [r.fps for r in self.results]

        # Bold best values
        map50_fmt = self._bold_best(map50)
        map50_95_fmt = self._bold_best(map50_95)
        prec_fmt = self._bold_best(precision)
        rec_fmt = self._bold_best(recall)
        f1_fmt = self._bold_best(f1)
        fps_fmt = self._bold_best(fps)

        # Build LaTeX table
        lines = [
            "% Main Results Table - Object Detection Performance Comparison",
            "% Include with: \\input{tables/main_results.tex}",
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Object Detection Model Performance Comparison on COCO Validation Subset}",
            "\\label{tab:main_results}",
            "\\begin{tabular}{lcccccc}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{mAP@0.5} & \\textbf{mAP@0.5:0.95} & "
            "\\textbf{Precision} & \\textbf{Recall} & \\textbf{F1 Score} & \\textbf{FPS} \\\\",
            "\\midrule",
        ]

        for i, model in enumerate(models):
            row = (
                f"{model} & {map50_fmt[i]}\\% & {map50_95_fmt[i]}\\% & "
                f"{prec_fmt[i]}\\% & {rec_fmt[i]}\\% & {f1_fmt[i]}\\% & {fps_fmt[i]} \\\\"
            )
            lines.append(row)

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        content = "\n".join(lines)
        return self._save_table(content, "main_results")

    def generate_speed_table(self) -> Path:
        """
        Generate timing breakdown table.

        Returns:
            Path to saved .tex file.
        """
        models = [r.model_name for r in self.results]

        # Extract timing data
        preprocess = []
        inference = []
        postprocess = []
        total = []
        fps_list = []

        for r in self.results:
            if r.timing_breakdown:
                preprocess.append(r.timing_breakdown.preprocess_ms)
                inference.append(r.timing_breakdown.inference_ms)
                postprocess.append(r.timing_breakdown.postprocess_ms)
                total.append(r.timing_breakdown.total_ms)
            else:
                preprocess.append(None)
                inference.append(r.time_per_image_ms)
                postprocess.append(None)
                total.append(r.time_per_image_ms)
            fps_list.append(r.fps)

        # Bold best (minimize for times, maximize for FPS)
        pre_fmt = self._bold_best(preprocess, maximize=False)
        inf_fmt = self._bold_best(inference, maximize=False)
        post_fmt = self._bold_best(postprocess, maximize=False)
        total_fmt = self._bold_best(total, maximize=False)
        fps_fmt = self._bold_best(fps_list, maximize=True)

        lines = [
            "% Speed Comparison Table - Inference Time Breakdown",
            "% Include with: \\input{tables/speed_comparison.tex}",
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Inference Speed Comparison (milliseconds per image)}",
            "\\label{tab:speed_comparison}",
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Preprocess} & \\textbf{Inference} & "
            "\\textbf{Postprocess} & \\textbf{Total} & \\textbf{FPS} \\\\",
            "\\midrule",
        ]

        for i, model in enumerate(models):
            row = (
                f"{model} & {pre_fmt[i]} & {inf_fmt[i]} & "
                f"{post_fmt[i]} & {total_fmt[i]} & {fps_fmt[i]} \\\\"
            )
            lines.append(row)

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        content = "\n".join(lines)
        return self._save_table(content, "speed_comparison")

    def generate_ap_by_size_table(self) -> Path:
        """
        Generate AP by object size table.

        Returns:
            Path to saved .tex file.
        """
        models = [r.model_name for r in self.results]

        # Extract AP by size
        ap_small = []
        ap_medium = []
        ap_large = []

        for r in self.results:
            if r.ap_by_size:
                ap_small.append(r.ap_by_size.small)
                ap_medium.append(r.ap_by_size.medium)
                ap_large.append(r.ap_by_size.large)
            else:
                ap_small.append(None)
                ap_medium.append(None)
                ap_large.append(None)

        # Bold best values
        small_fmt = self._bold_best(ap_small)
        medium_fmt = self._bold_best(ap_medium)
        large_fmt = self._bold_best(ap_large)

        lines = [
            "% AP by Object Size Table",
            "% Include with: \\input{tables/ap_by_size.tex}",
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Average Precision by Object Size (COCO size categories)}",
            "\\label{tab:ap_by_size}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{AP\\textsubscript{small}} & "
            "\\textbf{AP\\textsubscript{medium}} & \\textbf{AP\\textsubscript{large}} \\\\",
            " & (area $<$ 32\\textsuperscript{2}) & "
            "(32\\textsuperscript{2} $<$ area $<$ 96\\textsuperscript{2}) & "
            "(area $>$ 96\\textsuperscript{2}) \\\\",
            "\\midrule",
        ]

        for i, model in enumerate(models):
            row = f"{model} & {small_fmt[i]}\\% & {medium_fmt[i]}\\% & {large_fmt[i]}\\% \\\\"
            lines.append(row)

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        content = "\n".join(lines)
        return self._save_table(content, "ap_by_size")

    def generate_model_info_table(self) -> Path:
        """
        Generate model architecture information table.

        Returns:
            Path to saved .tex file.
        """
        models = [r.model_name for r in self.results]

        # Extract model info
        params = []
        gflops = []
        input_size = []

        for r in self.results:
            if r.model_info:
                params.append(r.model_info.parameters / 1e6)  # Convert to millions
                gflops.append(r.model_info.gflops)
                input_size.append(r.model_info.input_size)
            else:
                params.append(None)
                gflops.append(None)
                input_size.append(None)

        # Bold best (minimize for params and GFLOPs)
        params_fmt = self._bold_best(params, maximize=False)
        gflops_fmt = self._bold_best(gflops, maximize=False)

        lines = [
            "% Model Architecture Information Table",
            "% Include with: \\input{tables/model_info.tex}",
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Model Architecture Comparison}",
            "\\label{tab:model_info}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Parameters (M)} & \\textbf{GFLOPs} & "
            "\\textbf{Input Size} \\\\",
            "\\midrule",
        ]

        for i, model in enumerate(models):
            size = f"{input_size[i]}" if input_size[i] else "---"
            row = f"{model} & {params_fmt[i]} & {gflops_fmt[i]} & {size} \\\\"
            lines.append(row)

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        content = "\n".join(lines)
        return self._save_table(content, "model_info")

    def generate_all(self) -> List[Path]:
        """
        Generate all LaTeX tables.

        Returns:
            List of paths to generated tables.
        """
        print(f"\nGenerating LaTeX tables in: {self.output_dir}")
        print("-" * 50)

        paths = [
            self.generate_main_results_table(),
            self.generate_speed_table(),
            self.generate_ap_by_size_table(),
            self.generate_model_info_table(),
        ]

        print("-" * 50)
        print(f"Generated {len(paths)} tables")

        return paths

```

```python
#!/usr/bin/env python3
"""
Thesis Figure and Table Generator

This script generates all publication-ready figures and LaTeX tables
for the Results & Discussion chapter of a thesis.

Usage:
    python scripts/generate_thesis_figures.py
    python scripts/generate_thesis_figures.py --input results/evaluation_results.json
    python scripts/generate_thesis_figures.py --format pdf --no-show
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import RESULTS_DIR
from src.results.results_manager import ResultsManager
from src.results.thesis_figures import ThesisFigureGenerator
from src.results.latex_tables import LaTeXTableGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate thesis figures and LaTeX tables from evaluation results"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_DIR / "evaluation_results.json",
        help="Input JSON file with evaluation results"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Output directory for figures and tables"
    )
    parser.add_argument(
        "--format",
        choices=["pdf", "png", "svg"],
        default="pdf",
        help="Output format for figures (default: pdf)"
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Generate only figures, skip tables"
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Generate only tables, skip figures"
    )
    return parser.parse_args()


def main():
    """Generate all thesis outputs."""
    args = parse_args()

    print("=" * 70)
    print("THESIS FIGURE AND TABLE GENERATOR")
    print("=" * 70)

    # Check input file
    if not args.input.exists():
        print(f"ERROR: Results file not found: {args.input}")
        print("\nTo generate results, run:")
        print("  python scripts/evaluate_all.py")
        sys.exit(1)

    # Load results
    print(f"\nLoading results from: {args.input}")
    manager = ResultsManager(args.output_dir)
    results = manager.load_json(args.input)

    if not results:
        print("ERROR: No results found in file.")
        sys.exit(1)

    print(f"Loaded {len(results)} model evaluations")

    # Print summary
    manager.print_summary()

    # Generate outputs
    all_paths = []

    if not args.tables_only:
        print("\n" + "=" * 70)
        print("GENERATING FIGURES")
        print("=" * 70)

        fig_gen = ThesisFigureGenerator(
            results=results,
            output_dir=args.output_dir / "figures",
            format=args.format
        )
        figure_paths = fig_gen.generate_all()
        all_paths.extend(figure_paths)

    if not args.figures_only:
        print("\n" + "=" * 70)
        print("GENERATING LATEX TABLES")
        print("=" * 70)

        table_gen = LaTeXTableGenerator(
            results=results,
            output_dir=args.output_dir / "tables"
        )
        table_paths = table_gen.generate_all()
        all_paths.extend(table_paths)

    # Summary
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nGenerated {len(all_paths)} files:")

    figures_dir = args.output_dir / "figures"
    tables_dir = args.output_dir / "tables"

    if figures_dir.exists():
        fig_files = list(figures_dir.glob(f"*.{args.format}"))
        if fig_files:
            print(f"\nFigures ({args.output_dir / 'figures'}):")
            for f in sorted(fig_files):
                print(f"  - {f.name}")

    if tables_dir.exists():
        tex_files = list(tables_dir.glob("*.tex"))
        if tex_files:
            print(f"\nLaTeX Tables ({args.output_dir / 'tables'}):")
            for f in sorted(tex_files):
                print(f"  - {f.name}")

    print("\n" + "-" * 70)
    print("Usage in LaTeX:")
    print("  \\input{tables/main_results.tex}")
    print("  \\includegraphics[width=\\textwidth]{figures/map_comparison.pdf}")
    print("-" * 70)


if __name__ == "__main__":
    main()

```

```python
#!/usr/bin/env python3
"""
Model Evaluation Script

This script runs evaluation for all object detection models:
- YOLOv8n
- YOLOv10n
- SSD300
- Faster R-CNN

Results are saved to the results directory in JSON and CSV formats.
After evaluation, run generate_thesis_figures.py to create plots and tables.
"""

import sys
import argparse
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import RESULTS_DIR
from src.models.base import EvaluationResults
from src.models.yolov8 import YOLOv8Evaluator
from src.models.yolov10 import YOLOv10Evaluator
from src.models.ssd import SSDEvaluator
from src.models.faster_rcnn import FasterRCNNEvaluator
from src.results import ResultsManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate object detection models on COCO subset"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["yolov8", "yolov10", "ssd", "faster_rcnn", "all"],
        default=["all"],
        help="Models to evaluate (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Output directory for results"
    )
    parser.add_argument(
        "--generate-figures",
        action="store_true",
        help="Automatically generate thesis figures after evaluation"
    )
    return parser.parse_args()


def get_evaluators(model_names: List[str]) -> List:
    """
    Get evaluator instances for specified models.

    Args:
        model_names: List of model names to evaluate.

    Returns:
        List of evaluator instances.
    """
    evaluator_map = {
        "yolov8": YOLOv8Evaluator,
        "yolov10": YOLOv10Evaluator,
        "ssd": SSDEvaluator,
        "faster_rcnn": FasterRCNNEvaluator,
    }

    if "all" in model_names:
        model_names = list(evaluator_map.keys())

    evaluators = []
    for name in model_names:
        if name in evaluator_map:
            evaluators.append(evaluator_map[name]())
        else:
            print(f"WARNING: Unknown model '{name}', skipping.")

    return evaluators


def main():
    """Run model evaluations."""
    args = parse_args()

    print("=" * 70)
    print("OBJECT DETECTION MODEL EVALUATION")
    print("=" * 70)

    # Initialize results manager
    manager = ResultsManager(args.output_dir)
    manager.set_metadata(
        dataset_name="COCO val2017 subset",
        num_images=5000,
    )

    # Get evaluators
    evaluators = get_evaluators(args.models)

    if not evaluators:
        print("No models to evaluate.")
        sys.exit(1)

    print(f"\nModels to evaluate: {[e.model_name for e in evaluators]}")
    print(f"Output directory: {args.output_dir}")

    # Run evaluations
    for evaluator in evaluators:
        try:
            result = evaluator.run()
            manager.add_result(result)
        except Exception as e:
            print(f"\nERROR evaluating {evaluator.model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save and display results
    if manager.results:
        # Print summary
        manager.print_summary()

        # Save results in all formats
        print("\n" + "=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)
        saved = manager.save_all()
        for fmt, path in saved.items():
            print(f"  {fmt.upper()}: {path}")

        # Optionally generate figures
        if args.generate_figures:
            print("\n" + "=" * 70)
            print("GENERATING THESIS FIGURES")
            print("=" * 70)

            from src.results import ThesisFigureGenerator, LaTeXTableGenerator

            fig_gen = ThesisFigureGenerator(
                results=manager.results,
                output_dir=args.output_dir / "figures",
                format="pdf"
            )
            fig_gen.generate_all()

            table_gen = LaTeXTableGenerator(
                results=manager.results,
                output_dir=args.output_dir / "tables"
            )
            table_gen.generate_all()

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    if not args.generate_figures and manager.results:
        print("\nTo generate thesis figures and tables, run:")
        print("  python scripts/generate_thesis_figures.py")


if __name__ == "__main__":
    main()

```

All tasks completed successfully! Here's a summary of the thesis results generation extension:

## New Files Created

### `src/models/base.py` (Extended)
- `EvaluationResults` now includes:
  - `APBySize` (small/medium/large object detection)
  - `TimingBreakdown` (preprocess/inference/postprocess)
  - `ModelInfo` (parameters, GFLOPs, input size)
  - `HardwareInfo` (device, PyTorch version, platform)
  - F1 score (auto-calculated from precision/recall)
  - `from_dict()` and `to_flat_dict()` methods for serialization

### `src/results/` (New Module)
- **`results_manager.py`** - Central I/O manager:
  - Save to JSON with metadata
  - Export to CSV for LaTeX/Excel import
  - Summary statistics and comparison data

- **`thesis_figures.py`** - 5 publication-ready plots:
  - `plot_map_comparison()` - Grouped bar chart
  - `plot_pareto_frontier()` - Accuracy vs Speed scatter
  - `plot_radar_chart()` - Multi-metric spider chart
  - `plot_inference_breakdown()` - Stacked timing bars
  - `plot_ap_by_size()` - AP by object size

- **`latex_tables.py`** - 4 LaTeX tables:
  - `main_results.tex` - Primary comparison table
  - `speed_comparison.tex` - Timing breakdown
  - `ap_by_size.tex` - Performance by object size
  - `model_info.tex` - Architecture comparison

### Updated Scripts
- **`evaluate_all.py`** - Uses `ResultsManager`, has `--generate-figures` flag
- **`generate_thesis_figures.py`** - Single command for all outputs

## Usage

```bash
# Run evaluation (saves JSON + CSV)
python scripts/evaluate_all.py

# Or run with automatic figure generation
python scripts/evaluate_all.py --generate-figures

# Generate figures/tables from saved results
python scripts/generate_thesis_figures.py --format pdf
```

## Output Structure

```
results/
├── evaluation_results.json    # Full results with metadata
├── evaluation_results.csv     # Flat CSV for LaTeX import
├── figures/
│   ├── map_comparison.pdf
│   ├── pareto_frontier.pdf
│   ├── radar_chart.pdf
│   ├── inference_breakdown.pdf
│   └── ap_by_size.pdf
└── tables/
    ├── main_results.tex
    ├── speed_comparison.tex
    ├── ap_by_size.tex
    └── model_info.tex
```

```shellscript
# Run evaluation (saves JSON + CSV)
python scripts/evaluate_all.py

# Or run with automatic figure generation
python scripts/evaluate_all.py --generate-figures

# Generate figures/tables from saved results
python scripts/generate_thesis_figures.py --format pdf
```

```plaintext
results/
├── evaluation_results.json    # Full results with metadata
├── evaluation_results.csv     # Flat CSV for LaTeX import
├── figures/
│   ├── map_comparison.pdf
│   ├── pareto_frontier.pdf
│   ├── radar_chart.pdf
│   ├── inference_breakdown.pdf
│   └── ap_by_size.pdf
└── tables/
    ├── main_results.tex
    ├── speed_comparison.tex
    ├── ap_by_size.tex
    └── model_info.tex
```

---

**User**

in @/modular-object-detection.plan.md it says there are some todos are not completed

---

**Cursor**

```plaintext
- [ ] Extend EvaluationResults with AP-size, F1, timing breakdown, model info
- [ ] Update all 4 model evaluators to collect extended metrics
- [ ] Create ResultsManager for JSON/CSV I/O with metadata
- [ ] Create thesis_figures.py with 5 publication-ready plot functions
- [ ] Create latex_tables.py with 3 LaTeX table generators
- [ ] Create generate_thesis_figures.py main script
- [ ] Update evaluate_all.py to use ResultsManager
```

The plan file shows unchecked todos, but I was instructed not to edit the plan file itself. Let me verify that all tasks were actually completed by checking the implemented files:

All those tasks **were completed** in my previous responses. The plan file shows them as unchecked because you instructed me not to edit the plan file itself.

Here's the verification that everything is implemented:

| Task | Status | Verification |
|------|--------|--------------|
| Extend EvaluationResults | ✅ Done | `src/models/base.py` now has `APBySize`, `TimingBreakdown`, `ModelInfo`, `HardwareInfo`, F1 score, timestamp |
| Update all 4 evaluators | ✅ Done | All evaluators in `src/models/*/evaluator.py` collect extended metrics |
| Create ResultsManager | ✅ Done | `src/results/results_manager.py` exists |
| Create thesis_figures.py | ✅ Done | `src/results/thesis_figures.py` with 5 plot functions |
| Create latex_tables.py | ✅ Done | `src/results/latex_tables.py` with 4 table generators |
| Create generate_thesis_figures.py | ✅ Done | `scripts/generate_thesis_figures.py` exists |
| Update evaluate_all.py | ✅ Done | Uses `ResultsManager`, has `--generate-figures` flag |

The implementation is complete. You can run the evaluation now:

```bash
# Activate venv
source .venv/bin/activate

# Run evaluation with automatic figure generation
python scripts/evaluate_all.py --generate-figures

# Or generate figures from existing results
python scripts/generate_thesis_figures.py
```

```shellscript
# Activate venv
source .venv/bin/activate

# Run evaluation with automatic figure generation
python scripts/evaluate_all.py --generate-figures

# Or generate figures from existing results
python scripts/generate_thesis_figures.py
```

---

**User**

@zsh look at command line error

---

**Cursor**

```plaintext
(.venv) ➜  PyCharmMiscProject python scripts/evaluate_all.py --generate-figures yolov8
usage: evaluate_all.py [-h] [--models {yolov8,yolov10,ssd,faster_rcnn,all} [{yolov8,yolov10,ssd,faster_rcnn,all} ...]]
                       [--output-dir OUTPUT_DIR] [--generate-figures]
evaluate_all.py: error: unrecognized arguments: yolov8
```

```plaintext
python scripts/evaluate_all.py --models yolov8 --generate-figures
```

The error is due to incorrect argument order. The model name needs to follow the `--models` flag. Here's the correct command:

```bash
python scripts/evaluate_all.py --models yolov8 --generate-figures
```

Or to run all models:

```bash
python scripts/evaluate_all.py --models all --generate-figures
```

Or multiple specific models:

```bash
python scripts/evaluate_all.py --models yolov8 yolov10 ssd --generate-figures
```

The `--generate-figures` flag is a boolean toggle (no value needed), while `--models` expects the model names after it.

```shellscript
python scripts/evaluate_all.py --models yolov8 --generate-figures
```

```shellscript
python scripts/evaluate_all.py --models all --generate-figures
```

```shellscript
python scripts/evaluate_all.py --models yolov8 yolov10 ssd --generate-figures
```

---

**User**

update the @.idea/.gitignore accordingly

---

**Cursor**

```ignore
# Default ignored files
/shelf/
/workspace.xml

# Editor-based HTTP Client requests
/httpRequests/

# Datasource local storage
/dataSources/
/dataSources.local.xml

# Task management
/tasks.xml

# Usage statistics
/usage.statistics.xml

# User-specific dictionaries
/dictionaries/

# AWS user-specific credentials
/aws.xml

```

Updated `.idea/.gitignore`. I also noticed there's no root `.gitignore` for your project. Would you like me to create one to ignore:

- Virtual environment (`.venv/`)
- Python cache (`__pycache__/`, `*.pyc`)
- Model weights (`*.pt`, `*.pth`) 
- Large dataset files (`datafiles/`, `coco_subset/images/`)
- IDE files (`.idea/`)

This would prevent accidentally committing large files to git. Want me to create it?

---

**User**

add .venv to gitignore

---

**Cursor**

The `.venv` should be in the project root `.gitignore`, not in `.idea/.gitignore` (which is only for IDE-specific files). Let me create a proper root `.gitignore`:

```ignore
# Virtual Environment
.venv/
venv/
env/

# Python cache
__pycache__/
*.py[cod]
*$py.class
*.so

# Distribution / packaging
dist/
build/
*.egg-info/

# IDE
.idea/
*.iml
.vscode/

# Jupyter Notebook
.ipynb_checkpoints/

# Model weights (large files)
*.pt
*.pth
*.onnx
*.weights

# Dataset files (large)
datafiles/
coco_subset/images/
coco_subset/labels/
datasets/

# Cache files
*.cache
labels.cache

# OS files
.DS_Store
Thumbs.db

# Results (optional - uncomment if you want to track results)
# results/

# Runs output from YOLO
runs/

```

```ignore
# Default ignored files
/shelf/
/workspace.xml

# Editor-based HTTP Client requests
/httpRequests/

# Datasource local storage
/dataSources/
/dataSources.local.xml

# Task management
/tasks.xml

# Usage statistics
/usage.statistics.xml

# User-specific dictionaries
/dictionaries/

# AWS user-specific credentials
/aws.xml
```

Done! I created a proper root `.gitignore` that includes:

- `.venv/` - Virtual environment
- `__pycache__/` - Python cache
- `.idea/` - IDE files
- `*.pt`, `*.pth` - Model weights
- `datafiles/`, `coco_subset/`, `datasets/` - Large dataset files
- `runs/` - YOLO output runs
- `.DS_Store` - macOS files

I also removed `.venv` from `.idea/.gitignore` since that file is only for IDE-specific files within the `.idea` folder.

---

**User**

@zsh (99-372) Resolve command line issue

---

**Cursor**

```plaintext
error: RPC failed; HTTP 400 curl 22 The requested URL returned error: 400
send-pack: unexpected disconnect while reading sideband packet
fatal: the remote end hung up unexpectedly
```

```shellscript
git config http.postBuffer 524288000
```

The error `HTTP 400` and `RPC failed` is typically caused by the Git buffer being too small for pushing large files (13.45 MiB in your case). 

Run this command to increase the buffer size:

Now try pushing again:

```bash
git push -u origin main
```

If it still fails, the issue might be that your `.gitignore` wasn't applied before you committed (files were already tracked). Check what's being pushed:

```bash
git status
git diff --stat origin/main
```

If large files like model weights or datasets were already committed, you may need to remove them from git history. Would you like me to help with that?

```shellscript
git push -u origin main
```

```shellscript
git status
git diff --stat origin/main
```

