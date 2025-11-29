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


