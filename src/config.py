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

# YOLOv8 variants (nano, small, medium, large, xlarge)
YOLOV8_WEIGHTS = {
    "n": PROJECT_ROOT / "yolov8n.pt",
    "s": PROJECT_ROOT / "yolov8s.pt",
    "m": PROJECT_ROOT / "yolov8m.pt",
    "l": PROJECT_ROOT / "yolov8l.pt",
    "x": PROJECT_ROOT / "yolov8x.pt",
}

# YOLOv10 variants
YOLOV10_WEIGHTS = {
    "n": PROJECT_ROOT / "yolov10n.pt",
    "s": PROJECT_ROOT / "yolov10s.pt",
    "m": PROJECT_ROOT / "yolov10m.pt",
    "l": PROJECT_ROOT / "yolov10l.pt",
    "x": PROJECT_ROOT / "yolov10x.pt",
}

# Default model size
DEFAULT_YOLO_SIZE = "n"

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

# =============================================================================
# Assistive Technology Configuration (Visually Impaired Navigation)
# =============================================================================

# Critical classes for visually impaired navigation assistance
# Priority 1: Safety-critical (obstacles, vehicles, hazards)
CRITICAL_CLASSES_SAFETY = {
    0: "person",        # People to avoid collision
    1: "bicycle",       # Moving hazard
    2: "car",           # Vehicle - dangerous
    3: "motorcycle",    # Vehicle - dangerous
    5: "bus",           # Large vehicle
    7: "truck",         # Large vehicle
    9: "traffic light", # Navigation signal
    10: "fire hydrant", # Obstacle
    11: "stop sign",    # Navigation signal
    13: "bench",        # Seating/obstacle
    15: "cat",          # Pet - might move suddenly
    16: "dog",          # Pet - might move suddenly
    56: "chair",        # Furniture obstacle
}

# Priority 2: Navigation-relevant objects
CRITICAL_CLASSES_NAVIGATION = {
    12: "parking meter",  # Obstacle
    24: "backpack",       # Dropped item hazard
    25: "umbrella",       # Obstacle when open
    28: "suitcase",       # Obstacle
    57: "couch",          # Furniture
    58: "potted plant",   # Obstacle
    59: "bed",            # Furniture
    60: "dining table",   # Furniture
}

# All critical classes combined (for filtering evaluation)
CRITICAL_CLASSES = {**CRITICAL_CLASSES_SAFETY, **CRITICAL_CLASSES_NAVIGATION}

# COCO category IDs for critical classes (COCO uses 1-indexed, different from class index)
# Mapping: class_index -> coco_category_id
COCO_CLASS_TO_CATEGORY = {
    0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10,
    10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21,
    20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34,
    30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44,
    40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55,
    50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65,
    60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79,
    70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90,
}

CRITICAL_CATEGORY_IDS = [COCO_CLASS_TO_CATEGORY[k] for k in CRITICAL_CLASSES.keys()]

# =============================================================================
# Mobile Device Simulation Settings
# =============================================================================

# Simulated mobile device constraints
MOBILE_DEVICES = {
    "high_end_phone": {
        "name": "High-End Phone (iPhone 15/Pixel 8)",
        "target_fps": 30,
        "max_input_size": 640,
        "gpu_memory_mb": 4096,
    },
    "mid_range_phone": {
        "name": "Mid-Range Phone",
        "target_fps": 20,
        "max_input_size": 480,
        "gpu_memory_mb": 2048,
    },
    "raspberry_pi": {
        "name": "Raspberry Pi 5",
        "target_fps": 10,
        "max_input_size": 416,
        "gpu_memory_mb": 0,  # CPU only
    },
    "jetson_nano": {
        "name": "NVIDIA Jetson Nano",
        "target_fps": 25,
        "max_input_size": 640,
        "gpu_memory_mb": 4096,
    },
}

# Latency requirements for assistive technology
MAX_ACCEPTABLE_LATENCY_MS = 100  # 100ms for real-time feedback
TARGET_LATENCY_MS = 50  # Ideal latency


