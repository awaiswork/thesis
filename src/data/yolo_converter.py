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


