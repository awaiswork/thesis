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


