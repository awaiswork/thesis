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


