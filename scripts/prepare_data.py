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


