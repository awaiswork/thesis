# Object Detection Performance Analysis for Visually Impaired Assistance

A comprehensive evaluation framework for comparing object detection models, designed specifically for assistive technology applications serving visually impaired persons.

## Overview

This project evaluates **14 object detection models** on the COCO val2017 dataset, focusing on metrics critical for assistive technology:
- **High Recall**: Minimizing missed detections for safety
- **Real-time Performance**: Enabling mobile deployment
- **Accuracy-Speed Trade-offs**: Finding optimal models for different devices

## Models Evaluated

### YOLO Family (10 models)
| Model | mAP@0.5:0.95 | Recall | FPS | Parameters |
|-------|-------------|--------|-----|------------|
| YOLOv8n | 37.0% | 47.3% | 411 | 3.2M |
| YOLOv8s | 44.6% | 56.0% | 352 | 11.2M |
| YOLOv8m | 50.0% | 60.7% | 253 | 25.9M |
| YOLOv8l | 52.8% | 63.1% | 190 | 43.7M |
| YOLOv8x | 53.9% | 64.5% | 139 | 68.2M |
| YOLOv10n | 38.2% | 48.7% | 462 | 2.3M |
| YOLOv10s | 46.0% | 56.1% | 385 | 7.2M |
| YOLOv10m | 50.8% | 60.2% | 271 | 15.4M |
| YOLOv10l | 52.9% | 63.7% | 209 | 24.4M |
| YOLOv10x | **54.3%** | 63.6% | 160 | 29.5M |

### Other Architectures (4 models)
| Model | mAP@0.5:0.95 | Recall | FPS | Architecture |
|-------|-------------|--------|-----|--------------|
| RT-DETR-l | 51.9% | **65.2%** | 148 | Transformer |
| RetinaNet | 41.5% | 58.8% | 55 | Focal Loss |
| Faster R-CNN | 46.7% | N/A | 48 | Two-stage |
| SSD300 | 25.0% | 35.0% | 83 | Single-stage |

## Key Findings

- **Best Accuracy**: YOLOv10x (54.3% mAP@0.5:0.95)
- **Best Recall**: RT-DETR-l (65.2%) - critical for safety
- **Fastest Model**: YOLOv10n (462 FPS)
- **Best for Mobile**: YOLOv10s (balanced accuracy/speed)

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
│   │   ├── ssd/evaluator.py         # SSD/RetinaNet evaluation
│   │   ├── faster_rcnn/evaluator.py # Faster R-CNN evaluation
│   │   └── rtdetr/evaluator.py      # RT-DETR evaluation
│   └── results/
│       └── manager.py               # Results management
├── scripts/
│   ├── prepare_data.py              # Data preparation
│   ├── evaluate_all.py              # Run all evaluations
│   ├── evaluate_thesis.py           # Comprehensive 14-model evaluation
│   ├── evaluate_assistive.py        # Assistive tech focused evaluation
│   ├── generate_thesis_results.py   # Generate figures & tables
│   └── generate_thesis_chapter.py   # Generate PDF chapter
├── results/
│   ├── thesis/
│   │   ├── data/                    # JSON results
│   │   │   ├── all_models_results.json
│   │   │   ├── yolov8_results.json
│   │   │   └── yolov10_results.json
│   │   ├── figures/                 # PDF figures for thesis
│   │   │   ├── family/              # YOLO variant comparisons
│   │   │   ├── cross_family/        # YOLOv8 vs YOLOv10
│   │   │   └── overall/             # All models comparisons
│   │   ├── tables/                  # LaTeX tables
│   │   └── Results_and_Discussion_Chapter.pdf
│   └── evaluation_results.json
├── datafiles/                       # Source data (ZIP files)
├── coco_subset/                     # Prepared dataset
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

4. Model weights are automatically downloaded on first run.

## Usage

### 1. Prepare Data (one-time)

```bash
python scripts/prepare_data.py
```

This extracts a subset of 5000 COCO images and converts annotations to YOLO format.

### 2. Run Comprehensive Evaluation

Evaluate all 14 models:
```bash
python scripts/evaluate_thesis.py
```

Run specific models:
```bash
python scripts/evaluate_all.py --models yolov8 yolov10 --size s
python scripts/evaluate_all.py --models all --size m
```

### 3. Run Assistive Tech Evaluation

Focus on critical classes for visually impaired assistance:
```bash
python scripts/evaluate_assistive.py --models mobile --latency-samples 50
```

### 4. Generate Thesis Figures & Tables

```bash
python scripts/generate_thesis_results.py
```

This generates:
- 10 PDF figures (comparisons, Pareto frontier, radar charts)
- 6 LaTeX tables (ready for thesis)

### 5. Generate Results Chapter PDF

```bash
python scripts/generate_thesis_chapter.py
```

Generates `results/thesis/Results_and_Discussion_Chapter.pdf`

## Evaluation Metrics

| Metric | Description | Importance for Assistive Tech |
|--------|-------------|------------------------------|
| mAP@0.5 | Mean Average Precision at IoU 0.5 | Overall accuracy |
| mAP@0.5:0.95 | Mean AP averaged over IoU 0.5-0.95 | Strict accuracy |
| Recall | Detection recall | **Critical** - missing objects is dangerous |
| Precision | Detection precision | Reduces false alarms |
| FPS | Frames per second | Real-time performance |
| Latency | Milliseconds per image | Response time |

## Generated Outputs

### Figures (PDF)
- `yolov8_variants_comparison.pdf` - YOLOv8 n/s/m/l/x scaling
- `yolov10_variants_comparison.pdf` - YOLOv10 n/s/m/l/x scaling
- `yolov8_vs_yolov10.pdf` - Side-by-side comparison
- `all_models_map_comparison.pdf` - All 14 models accuracy
- `all_models_speed_comparison.pdf` - FPS comparison
- `pareto_frontier.pdf` - Accuracy vs Speed trade-off
- `model_scaling_analysis.pdf` - Model size vs accuracy
- `latency_comparison.pdf` - Inference latency
- `recall_comparison.pdf` - Critical for assistive tech
- `radar_charts.pdf` - Multi-metric comparison

### LaTeX Tables
- `yolov8_results.tex` / `yolov10_results.tex` - Family results
- `yolo_comparison.tex` - YOLOv8 vs YOLOv10
- `all_models_results.tex` - Complete comparison
- `model_complexity.tex` - Parameters, GFLOPs
- `speed_analysis.tex` - FPS and latency

## Hardware Requirements

The code automatically detects available hardware:
- **CUDA** - NVIDIA GPU (recommended)
- **MPS** - Apple Silicon GPU
- **CPU** - Fallback

Evaluation was performed on:
- GPU: NVIDIA GeForce RTX 4090
- Dataset: COCO val2017 (5000 images)

## Mobile Deployment Recommendations

Based on evaluation results:

| Device Type | Recommended Model | Expected FPS | mAP |
|------------|-------------------|--------------|-----|
| High-end Phone | YOLOv10s | ~25 | 46.0% |
| Mid-range Phone | YOLOv10n | ~20 | 38.2% |
| Raspberry Pi 5 | YOLOv10n | ~15 | 38.2% |
| Jetson Nano | YOLOv10m | ~20 | 50.8% |

## Citation

If you use this evaluation framework in your research, please cite:

```
@thesis{object_detection_assistive,
  title={Performance Analysis of Object Detection Models for Visually Impaired Assistance},
  year={2024},
  type={Master's Thesis}
}
```

## License

This project is for educational and research purposes.
