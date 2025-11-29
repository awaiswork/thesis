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


