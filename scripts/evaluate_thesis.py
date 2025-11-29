#!/usr/bin/env python3
"""
Comprehensive Thesis Evaluation Script

Evaluates all object detection models for Master's thesis:
- YOLOv8: n, s, m, l, x (5 variants)
- YOLOv10: n, s, m, l, x (5 variants)
- RT-DETR-l (Transformer-based)
- SSD300 (Single-shot)
- RetinaNet (Focal Loss)
- Faster R-CNN (Two-stage)

Total: 14 models
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.config import RESULTS_DIR, NUM_SAMPLES
from src.models.yolov8 import YOLOv8Evaluator
from src.models.yolov10 import YOLOv10Evaluator
from src.models.ssd import SSDEvaluator
from src.models.faster_rcnn import FasterRCNNEvaluator


@dataclass
class ModelResult:
    """Comprehensive result for a single model."""
    model_name: str
    model_family: str  # 'YOLOv8', 'YOLOv10', 'RT-DETR', 'SSD', 'RetinaNet', 'Faster R-CNN'
    model_variant: str  # 'n', 's', 'm', 'l', 'x' or 'base'
    
    # Accuracy metrics
    map50: float
    map50_95: float
    map75: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # AP by object size
    ap_small: Optional[float] = None
    ap_medium: Optional[float] = None
    ap_large: Optional[float] = None
    
    # Speed metrics
    fps: float = 0.0
    latency_ms: float = 0.0
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    
    # Model info
    parameters: int = 0
    gflops: float = 0.0
    model_size_mb: float = 0.0
    input_size: int = 640
    
    # Metadata
    num_images: int = 0
    total_time_seconds: float = 0.0
    timestamp: str = ""
    
    # Hardware
    device: str = ""
    device_name: str = ""


class RTDETREvaluator:
    """Evaluator for RT-DETR model."""
    
    def __init__(self, size: str = "l", device: Optional[torch.device] = None):
        self.size = size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.weights_path = f"rtdetr-{size}.pt"
        
    @property
    def model_name(self) -> str:
        return f"RT-DETR-{self.size}"
    
    def load_model(self):
        from ultralytics import RTDETR
        print(f"Loading RT-DETR from: {self.weights_path}")
        self.model = RTDETR(self.weights_path)
        
    def get_model_info(self):
        from src.models.base import ModelInfo
        size_info = {
            "l": {"parameters": 32_000_000, "gflops": 103.8},
            "x": {"parameters": 65_000_000, "gflops": 232.0},
        }
        info = size_info.get(self.size, size_info["l"])
        return ModelInfo(
            name=self.model_name,
            parameters=info["parameters"],
            gflops=info["gflops"],
            input_size=640,
        )
    
    def evaluate(self):
        from src.config import YOLO_CONFIG_FILE, NUM_SAMPLES
        from src.models.base import EvaluationResults, TimingBreakdown, APBySize
        
        if self.model is None:
            self.load_model()
        
        print(f"Validating RT-DETR on: {YOLO_CONFIG_FILE}")
        
        start_time = time.time()
        
        metrics = self.model.val(
            data=str(YOLO_CONFIG_FILE),
            device=str(self.device),
            verbose=False
        )
        
        elapsed = time.time() - start_time
        time_per_img = (elapsed / NUM_SAMPLES) * 1000
        fps = NUM_SAMPLES / elapsed
        
        timing = TimingBreakdown()
        if hasattr(metrics, 'speed') and metrics.speed:
            timing = TimingBreakdown(
                preprocess_ms=metrics.speed.get('preprocess', 0),
                inference_ms=metrics.speed.get('inference', 0),
                postprocess_ms=metrics.speed.get('postprocess', 0),
            )
        
        # Extract mAP@0.75
        map75 = None
        try:
            if hasattr(metrics.box, 'maps') and len(metrics.box.maps) > 5:
                map75 = metrics.box.maps[5] * 100
        except:
            pass
        
        return EvaluationResults(
            model_name=self.model_name,
            map50=metrics.box.map50 * 100,
            map50_95=metrics.box.map * 100,
            map75=map75,
            precision=metrics.box.mp * 100 if hasattr(metrics.box, 'mp') else None,
            recall=metrics.box.mr * 100 if hasattr(metrics.box, 'mr') else None,
            time_per_image_ms=time_per_img,
            fps=fps,
            timing_breakdown=timing,
            num_images=NUM_SAMPLES,
            total_time_seconds=elapsed,
            model_info=self.get_model_info(),
        )


def get_model_size_mb(weights_path: str) -> float:
    """Get model file size in MB."""
    path = Path(weights_path)
    if path.exists():
        return path.stat().st_size / (1024 * 1024)
    return 0.0


def evaluate_model(evaluator, family: str, variant: str) -> ModelResult:
    """Evaluate a single model and return comprehensive results."""
    print(f"\n{'='*60}")
    print(f"Evaluating {evaluator.model_name}")
    print(f"{'='*60}")
    
    # Load and evaluate
    if evaluator.model is None:
        evaluator.load_model()
    
    result = evaluator.evaluate()
    model_info = evaluator.get_model_info()
    
    # Get model size
    model_size = 0.0
    if hasattr(evaluator, 'weights_path'):
        model_size = get_model_size_mb(evaluator.weights_path)
    
    # Extract timing breakdown
    preprocess_ms = 0.0
    inference_ms = 0.0
    postprocess_ms = 0.0
    if result.timing_breakdown:
        preprocess_ms = result.timing_breakdown.preprocess_ms or 0.0
        inference_ms = result.timing_breakdown.inference_ms or 0.0
        postprocess_ms = result.timing_breakdown.postprocess_ms or 0.0
    
    # Extract AP by size
    ap_small, ap_medium, ap_large = None, None, None
    if result.ap_by_size:
        ap_small = result.ap_by_size.small
        ap_medium = result.ap_by_size.medium
        ap_large = result.ap_by_size.large
    
    # Compute F1 if not available
    f1 = result.f1_score
    if f1 is None and result.precision and result.recall:
        if result.precision + result.recall > 0:
            f1 = 2 * result.precision * result.recall / (result.precision + result.recall)
    
    # Get hardware info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = ""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    
    return ModelResult(
        model_name=evaluator.model_name,
        model_family=family,
        model_variant=variant,
        map50=result.map50,
        map50_95=result.map50_95,
        map75=result.map75,
        precision=result.precision,
        recall=result.recall,
        f1_score=f1,
        ap_small=ap_small,
        ap_medium=ap_medium,
        ap_large=ap_large,
        fps=result.fps,
        latency_ms=result.time_per_image_ms,
        preprocess_ms=preprocess_ms,
        inference_ms=inference_ms,
        postprocess_ms=postprocess_ms,
        parameters=model_info.parameters if model_info else 0,
        gflops=model_info.gflops if model_info else 0.0,
        model_size_mb=model_size,
        input_size=model_info.input_size if model_info else 640,
        num_images=result.num_images,
        total_time_seconds=result.total_time_seconds,
        timestamp=datetime.now().isoformat(),
        device=device,
        device_name=device_name,
    )


def run_all_evaluations(
    skip_existing: bool = False,
    models_filter: Optional[List[str]] = None,
) -> Dict[str, List[ModelResult]]:
    """
    Run evaluations for all models.
    
    Returns dict organized by model family.
    """
    results = {
        "YOLOv8": [],
        "YOLOv10": [],
        "RT-DETR": [],
        "SSD": [],
        "RetinaNet": [],
        "Faster R-CNN": [],
    }
    
    all_results = []
    
    # Define all models to evaluate
    models_to_evaluate = []
    
    # YOLOv8 variants
    for size in ["n", "s", "m", "l", "x"]:
        models_to_evaluate.append(("YOLOv8", size, lambda s=size: YOLOv8Evaluator(size=s)))
    
    # YOLOv10 variants
    for size in ["n", "s", "m", "l", "x"]:
        models_to_evaluate.append(("YOLOv10", size, lambda s=size: YOLOv10Evaluator(size=s)))
    
    # RT-DETR
    models_to_evaluate.append(("RT-DETR", "l", lambda: RTDETREvaluator(size="l")))
    
    # SSD300
    models_to_evaluate.append(("SSD", "300", lambda: SSDEvaluator(use_retinanet=False)))
    
    # RetinaNet
    models_to_evaluate.append(("RetinaNet", "base", lambda: SSDEvaluator(use_retinanet=True)))
    
    # Faster R-CNN
    models_to_evaluate.append(("Faster R-CNN", "base", lambda: FasterRCNNEvaluator()))
    
    print(f"\n{'='*70}")
    print("COMPREHENSIVE THESIS EVALUATION")
    print(f"{'='*70}")
    print(f"Total models to evaluate: {len(models_to_evaluate)}")
    print(f"Dataset: COCO val2017 subset ({NUM_SAMPLES} images)")
    
    # Run evaluations
    for i, (family, variant, evaluator_fn) in enumerate(models_to_evaluate, 1):
        model_id = f"{family}-{variant}"
        
        # Skip if filtered
        if models_filter and model_id not in models_filter:
            continue
        
        print(f"\n[{i}/{len(models_to_evaluate)}] {family} {variant}")
        
        try:
            evaluator = evaluator_fn()
            result = evaluate_model(evaluator, family, variant)
            results[family].append(result)
            all_results.append(result)
            
            # Print summary
            print(f"\nResults: {result.model_name}")
            print(f"  mAP@0.5: {result.map50:.1f}%")
            print(f"  mAP@0.5:0.95: {result.map50_95:.1f}%")
            if result.recall:
                print(f"  Recall: {result.recall:.1f}%")
            print(f"  FPS: {result.fps:.1f}")
            print(f"  Latency: {result.latency_ms:.1f}ms")
            
        except Exception as e:
            print(f"ERROR evaluating {family} {variant}: {e}")
            import traceback
            traceback.print_exc()
    
    return results, all_results


def save_results(
    results_by_family: Dict[str, List[ModelResult]],
    all_results: List[ModelResult],
    output_dir: Path
):
    """Save all results to JSON files."""
    output_dir = output_dir / "thesis" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save by family
    for family, family_results in results_by_family.items():
        if family_results:
            filename = family.lower().replace(" ", "_").replace("-", "") + "_results.json"
            filepath = output_dir / filename
            with open(filepath, "w") as f:
                json.dump({
                    "family": family,
                    "num_models": len(family_results),
                    "results": [asdict(r) for r in family_results]
                }, f, indent=2)
            print(f"Saved: {filepath}")
    
    # Save all results combined
    all_filepath = output_dir / "all_models_results.json"
    with open(all_filepath, "w") as f:
        json.dump({
            "evaluation_type": "Comprehensive Thesis Evaluation",
            "num_models": len(all_results),
            "dataset": "COCO val2017 subset",
            "num_images": NUM_SAMPLES,
            "evaluation_date": datetime.now().isoformat(),
            "results": [asdict(r) for r in all_results],
            "summary": generate_summary(all_results),
        }, f, indent=2)
    print(f"Saved: {all_filepath}")
    
    return all_filepath


def generate_summary(all_results: List[ModelResult]) -> Dict[str, Any]:
    """Generate summary statistics."""
    if not all_results:
        return {}
    
    # Find best models
    best_map50 = max(all_results, key=lambda x: x.map50)
    best_map50_95 = max(all_results, key=lambda x: x.map50_95)
    best_recall = max(all_results, key=lambda x: x.recall or 0)
    fastest = max(all_results, key=lambda x: x.fps)
    smallest = min(all_results, key=lambda x: x.parameters if x.parameters > 0 else float('inf'))
    
    return {
        "best_map50": {"model": best_map50.model_name, "value": best_map50.map50},
        "best_map50_95": {"model": best_map50_95.model_name, "value": best_map50_95.map50_95},
        "best_recall": {"model": best_recall.model_name, "value": best_recall.recall},
        "fastest": {"model": fastest.model_name, "fps": fastest.fps},
        "smallest": {"model": smallest.model_name, "parameters": smallest.parameters},
        "speed_range": {
            "min_fps": min(r.fps for r in all_results),
            "max_fps": max(r.fps for r in all_results),
        },
        "accuracy_range": {
            "min_map50_95": min(r.map50_95 for r in all_results),
            "max_map50_95": max(r.map50_95 for r in all_results),
        },
    }


def print_summary_table(all_results: List[ModelResult]):
    """Print a summary table of all results."""
    print("\n" + "=" * 100)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 100)
    
    # Sort by family then variant
    sorted_results = sorted(all_results, key=lambda x: (x.model_family, x.model_variant))
    
    print(f"\n{'Model':<20} {'mAP@0.5':>10} {'mAP@0.5:0.95':>14} {'Recall':>10} {'Precision':>10} {'FPS':>10} {'Params (M)':>12}")
    print("-" * 100)
    
    current_family = None
    for r in sorted_results:
        if r.model_family != current_family:
            if current_family is not None:
                print("-" * 100)
            current_family = r.model_family
        
        recall_str = f"{r.recall:.1f}%" if r.recall else "N/A"
        precision_str = f"{r.precision:.1f}%" if r.precision else "N/A"
        params_str = f"{r.parameters / 1e6:.1f}" if r.parameters else "N/A"
        
        print(f"{r.model_name:<20} {r.map50:>9.1f}% {r.map50_95:>13.1f}% {recall_str:>10} {precision_str:>10} {r.fps:>10.1f} {params_str:>12}")
    
    print("=" * 100)


def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive thesis model evaluation")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Output directory for results"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that already have results"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Run all evaluations
    results_by_family, all_results = run_all_evaluations()
    
    # Print summary
    print_summary_table(all_results)
    
    # Save results
    save_results(results_by_family, all_results, args.output_dir)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nTotal models evaluated: {len(all_results)}")
    print(f"Results saved to: {args.output_dir / 'thesis' / 'data'}")


if __name__ == "__main__":
    main()
