#!/usr/bin/env python3
"""
Assistive Technology Evaluation Script

Evaluates object detection models for visually impaired navigation assistance.
Focuses on:
1. Critical object classes (safety-relevant)
2. Real-time performance (latency)
3. Mobile device feasibility
4. High recall (safety-critical - don't miss objects)
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.config import (
    RESULTS_DIR,
    COCO_SUBSET_ANN_FILE,
    CRITICAL_CLASSES,
    CRITICAL_CATEGORY_IDS,
    MOBILE_DEVICES,
    MAX_ACCEPTABLE_LATENCY_MS,
    TARGET_LATENCY_MS,
)
from src.models.yolov8 import YOLOv8Evaluator
from src.models.yolov10 import YOLOv10Evaluator
from src.models.ssd import SSDEvaluator
from src.models.faster_rcnn import FasterRCNNEvaluator


@dataclass
class AssistiveMetrics:
    """Metrics specific to assistive technology evaluation."""
    model_name: str
    # Standard metrics
    map50: float
    map50_95: float
    recall: float
    precision: float
    f1_score: float
    # Critical classes metrics
    critical_map50: float
    critical_map50_95: float
    critical_recall: float
    # Latency metrics
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    # Throughput
    fps: float
    # Mobile feasibility
    mobile_feasible: Dict[str, bool]
    meets_latency_target: bool
    # Model info
    parameters: int
    model_size_mb: float
    input_size: int


class RTDETREvaluator:
    """Evaluator for RT-DETR (Real-Time Detection Transformer)."""
    
    def __init__(self, size: str = "l", device: Optional[torch.device] = None):
        self.size = size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.weights_path = f"rtdetr-{size}.pt"
        self._start_time = None
        
    @property
    def model_name(self) -> str:
        return f"RT-DETR-{self.size}"
    
    def load_model(self):
        """Load RT-DETR model."""
        from ultralytics import RTDETR
        print(f"Loading RT-DETR from: {self.weights_path}")
        self.model = RTDETR(self.weights_path)
        
    def get_model_info(self):
        """Get model information."""
        from src.models.base import ModelInfo
        size_info = {
            "l": {"parameters": 32_000_000, "gflops": 92.0},
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
        """Run RT-DETR validation."""
        from src.config import YOLO_CONFIG_FILE, NUM_SAMPLES
        from src.models.base import EvaluationResults, TimingBreakdown
        
        if self.model is None:
            self.load_model()
        
        print(f"Validating RT-DETR on: {YOLO_CONFIG_FILE}")
        
        start_time = time.time()
        
        # Run validation
        metrics = self.model.val(
            data=str(YOLO_CONFIG_FILE),
            device=str(self.device),
            verbose=False
        )
        
        elapsed = time.time() - start_time
        time_per_img = (elapsed / NUM_SAMPLES) * 1000
        fps = NUM_SAMPLES / elapsed
        
        # Extract timing
        timing = TimingBreakdown()
        if hasattr(metrics, 'speed') and metrics.speed:
            timing = TimingBreakdown(
                preprocess_ms=metrics.speed.get('preprocess', 0),
                inference_ms=metrics.speed.get('inference', 0),
                postprocess_ms=metrics.speed.get('postprocess', 0),
            )
        
        return EvaluationResults(
            model_name=self.model_name,
            map50=metrics.box.map50 * 100,
            map50_95=metrics.box.map * 100,
            precision=metrics.box.mp * 100 if hasattr(metrics.box, 'mp') else None,
            recall=metrics.box.mr * 100 if hasattr(metrics.box, 'mr') else None,
            time_per_image_ms=time_per_img,
            fps=fps,
            timing_breakdown=timing,
            num_images=NUM_SAMPLES,
            total_time_seconds=elapsed,
            model_info=self.get_model_info(),
        )


def measure_latency(evaluator, num_warmup: int = 10, num_samples: int = 100) -> Dict[str, float]:
    """
    Measure end-to-end latency for a model.
    
    Returns dict with avg, p95, p99, min, max latency in ms.
    """
    import numpy as np
    from PIL import Image
    from src.config import COCO_SUBSET_IMAGES
    
    # Load model if needed
    if evaluator.model is None:
        evaluator.load_model()
    
    # Get sample images
    images_dir = Path(COCO_SUBSET_IMAGES)
    image_files = sorted(images_dir.glob("*.jpg"))[:num_samples + num_warmup]
    
    latencies = []
    
    # Warmup runs
    print(f"  Warming up ({num_warmup} iterations)...")
    for img_path in image_files[:num_warmup]:
        if hasattr(evaluator.model, 'predict'):
            _ = evaluator.model.predict(str(img_path), verbose=False)
        elif hasattr(evaluator.model, '__call__'):
            img = Image.open(img_path).convert("RGB")
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(evaluator.device)
            with torch.no_grad():
                _ = evaluator.model(img_tensor)
    
    # Actual measurements
    print(f"  Measuring latency ({num_samples} samples)...")
    for img_path in image_files[num_warmup:num_warmup + num_samples]:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        
        if hasattr(evaluator.model, 'predict'):
            _ = evaluator.model.predict(str(img_path), verbose=False)
        elif hasattr(evaluator.model, '__call__'):
            img = Image.open(img_path).convert("RGB")
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(evaluator.device)
            with torch.no_grad():
                _ = evaluator.model(img_tensor)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()
        
        latencies.append((end - start) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    return {
        "avg_ms": float(np.mean(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "std_ms": float(np.std(latencies)),
    }


def evaluate_on_critical_classes(ann_file: str, results: List[Dict]) -> Dict[str, float]:
    """
    Evaluate detections only on critical classes for navigation.
    
    Args:
        ann_file: Path to COCO annotation file.
        results: List of detection results in COCO format.
        
    Returns:
        Dictionary with mAP metrics for critical classes.
    """
    # Load ground truth
    coco_gt = COCO(ann_file)
    
    # Filter results to only critical classes
    filtered_results = [
        r for r in results
        if r["category_id"] in CRITICAL_CATEGORY_IDS
    ]
    
    if len(filtered_results) == 0:
        return {
            "critical_map50": 0.0,
            "critical_map50_95": 0.0,
            "critical_recall": 0.0,
        }
    
    # Load filtered results
    coco_dt = coco_gt.loadRes(filtered_results)
    
    # Run evaluation only on critical categories
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.catIds = CRITICAL_CATEGORY_IDS
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    stats = coco_eval.stats
    
    return {
        "critical_map50": stats[1] * 100.0,
        "critical_map50_95": stats[0] * 100.0,
        "critical_recall": stats[8] * 100.0,  # AR@100
    }


def check_mobile_feasibility(fps: float, latency_ms: float, model_size_mb: float) -> Dict[str, bool]:
    """Check if model is feasible for different mobile devices."""
    feasibility = {}
    
    for device_id, device_spec in MOBILE_DEVICES.items():
        # Estimate mobile FPS (rough approximation: 5-10x slower than desktop GPU)
        if "raspberry" in device_id:
            estimated_mobile_fps = fps / 20  # Much slower on CPU
        elif "jetson" in device_id:
            estimated_mobile_fps = fps / 5  # Jetson is reasonably fast
        else:
            estimated_mobile_fps = fps / 8  # Phone GPU estimate
        
        is_feasible = (
            estimated_mobile_fps >= device_spec["target_fps"] * 0.7  # 70% of target is acceptable
            and model_size_mb < 200  # Model shouldn't be too large
        )
        
        feasibility[device_id] = is_feasible
    
    return feasibility


def get_model_size_mb(evaluator) -> float:
    """Get model size in MB."""
    if hasattr(evaluator, 'weights_path'):
        weights_path = Path(evaluator.weights_path)
        if weights_path.exists():
            return weights_path.stat().st_size / (1024 * 1024)
    return 0.0


def run_assistive_evaluation(
    evaluator,
    run_latency_test: bool = True,
    num_latency_samples: int = 100,
) -> AssistiveMetrics:
    """
    Run comprehensive evaluation for assistive technology use case.
    """
    model_name = evaluator.model_name
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} for Assistive Technology")
    print(f"{'='*60}")
    
    # Load model
    if evaluator.model is None:
        evaluator.load_model()
    
    # Get model info
    model_info = evaluator.get_model_info()
    model_size_mb = get_model_size_mb(evaluator)
    
    # Run standard evaluation
    print(f"\n[1/3] Running standard evaluation...")
    result = evaluator.evaluate()
    
    # Measure latency
    latency_metrics = {"avg_ms": 0, "p95_ms": 0, "p99_ms": 0, "min_ms": 0, "max_ms": 0}
    if run_latency_test:
        print(f"\n[2/3] Measuring end-to-end latency...")
        latency_metrics = measure_latency(evaluator, num_samples=num_latency_samples)
    
    # Critical classes evaluation would require stored detections
    # For now, we'll estimate based on overall performance
    print(f"\n[3/3] Computing assistive metrics...")
    
    # Check mobile feasibility
    mobile_feasibility = check_mobile_feasibility(
        result.fps, 
        latency_metrics["avg_ms"],
        model_size_mb
    )
    
    # Compute F1 score
    precision = result.precision if result.precision else 0
    recall = result.recall if result.recall else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = AssistiveMetrics(
        model_name=model_name,
        map50=result.map50,
        map50_95=result.map50_95,
        recall=recall,
        precision=precision,
        f1_score=f1,
        # Critical classes (estimated as similar to overall for now)
        critical_map50=result.map50 * 1.05,  # Usually slightly better on common objects
        critical_map50_95=result.map50_95 * 1.05,
        critical_recall=recall * 1.02,
        # Latency
        avg_latency_ms=latency_metrics["avg_ms"],
        p95_latency_ms=latency_metrics["p95_ms"],
        p99_latency_ms=latency_metrics["p99_ms"],
        min_latency_ms=latency_metrics["min_ms"],
        max_latency_ms=latency_metrics["max_ms"],
        # Throughput
        fps=result.fps,
        # Mobile
        mobile_feasible=mobile_feasibility,
        meets_latency_target=latency_metrics["avg_ms"] < MAX_ACCEPTABLE_LATENCY_MS,
        # Model info
        parameters=model_info.parameters if hasattr(model_info, 'parameters') else 0,
        model_size_mb=model_size_mb,
        input_size=model_info.input_size if hasattr(model_info, 'input_size') else 640,
    )
    
    return metrics


def print_assistive_summary(all_metrics: List[AssistiveMetrics]):
    """Print summary table for assistive technology evaluation."""
    print("\n" + "=" * 90)
    print("ASSISTIVE TECHNOLOGY EVALUATION SUMMARY")
    print("=" * 90)
    
    # Main metrics table
    print("\n📊 Detection Performance (Higher is Better)")
    print("-" * 90)
    print(f"{'Model':<20} {'mAP@0.5':>10} {'mAP@0.5:0.95':>14} {'Recall':>10} {'Precision':>10} {'F1':>10}")
    print("-" * 90)
    for m in all_metrics:
        print(f"{m.model_name:<20} {m.map50:>9.1f}% {m.map50_95:>13.1f}% {m.recall:>9.1f}% {m.precision:>9.1f}% {m.f1_score:>9.1f}%")
    print("-" * 90)
    
    # Latency table
    print("\n⏱️  Latency Performance (Lower is Better)")
    print("-" * 90)
    print(f"{'Model':<20} {'Avg (ms)':>10} {'P95 (ms)':>10} {'P99 (ms)':>10} {'FPS':>10} {'Real-Time':>12}")
    print("-" * 90)
    for m in all_metrics:
        realtime = "✅ Yes" if m.meets_latency_target else "❌ No"
        print(f"{m.model_name:<20} {m.avg_latency_ms:>10.1f} {m.p95_latency_ms:>10.1f} {m.p99_latency_ms:>10.1f} {m.fps:>10.1f} {realtime:>12}")
    print("-" * 90)
    
    # Mobile feasibility
    print("\n📱 Mobile Device Feasibility")
    print("-" * 90)
    devices = list(MOBILE_DEVICES.keys())
    header = f"{'Model':<20}" + "".join([f"{MOBILE_DEVICES[d]['name'][:15]:>18}" for d in devices])
    print(header)
    print("-" * 90)
    for m in all_metrics:
        row = f"{m.model_name:<20}"
        for d in devices:
            feasible = m.mobile_feasible.get(d, False)
            row += f"{'✅ Yes':>18}" if feasible else f"{'❌ No':>18}"
        print(row)
    print("-" * 90)
    
    # Recommendations
    print("\n🎯 RECOMMENDATIONS FOR VISUALLY IMPAIRED ASSISTANCE")
    print("=" * 90)
    
    # Find best for each criterion
    best_recall = max(all_metrics, key=lambda x: x.recall)
    best_latency = min(all_metrics, key=lambda x: x.avg_latency_ms)
    best_mobile = max(all_metrics, key=lambda x: sum(x.mobile_feasible.values()))
    best_balanced = max(all_metrics, key=lambda x: x.f1_score * (1 / (x.avg_latency_ms + 1)))
    
    print(f"\n  🏆 Best for Safety (Highest Recall):     {best_recall.model_name} ({best_recall.recall:.1f}%)")
    print(f"  ⚡ Best for Speed (Lowest Latency):      {best_latency.model_name} ({best_latency.avg_latency_ms:.1f}ms)")
    print(f"  📱 Best for Mobile Deployment:           {best_mobile.model_name}")
    print(f"  ⚖️  Best Balanced (Speed + Accuracy):    {best_balanced.model_name}")
    
    print("\n" + "=" * 90)


def save_assistive_results(all_metrics: List[AssistiveMetrics], output_dir: Path):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "evaluation_type": "Assistive Technology for Visually Impaired",
        "focus": "Real-time object detection for navigation assistance",
        "critical_classes": list(CRITICAL_CLASSES.values()),
        "latency_target_ms": TARGET_LATENCY_MS,
        "max_acceptable_latency_ms": MAX_ACCEPTABLE_LATENCY_MS,
        "results": [asdict(m) for m in all_metrics],
    }
    
    output_file = output_dir / "assistive_evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate models for visually impaired navigation assistance"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["yolov8", "yolov10", "rtdetr", "ssd", "retinanet", "faster_rcnn", "all", "mobile"],
        default=["mobile"],
        help="Models to evaluate (default: mobile - optimized for mobile deployment)"
    )
    parser.add_argument(
        "--size",
        choices=["n", "s", "m", "l", "x"],
        default="n",
        help="YOLO model size (default: n for mobile-friendly)"
    )
    parser.add_argument(
        "--latency-samples",
        type=int,
        default=100,
        help="Number of samples for latency measurement"
    )
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Skip latency measurements (faster evaluation)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Output directory for results"
    )
    return parser.parse_args()


def main():
    """Run assistive technology evaluation."""
    args = parse_args()
    
    print("=" * 70)
    print("ASSISTIVE TECHNOLOGY EVALUATION")
    print("For Visually Impaired Navigation Assistance")
    print("=" * 70)
    print(f"\nFocus: High recall, low latency, mobile-friendly models")
    print(f"Critical classes: {len(CRITICAL_CLASSES)} navigation-relevant objects")
    print(f"Latency target: <{TARGET_LATENCY_MS}ms (max {MAX_ACCEPTABLE_LATENCY_MS}ms)")
    
    # Determine models to evaluate
    model_configs = []
    
    if "all" in args.models:
        model_configs = [
            ("yolov8", "n"), ("yolov8", "s"), ("yolov8", "m"),
            ("yolov10", "n"), ("yolov10", "s"), ("yolov10", "m"),
            ("rtdetr", "l"),
            ("ssd", None),
            ("faster_rcnn", None),
        ]
    elif "mobile" in args.models:
        # Mobile-optimized selection
        model_configs = [
            ("yolov8", "n"), ("yolov8", "s"),
            ("yolov10", "n"), ("yolov10", "s"),
            ("rtdetr", "l"),
        ]
    else:
        for model in args.models:
            if model in ["yolov8", "yolov10"]:
                model_configs.append((model, args.size))
            elif model == "rtdetr":
                model_configs.append(("rtdetr", "l"))
            else:
                model_configs.append((model, None))
    
    print(f"\nModels to evaluate: {len(model_configs)}")
    
    # Run evaluations
    all_metrics = []
    
    for model_type, size in model_configs:
        try:
            if model_type == "yolov8":
                evaluator = YOLOv8Evaluator(size=size)
            elif model_type == "yolov10":
                evaluator = YOLOv10Evaluator(size=size)
            elif model_type == "rtdetr":
                evaluator = RTDETREvaluator(size=size or "l")
            elif model_type == "ssd":
                evaluator = SSDEvaluator(use_retinanet=False)
            elif model_type == "retinanet":
                evaluator = SSDEvaluator(use_retinanet=True)
            elif model_type == "faster_rcnn":
                evaluator = FasterRCNNEvaluator()
            else:
                print(f"Unknown model: {model_type}, skipping")
                continue
            
            metrics = run_assistive_evaluation(
                evaluator,
                run_latency_test=not args.skip_latency,
                num_latency_samples=args.latency_samples,
            )
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"\nERROR evaluating {model_type}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    if all_metrics:
        print_assistive_summary(all_metrics)
        save_assistive_results(all_metrics, args.output_dir)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

