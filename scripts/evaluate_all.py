#!/usr/bin/env python3
"""
Model Evaluation Script

This script runs evaluation for all object detection models:
- YOLOv8n
- YOLOv10n
- SSD300
- Faster R-CNN

Results are saved to the results directory in JSON and CSV formats.
After evaluation, run generate_thesis_figures.py to create plots and tables.
"""

import sys
import argparse
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import RESULTS_DIR
from src.models.base import EvaluationResults
from src.models.yolov8 import YOLOv8Evaluator
from src.models.yolov10 import YOLOv10Evaluator
from src.models.ssd import SSDEvaluator
from src.models.faster_rcnn import FasterRCNNEvaluator
from src.results import ResultsManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate object detection models on COCO subset"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["yolov8", "yolov10", "ssd", "faster_rcnn", "all"],
        default=["all"],
        help="Models to evaluate (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Output directory for results"
    )
    parser.add_argument(
        "--generate-figures",
        action="store_true",
        help="Automatically generate thesis figures after evaluation"
    )
    return parser.parse_args()


def get_evaluators(model_names: List[str]) -> List:
    """
    Get evaluator instances for specified models.

    Args:
        model_names: List of model names to evaluate.

    Returns:
        List of evaluator instances.
    """
    evaluator_map = {
        "yolov8": YOLOv8Evaluator,
        "yolov10": YOLOv10Evaluator,
        "ssd": SSDEvaluator,
        "faster_rcnn": FasterRCNNEvaluator,
    }

    if "all" in model_names:
        model_names = list(evaluator_map.keys())

    evaluators = []
    for name in model_names:
        if name in evaluator_map:
            evaluators.append(evaluator_map[name]())
        else:
            print(f"WARNING: Unknown model '{name}', skipping.")

    return evaluators


def main():
    """Run model evaluations."""
    args = parse_args()

    print("=" * 70)
    print("OBJECT DETECTION MODEL EVALUATION")
    print("=" * 70)

    # Initialize results manager
    manager = ResultsManager(args.output_dir)
    manager.set_metadata(
        dataset_name="COCO val2017 subset",
        num_images=5000,
    )

    # Get evaluators
    evaluators = get_evaluators(args.models)

    if not evaluators:
        print("No models to evaluate.")
        sys.exit(1)

    print(f"\nModels to evaluate: {[e.model_name for e in evaluators]}")
    print(f"Output directory: {args.output_dir}")

    # Run evaluations
    for evaluator in evaluators:
        try:
            result = evaluator.run()
            manager.add_result(result)
        except Exception as e:
            print(f"\nERROR evaluating {evaluator.model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save and display results
    if manager.results:
        # Print summary
        manager.print_summary()

        # Save results in all formats
        print("\n" + "=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)
        saved = manager.save_all()
        for fmt, path in saved.items():
            print(f"  {fmt.upper()}: {path}")

        # Optionally generate figures
        if args.generate_figures:
            print("\n" + "=" * 70)
            print("GENERATING THESIS FIGURES")
            print("=" * 70)

            from src.results import ThesisFigureGenerator, LaTeXTableGenerator

            fig_gen = ThesisFigureGenerator(
                results=manager.results,
                output_dir=args.output_dir / "figures",
                format="pdf"
            )
            fig_gen.generate_all()

            table_gen = LaTeXTableGenerator(
                results=manager.results,
                output_dir=args.output_dir / "tables"
            )
            table_gen.generate_all()

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    if not args.generate_figures and manager.results:
        print("\nTo generate thesis figures and tables, run:")
        print("  python scripts/generate_thesis_figures.py")


if __name__ == "__main__":
    main()
