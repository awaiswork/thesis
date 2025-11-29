#!/usr/bin/env python3
"""
Results Comparison Script

This script loads saved evaluation results and generates:
- Comparison tables
- Visualization charts
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import RESULTS_DIR
from src.models.base import EvaluationResults
from src.utils.metrics import load_results_json, format_results_table
from src.utils.visualization import plot_comparison, plot_accuracy_vs_speed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare and visualize evaluation results"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_DIR / "evaluation_results.json",
        help="Input JSON file with evaluation results"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Output directory for plots"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (just save them)"
    )
    return parser.parse_args()


def dict_to_results(data: dict) -> EvaluationResults:
    """Convert dictionary to EvaluationResults object."""
    return EvaluationResults(
        model_name=data["model_name"],
        map50=data["map50"],
        map50_95=data["map50_95"],
        precision=data.get("precision"),
        recall=data.get("recall"),
        time_per_image_ms=data.get("time_per_image_ms", 0),
        fps=data.get("fps", 0),
        num_images=data.get("num_images", 0),
        total_time_seconds=data.get("total_time_seconds", 0),
    )


def main():
    """Generate comparison visualizations."""
    args = parse_args()

    print("=" * 60)
    print("Results Comparison and Visualization")
    print("=" * 60)

    # Check input file
    if not args.input.exists():
        print(f"ERROR: Results file not found: {args.input}")
        print("Run evaluate_all.py first to generate results.")
        sys.exit(1)

    # Load results
    print(f"\nLoading results from: {args.input}")
    results_data = load_results_json(args.input)

    if not results_data:
        print("No results found in file.")
        sys.exit(1)

    # Convert to EvaluationResults objects
    results = [dict_to_results(r) for r in results_data]
    print(f"Loaded {len(results)} model results")

    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(format_results_table(results))

    # Generate plots
    print("\nGenerating visualizations...")
    show = not args.no_show

    # Bar chart comparison
    plot_comparison(
        results,
        output_path=args.output_dir / "model_comparison.png",
        show=show
    )

    # Accuracy vs speed scatter plot
    plot_accuracy_vs_speed(
        results,
        output_path=args.output_dir / "accuracy_vs_speed.png",
        show=show
    )

    # Print summary insights
    print("\n" + "=" * 60)
    print("INSIGHTS")
    print("=" * 60)

    # Find best models
    best_accuracy = max(results, key=lambda r: r.map50_95)
    fastest = max(results, key=lambda r: r.fps)
    best_map50 = max(results, key=lambda r: r.map50)

    print(f"Highest mAP@[0.5:0.95]: {best_accuracy.model_name} ({best_accuracy.map50_95:.1f}%)")
    print(f"Highest mAP@0.5:        {best_map50.model_name} ({best_map50.map50:.1f}%)")
    print(f"Fastest (FPS):          {fastest.model_name} ({fastest.fps:.1f} FPS)")

    # Speed comparison
    if len(results) > 1:
        slowest = min(results, key=lambda r: r.fps if r.fps > 0 else float('inf'))
        if slowest.fps > 0:
            speedup = fastest.fps / slowest.fps
            print(f"Speed ratio:            {fastest.model_name} is {speedup:.1f}x faster than {slowest.model_name}")

    print("\nDone!")


if __name__ == "__main__":
    main()


