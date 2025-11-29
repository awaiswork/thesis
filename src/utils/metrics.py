"""Metrics extraction and formatting utilities."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.models.base import EvaluationResults


def extract_metrics(results: EvaluationResults) -> Dict[str, Any]:
    """
    Extract metrics from evaluation results as a dictionary.

    Args:
        results: EvaluationResults object.

    Returns:
        Dictionary of metrics.
    """
    return results.to_dict()


def format_results(results: EvaluationResults) -> str:
    """
    Format evaluation results as a human-readable string.

    Args:
        results: EvaluationResults object.

    Returns:
        Formatted string.
    """
    return str(results)


def format_results_table(results_list: List[EvaluationResults]) -> str:
    """
    Format multiple evaluation results as a comparison table.

    Args:
        results_list: List of EvaluationResults objects.

    Returns:
        Formatted table string.
    """
    if not results_list:
        return "No results to display."

    # Header
    header = (
        f"{'Model':<15} {'mAP@0.5':>10} {'mAP@0.5:0.95':>14} "
        f"{'Precision':>10} {'Recall':>10} {'ms/img':>10} {'FPS':>8}"
    )
    separator = "-" * len(header)

    lines = [separator, header, separator]

    for r in results_list:
        prec = f"{r.precision:.1f}%" if r.precision is not None else "N/A"
        rec = f"{r.recall:.1f}%" if r.recall is not None else "N/A"
        line = (
            f"{r.model_name:<15} {r.map50:>9.1f}% {r.map50_95:>13.1f}% "
            f"{prec:>10} {rec:>10} {r.time_per_image_ms:>9.1f} {r.fps:>8.1f}"
        )
        lines.append(line)

    lines.append(separator)
    return "\n".join(lines)


def save_results_json(
    results_list: List[EvaluationResults],
    output_path: Path
) -> None:
    """
    Save evaluation results to a JSON file.

    Args:
        results_list: List of EvaluationResults objects.
        output_path: Path to save the JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "results": [r.to_dict() for r in results_list]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to: {output_path}")


def load_results_json(input_path: Path) -> List[Dict[str, Any]]:
    """
    Load evaluation results from a JSON file.

    Args:
        input_path: Path to the JSON file.

    Returns:
        List of result dictionaries.
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    return data.get("results", [])


def calculate_speedup(
    baseline_fps: float,
    comparison_fps: float
) -> float:
    """
    Calculate speedup factor between two models.

    Args:
        baseline_fps: FPS of the baseline model.
        comparison_fps: FPS of the comparison model.

    Returns:
        Speedup factor (comparison_fps / baseline_fps).
    """
    if baseline_fps <= 0:
        return 0.0
    return comparison_fps / baseline_fps


