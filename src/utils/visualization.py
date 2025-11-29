"""Visualization utilities for comparing model performance."""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.models.base import EvaluationResults
from src.config import RESULTS_DIR


def plot_comparison(
    results_list: List[EvaluationResults],
    output_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """
    Create bar charts comparing model performance.

    Args:
        results_list: List of EvaluationResults objects.
        output_path: Path to save the plot. If None, uses default.
        show: Whether to display the plot.
    """
    if not results_list:
        print("No results to plot.")
        return

    if output_path is None:
        output_path = RESULTS_DIR / "comparison.png"

    # Extract data
    models = [r.model_name for r in results_list]
    map50 = [r.map50 for r in results_list]
    map50_95 = [r.map50_95 for r in results_list]
    fps = [r.fps for r in results_list]

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Colors for each model
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    # mAP@0.5 comparison
    ax1 = axes[0]
    bars1 = ax1.bar(models, map50, color=colors)
    ax1.set_ylabel('mAP@0.5 (%)')
    ax1.set_title('mAP@0.5 Comparison')
    ax1.set_ylim(0, 100)
    for bar, val in zip(bars1, map50):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # mAP@[0.5:0.95] comparison
    ax2 = axes[1]
    bars2 = ax2.bar(models, map50_95, color=colors)
    ax2.set_ylabel('mAP@[0.5:0.95] (%)')
    ax2.set_title('mAP@[0.5:0.95] Comparison')
    ax2.set_ylim(0, 100)
    for bar, val in zip(bars2, map50_95):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # FPS comparison
    ax3 = axes[2]
    bars3 = ax3.bar(models, fps, color=colors)
    ax3.set_ylabel('FPS')
    ax3.set_title('Speed Comparison (FPS)')
    for bar, val in zip(bars3, fps):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # Rotate x labels if needed
    for ax in axes:
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_accuracy_vs_speed(
    results_list: List[EvaluationResults],
    output_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """
    Create scatter plot of accuracy vs speed trade-off.

    Args:
        results_list: List of EvaluationResults objects.
        output_path: Path to save the plot. If None, uses default.
        show: Whether to display the plot.
    """
    if not results_list:
        print("No results to plot.")
        return

    if output_path is None:
        output_path = RESULTS_DIR / "accuracy_vs_speed.png"

    # Extract data
    models = [r.model_name for r in results_list]
    map50_95 = [r.map50_95 for r in results_list]
    fps = [r.fps for r in results_list]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot with different colors
    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
    scatter = ax.scatter(fps, map50_95, c=colors, s=200, edgecolors='black', linewidth=1.5)

    # Add labels for each point
    for i, model in enumerate(models):
        ax.annotate(
            model,
            (fps[i], map50_95[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=10,
            fontweight='bold'
        )

    ax.set_xlabel('FPS (Frames Per Second)', fontsize=12)
    ax.set_ylabel('mAP@[0.5:0.95] (%)', fontsize=12)
    ax.set_title('Accuracy vs Speed Trade-off', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Accuracy vs speed plot saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


