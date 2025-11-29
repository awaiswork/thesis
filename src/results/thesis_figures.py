"""Publication-ready figure generation for thesis."""

from pathlib import Path
from typing import List, Optional, Literal

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.models.base import EvaluationResults
from src.config import RESULTS_DIR

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Consistent color scheme for models
MODEL_COLORS = {
    'YOLOv8n': '#2ecc71',      # Green
    'YOLOv10n': '#3498db',     # Blue
    'SSD300': '#e74c3c',       # Red
    'Faster R-CNN': '#9b59b6', # Purple
}


def get_color(model_name: str) -> str:
    """Get consistent color for a model."""
    return MODEL_COLORS.get(model_name, '#95a5a6')


class ThesisFigureGenerator:
    """Generate publication-ready figures for thesis."""

    def __init__(
        self,
        results: List[EvaluationResults],
        output_dir: Optional[Path] = None,
        format: Literal['pdf', 'png', 'svg'] = 'pdf'
    ):
        """
        Initialize the figure generator.

        Args:
            results: List of evaluation results.
            output_dir: Directory to save figures.
            format: Output format for figures.
        """
        self.results = results
        self.output_dir = Path(output_dir or RESULTS_DIR / "figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format

    def _save_figure(self, fig: plt.Figure, name: str) -> Path:
        """Save figure to file."""
        path = self.output_dir / f"{name}.{self.format}"
        fig.savefig(path, format=self.format)
        plt.close(fig)
        print(f"Saved: {path}")
        return path

    def plot_map_comparison(self) -> Path:
        """
        Create grouped bar chart comparing mAP@0.5 and mAP@0.5:0.95.

        Returns:
            Path to saved figure.
        """
        models = [r.model_name for r in self.results]
        map50 = [r.map50 for r in self.results]
        map50_95 = [r.map50_95 for r in self.results]

        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))

        bars1 = ax.bar(x - width/2, map50, width, label='mAP@0.5',
                       color=[get_color(m) for m in models], alpha=0.9)
        bars2 = ax.bar(x + width/2, map50_95, width, label='mAP@[0.5:0.95]',
                       color=[get_color(m) for m in models], alpha=0.6,
                       hatch='//')

        ax.set_xlabel('Model')
        ax.set_ylabel('Average Precision (%)')
        ax.set_title('Object Detection Accuracy Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        return self._save_figure(fig, 'map_comparison')

    def plot_pareto_frontier(self) -> Path:
        """
        Create accuracy vs speed scatter plot with Pareto frontier.

        Returns:
            Path to saved figure.
        """
        models = [r.model_name for r in self.results]
        map50_95 = [r.map50_95 for r in self.results]
        fps = [r.fps for r in self.results]

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot each model
        for i, (model, acc, speed) in enumerate(zip(models, map50_95, fps)):
            ax.scatter(speed, acc, s=200, c=get_color(model),
                       edgecolors='black', linewidth=1.5, zorder=5,
                       label=model)

        # Find and plot Pareto frontier
        points = list(zip(fps, map50_95))
        pareto_points = self._get_pareto_frontier(points)
        if len(pareto_points) > 1:
            pareto_x, pareto_y = zip(*sorted(pareto_points))
            ax.plot(pareto_x, pareto_y, 'k--', alpha=0.5, linewidth=1.5,
                    label='Pareto Frontier', zorder=1)

        ax.set_xlabel('Inference Speed (FPS)')
        ax.set_ylabel('mAP@[0.5:0.95] (%)')
        ax.set_title('Accuracy vs Speed Trade-off')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Set reasonable axis limits
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=max(map50_95) * 1.15)

        plt.tight_layout()
        return self._save_figure(fig, 'pareto_frontier')

    def _get_pareto_frontier(self, points: List[tuple]) -> List[tuple]:
        """Find Pareto-optimal points (maximize both x and y)."""
        sorted_points = sorted(points, key=lambda p: p[0], reverse=True)
        pareto = []
        max_y = float('-inf')

        for x, y in sorted_points:
            if y >= max_y:
                pareto.append((x, y))
                max_y = y

        return pareto

    def plot_radar_chart(self) -> Path:
        """
        Create radar/spider chart for multi-metric comparison.

        Returns:
            Path to saved figure.
        """
        # Metrics to include (normalized to 0-100 scale)
        metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1 Score']
        num_metrics = len(metrics)

        # Extract and normalize data
        data = []
        for r in self.results:
            values = [
                r.map50,
                r.map50_95,
                r.precision if r.precision else 0,
                r.recall if r.recall else 0,
                r.f1_score if r.f1_score else 0,
            ]
            data.append(values)

        # Compute angles for radar chart
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for i, r in enumerate(self.results):
            values = data[i] + data[i][:1]  # Complete the loop
            ax.plot(angles, values, 'o-', linewidth=2, label=r.model_name,
                    color=get_color(r.model_name))
            ax.fill(angles, values, alpha=0.15, color=get_color(r.model_name))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 100)
        ax.set_title('Multi-Metric Model Comparison', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        plt.tight_layout()
        return self._save_figure(fig, 'radar_chart')

    def plot_inference_breakdown(self) -> Path:
        """
        Create stacked bar chart showing inference time breakdown.

        Returns:
            Path to saved figure.
        """
        models = [r.model_name for r in self.results]

        # Extract timing breakdown
        preprocess = []
        inference = []
        postprocess = []

        for r in self.results:
            if r.timing_breakdown:
                preprocess.append(r.timing_breakdown.preprocess_ms)
                inference.append(r.timing_breakdown.inference_ms)
                postprocess.append(r.timing_breakdown.postprocess_ms)
            else:
                # Fallback if no breakdown available
                preprocess.append(0)
                inference.append(r.time_per_image_ms)
                postprocess.append(0)

        fig, ax = plt.subplots(figsize=(8, 5))

        x = np.arange(len(models))
        width = 0.6

        # Stacked bars
        ax.bar(x, preprocess, width, label='Preprocessing', color='#3498db')
        ax.bar(x, inference, width, bottom=preprocess, label='Inference', color='#2ecc71')
        ax.bar(x, postprocess, width,
               bottom=np.array(preprocess) + np.array(inference),
               label='Postprocessing', color='#e74c3c')

        ax.set_xlabel('Model')
        ax.set_ylabel('Time per Image (ms)')
        ax.set_title('Inference Time Breakdown')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add total time labels
        totals = np.array(preprocess) + np.array(inference) + np.array(postprocess)
        for i, total in enumerate(totals):
            ax.annotate(f'{total:.1f}ms',
                        xy=(i, total),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        return self._save_figure(fig, 'inference_breakdown')

    def plot_ap_by_size(self) -> Path:
        """
        Create grouped bar chart for AP by object size.

        Returns:
            Path to saved figure.
        """
        models = [r.model_name for r in self.results]

        # Extract AP by size
        ap_small = []
        ap_medium = []
        ap_large = []

        for r in self.results:
            if r.ap_by_size:
                ap_small.append(r.ap_by_size.small)
                ap_medium.append(r.ap_by_size.medium)
                ap_large.append(r.ap_by_size.large)
            else:
                ap_small.append(0)
                ap_medium.append(0)
                ap_large.append(0)

        x = np.arange(len(models))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 5))

        bars1 = ax.bar(x - width, ap_small, width, label='Small (area < 32²)',
                       color='#3498db')
        bars2 = ax.bar(x, ap_medium, width, label='Medium (32² < area < 96²)',
                       color='#2ecc71')
        bars3 = ax.bar(x + width, ap_large, width, label='Large (area > 96²)',
                       color='#e74c3c')

        ax.set_xlabel('Model')
        ax.set_ylabel('Average Precision (%)')
        ax.set_title('Detection Performance by Object Size')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.1f}',
                                xy=(bar.get_x() + bar.get_width()/2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        return self._save_figure(fig, 'ap_by_size')

    def generate_all(self) -> List[Path]:
        """
        Generate all thesis figures.

        Returns:
            List of paths to generated figures.
        """
        print(f"\nGenerating thesis figures in: {self.output_dir}")
        print("-" * 50)

        paths = [
            self.plot_map_comparison(),
            self.plot_pareto_frontier(),
            self.plot_radar_chart(),
            self.plot_inference_breakdown(),
            self.plot_ap_by_size(),
        ]

        print("-" * 50)
        print(f"Generated {len(paths)} figures")

        return paths

