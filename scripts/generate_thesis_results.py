#!/usr/bin/env python3
"""
Generate publication-quality figures and tables for thesis.

Creates comprehensive visualizations comparing all evaluated models:
- YOLOv8 variants (n, s, m, l, x)
- YOLOv10 variants (n, s, m, l, x)
- RT-DETR-l
- SSD300
- RetinaNet
- Faster R-CNN
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.cm as cm

# Set publication-quality matplotlib settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
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

# Color schemes
YOLOV8_COLORS = ['#1f77b4', '#2980b9', '#3498db', '#5dade2', '#85c1e9']  # Blues
YOLOV10_COLORS = ['#e74c3c', '#c0392b', '#a93226', '#922b21', '#641e16']  # Reds
OTHER_COLORS = {
    'RT-DETR-l': '#27ae60',     # Green
    'SSD300': '#f39c12',        # Orange
    'RetinaNet': '#9b59b6',     # Purple
    'Faster R-CNN': '#1abc9c',  # Teal
}


@dataclass
class ModelResults:
    """Container for model evaluation results."""
    name: str
    family: str
    variant: str
    map50: float
    map50_95: float
    recall: Optional[float]
    precision: Optional[float]
    fps: float
    latency_ms: float
    parameters: int
    gflops: float


def load_results(results_dir: Path) -> Tuple[List[ModelResults], Dict[str, Any]]:
    """Load all model results from JSON files."""
    all_results_file = results_dir / "thesis" / "data" / "all_models_results.json"
    
    if not all_results_file.exists():
        raise FileNotFoundError(f"Results file not found: {all_results_file}")
    
    with open(all_results_file, 'r') as f:
        data = json.load(f)
    
    results = []
    for r in data['results']:
        results.append(ModelResults(
            name=r['model_name'],
            family=r['model_family'],
            variant=r['model_variant'],
            map50=r['map50'],
            map50_95=r['map50_95'],
            recall=r.get('recall'),
            precision=r.get('precision'),
            fps=r['fps'],
            latency_ms=r['latency_ms'],
            parameters=r['parameters'],
            gflops=r['gflops'],
        ))
    
    return results, data.get('summary', {})


def get_model_color(model: ModelResults) -> str:
    """Get color for a model based on its family."""
    if model.family == 'YOLOv8':
        variants = ['n', 's', 'm', 'l', 'x']
        idx = variants.index(model.variant) if model.variant in variants else 0
        return YOLOV8_COLORS[idx]
    elif model.family == 'YOLOv10':
        variants = ['n', 's', 'm', 'l', 'x']
        idx = variants.index(model.variant) if model.variant in variants else 0
        return YOLOV10_COLORS[idx]
    else:
        return OTHER_COLORS.get(model.name, '#7f8c8d')


def create_output_dirs(results_dir: Path) -> Dict[str, Path]:
    """Create output directory structure."""
    dirs = {
        'figures': results_dir / 'thesis' / 'figures',
        'family': results_dir / 'thesis' / 'figures' / 'family',
        'cross_family': results_dir / 'thesis' / 'figures' / 'cross_family',
        'overall': results_dir / 'thesis' / 'figures' / 'overall',
        'tables': results_dir / 'thesis' / 'tables',
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ============================================================================
# Figure Generation Functions
# ============================================================================

def plot_yolov8_variants(results: List[ModelResults], output_dir: Path):
    """Generate YOLOv8 variants comparison chart."""
    yolov8 = [r for r in results if r.family == 'YOLOv8']
    yolov8 = sorted(yolov8, key=lambda x: ['n', 's', 'm', 'l', 'x'].index(x.variant))
    
    if not yolov8:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    names = [r.name for r in yolov8]
    x = np.arange(len(names))
    width = 0.6
    
    # mAP comparison
    ax = axes[0]
    map50 = [r.map50 for r in yolov8]
    map50_95 = [r.map50_95 for r in yolov8]
    
    bars1 = ax.bar(x - 0.2, map50, width/2, label='mAP@0.5', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + 0.2, map50_95, width/2, label='mAP@0.5:0.95', color='#2980b9', alpha=0.8)
    
    ax.set_xlabel('Model Variant')
    ax.set_ylabel('mAP (%)')
    ax.set_title('YOLOv8 Accuracy Scaling')
    ax.set_xticks(x)
    ax.set_xticklabels([r.variant.upper() for r in yolov8])
    ax.legend()
    ax.set_ylim(0, 80)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
    
    # Speed comparison
    ax = axes[1]
    fps = [r.fps for r in yolov8]
    bars = ax.bar(x, fps, width, color=YOLOV8_COLORS, alpha=0.8)
    
    ax.set_xlabel('Model Variant')
    ax.set_ylabel('FPS')
    ax.set_title('YOLOv8 Inference Speed')
    ax.set_xticks(x)
    ax.set_xticklabels([r.variant.upper() for r in yolov8])
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=9)
    
    # Recall comparison
    ax = axes[2]
    recall = [r.recall if r.recall else 0 for r in yolov8]
    bars = ax.bar(x, recall, width, color=YOLOV8_COLORS, alpha=0.8)
    
    ax.set_xlabel('Model Variant')
    ax.set_ylabel('Recall (%)')
    ax.set_title('YOLOv8 Detection Recall')
    ax.set_xticks(x)
    ax.set_xticklabels([r.variant.upper() for r in yolov8])
    ax.set_ylim(0, 80)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'yolov8_variants_comparison.pdf')
    plt.close()
    print(f"Saved: {output_dir / 'yolov8_variants_comparison.pdf'}")


def plot_yolov10_variants(results: List[ModelResults], output_dir: Path):
    """Generate YOLOv10 variants comparison chart."""
    yolov10 = [r for r in results if r.family == 'YOLOv10']
    yolov10 = sorted(yolov10, key=lambda x: ['n', 's', 'm', 'l', 'x'].index(x.variant))
    
    if not yolov10:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    names = [r.name for r in yolov10]
    x = np.arange(len(names))
    width = 0.6
    
    # mAP comparison
    ax = axes[0]
    map50 = [r.map50 for r in yolov10]
    map50_95 = [r.map50_95 for r in yolov10]
    
    bars1 = ax.bar(x - 0.2, map50, width/2, label='mAP@0.5', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + 0.2, map50_95, width/2, label='mAP@0.5:0.95', color='#c0392b', alpha=0.8)
    
    ax.set_xlabel('Model Variant')
    ax.set_ylabel('mAP (%)')
    ax.set_title('YOLOv10 Accuracy Scaling')
    ax.set_xticks(x)
    ax.set_xticklabels([r.variant.upper() for r in yolov10])
    ax.legend()
    ax.set_ylim(0, 80)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
    
    # Speed comparison
    ax = axes[1]
    fps = [r.fps for r in yolov10]
    bars = ax.bar(x, fps, width, color=YOLOV10_COLORS, alpha=0.8)
    
    ax.set_xlabel('Model Variant')
    ax.set_ylabel('FPS')
    ax.set_title('YOLOv10 Inference Speed')
    ax.set_xticks(x)
    ax.set_xticklabels([r.variant.upper() for r in yolov10])
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=9)
    
    # Recall comparison
    ax = axes[2]
    recall = [r.recall if r.recall else 0 for r in yolov10]
    bars = ax.bar(x, recall, width, color=YOLOV10_COLORS, alpha=0.8)
    
    ax.set_xlabel('Model Variant')
    ax.set_ylabel('Recall (%)')
    ax.set_title('YOLOv10 Detection Recall')
    ax.set_xticks(x)
    ax.set_xticklabels([r.variant.upper() for r in yolov10])
    ax.set_ylim(0, 80)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'yolov10_variants_comparison.pdf')
    plt.close()
    print(f"Saved: {output_dir / 'yolov10_variants_comparison.pdf'}")


def plot_yolov8_vs_yolov10(results: List[ModelResults], output_dir: Path):
    """Generate YOLOv8 vs YOLOv10 comparison chart."""
    yolov8 = [r for r in results if r.family == 'YOLOv8']
    yolov10 = [r for r in results if r.family == 'YOLOv10']
    
    yolov8 = sorted(yolov8, key=lambda x: ['n', 's', 'm', 'l', 'x'].index(x.variant))
    yolov10 = sorted(yolov10, key=lambda x: ['n', 's', 'm', 'l', 'x'].index(x.variant))
    
    if not yolov8 or not yolov10:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    variants = ['N', 'S', 'M', 'L', 'X']
    x = np.arange(len(variants))
    width = 0.35
    
    # mAP@0.5 comparison
    ax = axes[0, 0]
    v8_map50 = [r.map50 for r in yolov8]
    v10_map50 = [r.map50 for r in yolov10]
    
    bars1 = ax.bar(x - width/2, v8_map50, width, label='YOLOv8', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, v10_map50, width, label='YOLOv10', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Model Size')
    ax.set_ylabel('mAP@0.5 (%)')
    ax.set_title('mAP@0.5 Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.legend()
    ax.set_ylim(0, 80)
    ax.grid(axis='y', alpha=0.3)
    
    # mAP@0.5:0.95 comparison
    ax = axes[0, 1]
    v8_map = [r.map50_95 for r in yolov8]
    v10_map = [r.map50_95 for r in yolov10]
    
    bars1 = ax.bar(x - width/2, v8_map, width, label='YOLOv8', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, v10_map, width, label='YOLOv10', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Model Size')
    ax.set_ylabel('mAP@0.5:0.95 (%)')
    ax.set_title('mAP@0.5:0.95 Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.legend()
    ax.set_ylim(0, 60)
    ax.grid(axis='y', alpha=0.3)
    
    # FPS comparison
    ax = axes[1, 0]
    v8_fps = [r.fps for r in yolov8]
    v10_fps = [r.fps for r in yolov10]
    
    bars1 = ax.bar(x - width/2, v8_fps, width, label='YOLOv8', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, v10_fps, width, label='YOLOv10', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Model Size')
    ax.set_ylabel('FPS')
    ax.set_title('Inference Speed Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Recall comparison
    ax = axes[1, 1]
    v8_recall = [r.recall if r.recall else 0 for r in yolov8]
    v10_recall = [r.recall if r.recall else 0 for r in yolov10]
    
    bars1 = ax.bar(x - width/2, v8_recall, width, label='YOLOv8', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, v10_recall, width, label='YOLOv10', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Model Size')
    ax.set_ylabel('Recall (%)')
    ax.set_title('Detection Recall Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.legend()
    ax.set_ylim(0, 80)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'yolov8_vs_yolov10.pdf')
    plt.close()
    print(f"Saved: {output_dir / 'yolov8_vs_yolov10.pdf'}")


def plot_all_models_map(results: List[ModelResults], output_dir: Path):
    """Generate all models mAP comparison chart."""
    # Sort by mAP@0.5:0.95
    sorted_results = sorted(results, key=lambda x: x.map50_95, reverse=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    names = [r.name for r in sorted_results]
    map50 = [r.map50 for r in sorted_results]
    map50_95 = [r.map50_95 for r in sorted_results]
    colors = [get_model_color(r) for r in sorted_results]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, map50, width, label='mAP@0.5', alpha=0.8, color=colors)
    bars2 = ax.bar(x + width/2, map50_95, width, label='mAP@0.5:0.95', alpha=0.6, color=colors)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('mAP (%)')
    ax.set_title('Detection Accuracy Comparison - All Models')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 80)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_models_map_comparison.pdf')
    plt.close()
    print(f"Saved: {output_dir / 'all_models_map_comparison.pdf'}")


def plot_all_models_speed(results: List[ModelResults], output_dir: Path):
    """Generate all models speed comparison chart."""
    # Sort by FPS
    sorted_results = sorted(results, key=lambda x: x.fps, reverse=True)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    names = [r.name for r in sorted_results]
    fps = [r.fps for r in sorted_results]
    colors = [get_model_color(r) for r in sorted_results]
    
    bars = ax.barh(names, fps, color=colors, alpha=0.8)
    
    ax.set_xlabel('FPS (Frames Per Second)')
    ax.set_ylabel('Model')
    ax.set_title('Inference Speed Comparison - All Models')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                f'{bar.get_width():.0f}', ha='left', va='center', fontsize=9)
    
    # Add legend
    legend_patches = [
        mpatches.Patch(color='#3498db', label='YOLOv8'),
        mpatches.Patch(color='#e74c3c', label='YOLOv10'),
        mpatches.Patch(color='#27ae60', label='RT-DETR'),
        mpatches.Patch(color='#f39c12', label='SSD'),
        mpatches.Patch(color='#9b59b6', label='RetinaNet'),
        mpatches.Patch(color='#1abc9c', label='Faster R-CNN'),
    ]
    ax.legend(handles=legend_patches, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_models_speed_comparison.pdf')
    plt.close()
    print(f"Saved: {output_dir / 'all_models_speed_comparison.pdf'}")


def plot_pareto_frontier(results: List[ModelResults], output_dir: Path):
    """Generate Pareto frontier plot (accuracy vs speed)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for r in results:
        color = get_model_color(r)
        marker = 'o' if 'YOLO' in r.family else 's'
        ax.scatter(r.fps, r.map50_95, c=color, s=150, alpha=0.8, marker=marker, 
                   edgecolors='white', linewidths=1.5)
        
        # Add label
        offset = (5, 5)
        if r.fps > 350:
            offset = (-50, 5)
        ax.annotate(r.name, (r.fps, r.map50_95), textcoords='offset points', 
                    xytext=offset, fontsize=8, alpha=0.9)
    
    ax.set_xlabel('FPS (Frames Per Second)')
    ax.set_ylabel('mAP@0.5:0.95 (%)')
    ax.set_title('Accuracy vs Speed Trade-off (Pareto Frontier)')
    ax.grid(alpha=0.3)
    
    # Add reference lines for real-time thresholds
    ax.axvline(x=30, color='gray', linestyle='--', alpha=0.5, label='30 FPS (Real-time)')
    ax.axvline(x=60, color='gray', linestyle=':', alpha=0.5, label='60 FPS')
    
    # Add legend
    legend_patches = [
        mpatches.Patch(color='#3498db', label='YOLOv8'),
        mpatches.Patch(color='#e74c3c', label='YOLOv10'),
        mpatches.Patch(color='#27ae60', label='RT-DETR'),
        mpatches.Patch(color='#f39c12', label='SSD'),
        mpatches.Patch(color='#9b59b6', label='RetinaNet'),
        mpatches.Patch(color='#1abc9c', label='Faster R-CNN'),
    ]
    ax.legend(handles=legend_patches, loc='lower right')
    
    ax.set_xlim(0, max(r.fps for r in results) * 1.15)
    ax.set_ylim(20, 60)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_frontier.pdf')
    plt.close()
    print(f"Saved: {output_dir / 'pareto_frontier.pdf'}")


def plot_model_scaling(results: List[ModelResults], output_dir: Path):
    """Generate model scaling analysis chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # YOLOv8 scaling
    ax = axes[0]
    yolov8 = [r for r in results if r.family == 'YOLOv8']
    yolov8 = sorted(yolov8, key=lambda x: x.parameters)
    
    params = [r.parameters / 1e6 for r in yolov8]
    map50_95 = [r.map50_95 for r in yolov8]
    fps = [r.fps for r in yolov8]
    
    ax2 = ax.twinx()
    
    line1 = ax.plot(params, map50_95, 'o-', color='#3498db', linewidth=2, markersize=10, label='mAP@0.5:0.95')
    line2 = ax2.plot(params, fps, 's--', color='#e74c3c', linewidth=2, markersize=10, label='FPS')
    
    ax.set_xlabel('Parameters (M)')
    ax.set_ylabel('mAP@0.5:0.95 (%)', color='#3498db')
    ax2.set_ylabel('FPS', color='#e74c3c')
    ax.set_title('YOLOv8 Model Scaling')
    ax.grid(alpha=0.3)
    
    # Add variant labels
    for i, r in enumerate(yolov8):
        ax.annotate(r.variant.upper(), (params[i], map50_95[i]), textcoords='offset points', 
                    xytext=(0, 10), ha='center', fontsize=9)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right')
    
    # YOLOv10 scaling
    ax = axes[1]
    yolov10 = [r for r in results if r.family == 'YOLOv10']
    yolov10 = sorted(yolov10, key=lambda x: ['n', 's', 'm', 'l', 'x'].index(x.variant))
    
    # Use variant order for x-axis
    x = np.arange(len(yolov10))
    map50_95 = [r.map50_95 for r in yolov10]
    fps = [r.fps for r in yolov10]
    
    ax2 = ax.twinx()
    
    line1 = ax.plot(x, map50_95, 'o-', color='#e74c3c', linewidth=2, markersize=10, label='mAP@0.5:0.95')
    line2 = ax2.plot(x, fps, 's--', color='#3498db', linewidth=2, markersize=10, label='FPS')
    
    ax.set_xlabel('Model Variant')
    ax.set_ylabel('mAP@0.5:0.95 (%)', color='#e74c3c')
    ax2.set_ylabel('FPS', color='#3498db')
    ax.set_title('YOLOv10 Model Scaling')
    ax.set_xticks(x)
    ax.set_xticklabels([r.variant.upper() for r in yolov10])
    ax.grid(alpha=0.3)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_scaling_analysis.pdf')
    plt.close()
    print(f"Saved: {output_dir / 'model_scaling_analysis.pdf'}")


def plot_latency_comparison(results: List[ModelResults], output_dir: Path):
    """Generate latency comparison chart."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Sort by latency
    sorted_results = sorted(results, key=lambda x: x.latency_ms)
    
    names = [r.name for r in sorted_results]
    latency = [r.latency_ms for r in sorted_results]
    colors = [get_model_color(r) for r in sorted_results]
    
    bars = ax.barh(names, latency, color=colors, alpha=0.8)
    
    # Add real-time threshold lines
    ax.axvline(x=33.3, color='green', linestyle='--', alpha=0.7, label='30 FPS (33.3ms)')
    ax.axvline(x=16.7, color='blue', linestyle=':', alpha=0.7, label='60 FPS (16.7ms)')
    ax.axvline(x=50, color='orange', linestyle='-.', alpha=0.7, label='20 FPS (50ms)')
    
    ax.set_xlabel('Latency (ms per image)')
    ax.set_ylabel('Model')
    ax.set_title('Inference Latency Comparison')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                f'{bar.get_width():.1f}ms', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_comparison.pdf')
    plt.close()
    print(f"Saved: {output_dir / 'latency_comparison.pdf'}")


def plot_recall_comparison(results: List[ModelResults], output_dir: Path):
    """Generate recall comparison chart - critical for assistive technology."""
    # Filter models with recall data
    with_recall = [r for r in results if r.recall is not None and r.recall > 0]
    
    if not with_recall:
        print("No recall data available")
        return
    
    # Sort by recall
    sorted_results = sorted(with_recall, key=lambda x: x.recall, reverse=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    names = [r.name for r in sorted_results]
    recall = [r.recall for r in sorted_results]
    colors = [get_model_color(r) for r in sorted_results]
    
    bars = ax.barh(names, recall, color=colors, alpha=0.8)
    
    # Add threshold lines
    ax.axvline(x=60, color='orange', linestyle='--', alpha=0.7, label='60% Recall')
    ax.axvline(x=65, color='green', linestyle=':', alpha=0.7, label='65% Recall (Target)')
    
    ax.set_xlabel('Recall (%)')
    ax.set_ylabel('Model')
    ax.set_title('Detection Recall Comparison\n(Critical for Visually Impaired Assistance)')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 80)
    
    # Add value labels
    for bar in bars:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{bar.get_width():.1f}%', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'recall_comparison.pdf')
    plt.close()
    print(f"Saved: {output_dir / 'recall_comparison.pdf'}")


def plot_radar_charts(results: List[ModelResults], output_dir: Path):
    """Generate radar charts for multi-metric comparison."""
    # Select representative models
    representative_models = [
        'YOLOv8n', 'YOLOv8x', 'YOLOv10n', 'YOLOv10x', 
        'RT-DETR-l', 'Faster R-CNN', 'RetinaNet', 'SSD300'
    ]
    
    selected = [r for r in results if r.name in representative_models]
    
    if len(selected) < 3:
        selected = results[:8]  # Take first 8 if not enough matches
    
    # Normalize metrics to 0-1 scale
    max_map = max(r.map50_95 for r in selected)
    max_fps = max(r.fps for r in selected)
    max_recall = max(r.recall if r.recall else 0 for r in selected)
    max_precision = max(r.precision if r.precision else 0 for r in selected)
    
    categories = ['mAP@0.5:0.95', 'Speed (FPS)', 'Recall', 'Precision', 'mAP@0.5']
    num_vars = len(categories)
    
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for r in selected:
        values = [
            r.map50_95 / max_map if max_map > 0 else 0,
            r.fps / max_fps if max_fps > 0 else 0,
            (r.recall / max_recall) if (r.recall and max_recall > 0) else 0,
            (r.precision / max_precision) if (r.precision and max_precision > 0) else 0,
            r.map50 / 80,  # Normalize to reasonable max
        ]
        values += values[:1]  # Complete the loop
        
        color = get_model_color(r)
        ax.plot(angles, values, 'o-', linewidth=2, label=r.name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Multi-Metric Model Comparison (Normalized)', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'radar_charts.pdf')
    plt.close()
    print(f"Saved: {output_dir / 'radar_charts.pdf'}")


# ============================================================================
# Table Generation Functions
# ============================================================================

def generate_yolov8_table(results: List[ModelResults], output_dir: Path):
    """Generate LaTeX table for YOLOv8 results."""
    yolov8 = [r for r in results if r.family == 'YOLOv8']
    yolov8 = sorted(yolov8, key=lambda x: ['n', 's', 'm', 'l', 'x'].index(x.variant))
    
    latex = r"""
\begin{table}[htbp]
\centering
\caption{YOLOv8 Model Variants Performance Comparison}
\label{tab:yolov8_results}
\begin{tabular}{lccccccc}
\toprule
\textbf{Model} & \textbf{Params (M)} & \textbf{GFLOPs} & \textbf{mAP@0.5} & \textbf{mAP@0.5:0.95} & \textbf{Recall} & \textbf{Precision} & \textbf{FPS} \\
\midrule
"""
    
    for r in yolov8:
        params = f"{r.parameters / 1e6:.1f}" if r.parameters else "N/A"
        gflops = f"{r.gflops:.1f}" if r.gflops else "N/A"
        recall = f"{r.recall:.1f}" if r.recall else "N/A"
        precision = f"{r.precision:.1f}" if r.precision else "N/A"
        latex += f"{r.name} & {params} & {gflops} & {r.map50:.1f} & {r.map50_95:.1f} & {recall} & {precision} & {r.fps:.1f} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_dir / 'yolov8_results.tex', 'w') as f:
        f.write(latex)
    print(f"Saved: {output_dir / 'yolov8_results.tex'}")


def generate_yolov10_table(results: List[ModelResults], output_dir: Path):
    """Generate LaTeX table for YOLOv10 results."""
    yolov10 = [r for r in results if r.family == 'YOLOv10']
    yolov10 = sorted(yolov10, key=lambda x: ['n', 's', 'm', 'l', 'x'].index(x.variant))
    
    latex = r"""
\begin{table}[htbp]
\centering
\caption{YOLOv10 Model Variants Performance Comparison}
\label{tab:yolov10_results}
\begin{tabular}{lccccccc}
\toprule
\textbf{Model} & \textbf{Params (M)} & \textbf{GFLOPs} & \textbf{mAP@0.5} & \textbf{mAP@0.5:0.95} & \textbf{Recall} & \textbf{Precision} & \textbf{FPS} \\
\midrule
"""
    
    for r in yolov10:
        params = f"{r.parameters / 1e6:.1f}" if r.parameters else "N/A"
        gflops = f"{r.gflops:.1f}" if r.gflops else "N/A"
        recall = f"{r.recall:.1f}" if r.recall else "N/A"
        precision = f"{r.precision:.1f}" if r.precision else "N/A"
        latex += f"{r.name} & {params} & {gflops} & {r.map50:.1f} & {r.map50_95:.1f} & {recall} & {precision} & {r.fps:.1f} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_dir / 'yolov10_results.tex', 'w') as f:
        f.write(latex)
    print(f"Saved: {output_dir / 'yolov10_results.tex'}")


def generate_yolo_comparison_table(results: List[ModelResults], output_dir: Path):
    """Generate LaTeX table comparing YOLOv8 vs YOLOv10."""
    yolov8 = [r for r in results if r.family == 'YOLOv8']
    yolov10 = [r for r in results if r.family == 'YOLOv10']
    
    yolov8 = sorted(yolov8, key=lambda x: ['n', 's', 'm', 'l', 'x'].index(x.variant))
    yolov10 = sorted(yolov10, key=lambda x: ['n', 's', 'm', 'l', 'x'].index(x.variant))
    
    latex = r"""
\begin{table}[htbp]
\centering
\caption{YOLOv8 vs YOLOv10 Direct Comparison}
\label{tab:yolo_comparison}
\begin{tabular}{lcccccc}
\toprule
\textbf{Size} & \multicolumn{2}{c}{\textbf{mAP@0.5:0.95}} & \multicolumn{2}{c}{\textbf{Recall (\%)}} & \multicolumn{2}{c}{\textbf{FPS}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
 & YOLOv8 & YOLOv10 & YOLOv8 & YOLOv10 & YOLOv8 & YOLOv10 \\
\midrule
"""
    
    for v8, v10 in zip(yolov8, yolov10):
        v8_recall = f"{v8.recall:.1f}" if v8.recall else "N/A"
        v10_recall = f"{v10.recall:.1f}" if v10.recall else "N/A"
        latex += f"{v8.variant.upper()} & {v8.map50_95:.1f} & {v10.map50_95:.1f} & {v8_recall} & {v10_recall} & {v8.fps:.0f} & {v10.fps:.0f} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_dir / 'yolo_comparison.tex', 'w') as f:
        f.write(latex)
    print(f"Saved: {output_dir / 'yolo_comparison.tex'}")


def generate_all_models_table(results: List[ModelResults], output_dir: Path):
    """Generate comprehensive LaTeX table for all models."""
    # Sort by mAP@0.5:0.95
    sorted_results = sorted(results, key=lambda x: x.map50_95, reverse=True)
    
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Comprehensive Object Detection Model Comparison}
\label{tab:all_models_results}
\begin{tabular}{llcccccc}
\toprule
\textbf{Model} & \textbf{Family} & \textbf{mAP@0.5} & \textbf{mAP@0.5:0.95} & \textbf{Recall} & \textbf{Precision} & \textbf{FPS} & \textbf{Latency (ms)} \\
\midrule
"""
    
    for r in sorted_results:
        recall = f"{r.recall:.1f}" if r.recall else "N/A"
        precision = f"{r.precision:.1f}" if r.precision else "N/A"
        latex += f"{r.name} & {r.family} & {r.map50:.1f} & {r.map50_95:.1f} & {recall} & {precision} & {r.fps:.1f} & {r.latency_ms:.1f} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_dir / 'all_models_results.tex', 'w') as f:
        f.write(latex)
    print(f"Saved: {output_dir / 'all_models_results.tex'}")


def generate_model_complexity_table(results: List[ModelResults], output_dir: Path):
    """Generate LaTeX table for model complexity analysis."""
    # Sort by parameters
    sorted_results = sorted(results, key=lambda x: x.parameters if x.parameters else 0)
    
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Model Complexity Analysis}
\label{tab:model_complexity}
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{Parameters (M)} & \textbf{GFLOPs} & \textbf{mAP@0.5:0.95} & \textbf{FPS} & \textbf{Efficiency*} \\
\midrule
"""
    
    for r in sorted_results:
        params = r.parameters / 1e6 if r.parameters else 0
        efficiency = r.map50_95 / params if params > 0 else 0
        params_str = f"{params:.1f}" if params > 0 else "N/A"
        gflops_str = f"{r.gflops:.1f}" if r.gflops else "N/A"
        latex += f"{r.name} & {params_str} & {gflops_str} & {r.map50_95:.1f} & {r.fps:.1f} & {efficiency:.2f} \\\\\n"
    
    latex += r"""
\bottomrule
\multicolumn{6}{l}{\footnotesize *Efficiency = mAP@0.5:0.95 / Parameters (M)}
\end{tabular}
\end{table}
"""
    
    with open(output_dir / 'model_complexity.tex', 'w') as f:
        f.write(latex)
    print(f"Saved: {output_dir / 'model_complexity.tex'}")


def generate_speed_analysis_table(results: List[ModelResults], output_dir: Path):
    """Generate LaTeX table for speed analysis."""
    # Sort by FPS
    sorted_results = sorted(results, key=lambda x: x.fps, reverse=True)
    
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Inference Speed Analysis}
\label{tab:speed_analysis}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{FPS} & \textbf{Latency (ms)} & \textbf{Real-time*} & \textbf{Mobile Feasible†} \\
\midrule
"""
    
    for r in sorted_results:
        realtime = "Yes" if r.fps >= 30 else "No"
        mobile = "Yes" if r.fps >= 60 else ("Maybe" if r.fps >= 30 else "No")
        latex += f"{r.name} & {r.fps:.1f} & {r.latency_ms:.1f} & {realtime} & {mobile} \\\\\n"
    
    latex += r"""
\bottomrule
\multicolumn{5}{l}{\footnotesize *Real-time: $\geq$30 FPS; †Mobile Feasible: $\geq$60 FPS on GPU (scaled for mobile)}
\end{tabular}
\end{table}
"""
    
    with open(output_dir / 'speed_analysis.tex', 'w') as f:
        f.write(latex)
    print(f"Saved: {output_dir / 'speed_analysis.tex'}")


def main():
    """Main function to generate all thesis figures and tables."""
    results_dir = project_root / 'results'
    
    print("=" * 70)
    print("GENERATING THESIS FIGURES AND TABLES")
    print("=" * 70)
    
    # Load results
    print("\nLoading evaluation results...")
    results, summary = load_results(results_dir)
    print(f"Loaded {len(results)} model results")
    
    # Create output directories
    dirs = create_output_dirs(results_dir)
    
    # Generate figures
    print("\n" + "-" * 50)
    print("Generating Figures")
    print("-" * 50)
    
    # Family comparisons
    print("\n[Family Comparisons]")
    plot_yolov8_variants(results, dirs['family'])
    plot_yolov10_variants(results, dirs['family'])
    
    # Cross-family comparisons
    print("\n[Cross-Family Comparisons]")
    plot_yolov8_vs_yolov10(results, dirs['cross_family'])
    
    # Overall analysis
    print("\n[Overall Analysis]")
    plot_all_models_map(results, dirs['overall'])
    plot_all_models_speed(results, dirs['overall'])
    plot_pareto_frontier(results, dirs['overall'])
    plot_model_scaling(results, dirs['overall'])
    plot_latency_comparison(results, dirs['overall'])
    plot_recall_comparison(results, dirs['overall'])
    plot_radar_charts(results, dirs['overall'])
    
    # Generate tables
    print("\n" + "-" * 50)
    print("Generating LaTeX Tables")
    print("-" * 50)
    
    generate_yolov8_table(results, dirs['tables'])
    generate_yolov10_table(results, dirs['tables'])
    generate_yolo_comparison_table(results, dirs['tables'])
    generate_all_models_table(results, dirs['tables'])
    generate_model_complexity_table(results, dirs['tables'])
    generate_speed_analysis_table(results, dirs['tables'])
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nFigures saved to: {dirs['figures']}")
    print(f"Tables saved to: {dirs['tables']}")
    
    # Print summary
    print("\n" + "-" * 50)
    print("Generated Files Summary")
    print("-" * 50)
    
    figure_count = sum(1 for _ in dirs['figures'].rglob('*.pdf'))
    table_count = sum(1 for _ in dirs['tables'].glob('*.tex'))
    
    print(f"Total PDF figures: {figure_count}")
    print(f"Total LaTeX tables: {table_count}")


if __name__ == "__main__":
    main()

