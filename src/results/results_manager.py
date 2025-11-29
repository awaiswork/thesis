"""Central results storage and I/O management."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import RESULTS_DIR, NUM_SAMPLES
from src.models.base import EvaluationResults


class ResultsManager:
    """
    Central manager for evaluation results.

    Handles saving, loading, and exporting results in multiple formats
    with full metadata for reproducibility.
    """

    def __init__(self, results_dir: Optional[Path] = None):
        """
        Initialize the results manager.

        Args:
            results_dir: Directory for storing results. Defaults to config value.
        """
        self.results_dir = Path(results_dir or RESULTS_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.figures_dir = self.results_dir / "figures"
        self.tables_dir = self.results_dir / "tables"
        self.figures_dir.mkdir(exist_ok=True)
        self.tables_dir.mkdir(exist_ok=True)

        # File paths
        self.json_path = self.results_dir / "evaluation_results.json"
        self.csv_path = self.results_dir / "evaluation_results.csv"

        # In-memory results storage
        self._results: List[EvaluationResults] = []
        self._metadata: Dict[str, Any] = {}

    @property
    def results(self) -> List[EvaluationResults]:
        """Get current results list."""
        return self._results

    def add_result(self, result: EvaluationResults) -> None:
        """
        Add a single evaluation result.

        Args:
            result: EvaluationResults to add.
        """
        self._results.append(result)

    def add_results(self, results: List[EvaluationResults]) -> None:
        """
        Add multiple evaluation results.

        Args:
            results: List of EvaluationResults to add.
        """
        self._results.extend(results)

    def clear(self) -> None:
        """Clear all stored results."""
        self._results = []
        self._metadata = {}

    def set_metadata(
        self,
        dataset_name: str = "COCO val2017 subset",
        num_images: int = NUM_SAMPLES,
        **kwargs
    ) -> None:
        """
        Set evaluation metadata.

        Args:
            dataset_name: Name of the dataset used.
            num_images: Number of images evaluated.
            **kwargs: Additional metadata fields.
        """
        self._metadata = {
            "dataset_name": dataset_name,
            "num_images": num_images,
            "evaluation_date": datetime.now().isoformat(),
            **kwargs
        }

    def save_json(self, path: Optional[Path] = None) -> Path:
        """
        Save results to JSON with full metadata.

        Args:
            path: Output path. Defaults to standard location.

        Returns:
            Path to saved file.
        """
        path = Path(path or self.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": self._metadata,
            "results": [r.to_dict() for r in self._results],
            "summary": self._generate_summary(),
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to JSON: {path}")
        return path

    def save_csv(self, path: Optional[Path] = None) -> Path:
        """
        Save results to CSV for easy import into LaTeX/Excel.

        Args:
            path: Output path. Defaults to standard location.

        Returns:
            Path to saved file.
        """
        path = Path(path or self.csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self._results:
            print("No results to save.")
            return path

        # Get all fields from flat dict
        flat_results = [r.to_flat_dict() for r in self._results]
        fieldnames = list(flat_results[0].keys())

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_results)

        print(f"Results saved to CSV: {path}")
        return path

    def save_all(self) -> Dict[str, Path]:
        """
        Save results in all formats.

        Returns:
            Dictionary mapping format name to saved path.
        """
        return {
            "json": self.save_json(),
            "csv": self.save_csv(),
        }

    def load_json(self, path: Optional[Path] = None) -> List[EvaluationResults]:
        """
        Load results from JSON file.

        Args:
            path: Input path. Defaults to standard location.

        Returns:
            List of loaded EvaluationResults.
        """
        path = Path(path or self.json_path)

        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        self._metadata = data.get("metadata", {})
        self._results = [
            EvaluationResults.from_dict(r)
            for r in data.get("results", [])
        ]

        print(f"Loaded {len(self._results)} results from: {path}")
        return self._results

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        if not self._results:
            return {}

        # Find best performers
        best_accuracy = max(self._results, key=lambda r: r.map50_95)
        best_map50 = max(self._results, key=lambda r: r.map50)
        fastest = max(self._results, key=lambda r: r.fps)
        slowest = min(self._results, key=lambda r: r.fps if r.fps > 0 else float('inf'))

        summary = {
            "num_models": len(self._results),
            "best_map50_95": {
                "model": best_accuracy.model_name,
                "value": best_accuracy.map50_95,
            },
            "best_map50": {
                "model": best_map50.model_name,
                "value": best_map50.map50,
            },
            "fastest": {
                "model": fastest.model_name,
                "fps": fastest.fps,
            },
            "slowest": {
                "model": slowest.model_name,
                "fps": slowest.fps,
            },
        }

        if slowest.fps > 0:
            summary["speed_ratio"] = fastest.fps / slowest.fps

        return summary

    def get_comparison_data(self) -> Dict[str, List[Any]]:
        """
        Get data structured for comparison plots.

        Returns:
            Dictionary with model names and metric lists.
        """
        return {
            "models": [r.model_name for r in self._results],
            "map50": [r.map50 for r in self._results],
            "map50_95": [r.map50_95 for r in self._results],
            "map75": [r.map75 for r in self._results],
            "precision": [r.precision for r in self._results],
            "recall": [r.recall for r in self._results],
            "f1_score": [r.f1_score for r in self._results],
            "fps": [r.fps for r in self._results],
            "time_per_image_ms": [r.time_per_image_ms for r in self._results],
        }

    def print_summary(self) -> None:
        """Print a formatted summary of results."""
        if not self._results:
            print("No results to summarize.")
            return

        print("\n" + "=" * 70)
        print("EVALUATION RESULTS SUMMARY")
        print("=" * 70)

        # Header
        header = (
            f"{'Model':<15} {'mAP@0.5':>10} {'mAP@0.5:0.95':>14} "
            f"{'Precision':>10} {'Recall':>10} {'F1':>8} {'FPS':>8}"
        )
        print("-" * 70)
        print(header)
        print("-" * 70)

        for r in self._results:
            prec = f"{r.precision:.1f}%" if r.precision else "N/A"
            rec = f"{r.recall:.1f}%" if r.recall else "N/A"
            f1 = f"{r.f1_score:.1f}%" if r.f1_score else "N/A"
            print(
                f"{r.model_name:<15} {r.map50:>9.1f}% {r.map50_95:>13.1f}% "
                f"{prec:>10} {rec:>10} {f1:>8} {r.fps:>7.1f}"
            )

        print("-" * 70)

        # Summary insights
        summary = self._generate_summary()
        print(f"\nBest mAP@[0.5:0.95]: {summary['best_map50_95']['model']} "
              f"({summary['best_map50_95']['value']:.1f}%)")
        print(f"Fastest model:       {summary['fastest']['model']} "
              f"({summary['fastest']['fps']:.1f} FPS)")
        if 'speed_ratio' in summary:
            print(f"Speed ratio:         {summary['speed_ratio']:.1f}x")
        print("=" * 70)

