"""LaTeX table generation for thesis."""

from pathlib import Path
from typing import List, Optional

from src.models.base import EvaluationResults
from src.config import RESULTS_DIR


class LaTeXTableGenerator:
    """Generate LaTeX tables for thesis."""

    def __init__(
        self,
        results: List[EvaluationResults],
        output_dir: Optional[Path] = None
    ):
        """
        Initialize the LaTeX table generator.

        Args:
            results: List of evaluation results.
            output_dir: Directory to save tables.
        """
        self.results = results
        self.output_dir = Path(output_dir or RESULTS_DIR / "tables")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_table(self, content: str, name: str) -> Path:
        """Save LaTeX table to file."""
        path = self.output_dir / f"{name}.tex"
        with open(path, 'w') as f:
            f.write(content)
        print(f"Saved: {path}")
        return path

    def _format_value(self, value: Optional[float], precision: int = 1) -> str:
        """Format a value for LaTeX, handling None."""
        if value is None:
            return "---"
        return f"{value:.{precision}f}"

    def _bold_best(self, values: List[Optional[float]], maximize: bool = True) -> List[str]:
        """
        Format values with best one in bold.

        Args:
            values: List of values (may contain None).
            maximize: If True, bold the maximum; else bold the minimum.

        Returns:
            List of formatted strings with best in bold.
        """
        formatted = []
        valid_values = [v for v in values if v is not None]

        if not valid_values:
            return [self._format_value(v) for v in values]

        best = max(valid_values) if maximize else min(valid_values)

        for v in values:
            if v is None:
                formatted.append("---")
            elif abs(v - best) < 0.01:  # Account for float comparison
                formatted.append(f"\\textbf{{{v:.1f}}}")
            else:
                formatted.append(f"{v:.1f}")

        return formatted

    def generate_main_results_table(self) -> Path:
        """
        Generate the main results comparison table.

        Returns:
            Path to saved .tex file.
        """
        # Extract metrics
        models = [r.model_name for r in self.results]
        map50 = [r.map50 for r in self.results]
        map50_95 = [r.map50_95 for r in self.results]
        precision = [r.precision for r in self.results]
        recall = [r.recall for r in self.results]
        f1 = [r.f1_score for r in self.results]
        fps = [r.fps for r in self.results]

        # Bold best values
        map50_fmt = self._bold_best(map50)
        map50_95_fmt = self._bold_best(map50_95)
        prec_fmt = self._bold_best(precision)
        rec_fmt = self._bold_best(recall)
        f1_fmt = self._bold_best(f1)
        fps_fmt = self._bold_best(fps)

        # Build LaTeX table
        lines = [
            "% Main Results Table - Object Detection Performance Comparison",
            "% Include with: \\input{tables/main_results.tex}",
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Object Detection Model Performance Comparison on COCO Validation Subset}",
            "\\label{tab:main_results}",
            "\\begin{tabular}{lcccccc}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{mAP@0.5} & \\textbf{mAP@0.5:0.95} & "
            "\\textbf{Precision} & \\textbf{Recall} & \\textbf{F1 Score} & \\textbf{FPS} \\\\",
            "\\midrule",
        ]

        for i, model in enumerate(models):
            row = (
                f"{model} & {map50_fmt[i]}\\% & {map50_95_fmt[i]}\\% & "
                f"{prec_fmt[i]}\\% & {rec_fmt[i]}\\% & {f1_fmt[i]}\\% & {fps_fmt[i]} \\\\"
            )
            lines.append(row)

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        content = "\n".join(lines)
        return self._save_table(content, "main_results")

    def generate_speed_table(self) -> Path:
        """
        Generate timing breakdown table.

        Returns:
            Path to saved .tex file.
        """
        models = [r.model_name for r in self.results]

        # Extract timing data
        preprocess = []
        inference = []
        postprocess = []
        total = []
        fps_list = []

        for r in self.results:
            if r.timing_breakdown:
                preprocess.append(r.timing_breakdown.preprocess_ms)
                inference.append(r.timing_breakdown.inference_ms)
                postprocess.append(r.timing_breakdown.postprocess_ms)
                total.append(r.timing_breakdown.total_ms)
            else:
                preprocess.append(None)
                inference.append(r.time_per_image_ms)
                postprocess.append(None)
                total.append(r.time_per_image_ms)
            fps_list.append(r.fps)

        # Bold best (minimize for times, maximize for FPS)
        pre_fmt = self._bold_best(preprocess, maximize=False)
        inf_fmt = self._bold_best(inference, maximize=False)
        post_fmt = self._bold_best(postprocess, maximize=False)
        total_fmt = self._bold_best(total, maximize=False)
        fps_fmt = self._bold_best(fps_list, maximize=True)

        lines = [
            "% Speed Comparison Table - Inference Time Breakdown",
            "% Include with: \\input{tables/speed_comparison.tex}",
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Inference Speed Comparison (milliseconds per image)}",
            "\\label{tab:speed_comparison}",
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Preprocess} & \\textbf{Inference} & "
            "\\textbf{Postprocess} & \\textbf{Total} & \\textbf{FPS} \\\\",
            "\\midrule",
        ]

        for i, model in enumerate(models):
            row = (
                f"{model} & {pre_fmt[i]} & {inf_fmt[i]} & "
                f"{post_fmt[i]} & {total_fmt[i]} & {fps_fmt[i]} \\\\"
            )
            lines.append(row)

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        content = "\n".join(lines)
        return self._save_table(content, "speed_comparison")

    def generate_ap_by_size_table(self) -> Path:
        """
        Generate AP by object size table.

        Returns:
            Path to saved .tex file.
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
                ap_small.append(None)
                ap_medium.append(None)
                ap_large.append(None)

        # Bold best values
        small_fmt = self._bold_best(ap_small)
        medium_fmt = self._bold_best(ap_medium)
        large_fmt = self._bold_best(ap_large)

        lines = [
            "% AP by Object Size Table",
            "% Include with: \\input{tables/ap_by_size.tex}",
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Average Precision by Object Size (COCO size categories)}",
            "\\label{tab:ap_by_size}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{AP\\textsubscript{small}} & "
            "\\textbf{AP\\textsubscript{medium}} & \\textbf{AP\\textsubscript{large}} \\\\",
            " & (area $<$ 32\\textsuperscript{2}) & "
            "(32\\textsuperscript{2} $<$ area $<$ 96\\textsuperscript{2}) & "
            "(area $>$ 96\\textsuperscript{2}) \\\\",
            "\\midrule",
        ]

        for i, model in enumerate(models):
            row = f"{model} & {small_fmt[i]}\\% & {medium_fmt[i]}\\% & {large_fmt[i]}\\% \\\\"
            lines.append(row)

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        content = "\n".join(lines)
        return self._save_table(content, "ap_by_size")

    def generate_model_info_table(self) -> Path:
        """
        Generate model architecture information table.

        Returns:
            Path to saved .tex file.
        """
        models = [r.model_name for r in self.results]

        # Extract model info
        params = []
        gflops = []
        input_size = []

        for r in self.results:
            if r.model_info:
                params.append(r.model_info.parameters / 1e6)  # Convert to millions
                gflops.append(r.model_info.gflops)
                input_size.append(r.model_info.input_size)
            else:
                params.append(None)
                gflops.append(None)
                input_size.append(None)

        # Bold best (minimize for params and GFLOPs)
        params_fmt = self._bold_best(params, maximize=False)
        gflops_fmt = self._bold_best(gflops, maximize=False)

        lines = [
            "% Model Architecture Information Table",
            "% Include with: \\input{tables/model_info.tex}",
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Model Architecture Comparison}",
            "\\label{tab:model_info}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Parameters (M)} & \\textbf{GFLOPs} & "
            "\\textbf{Input Size} \\\\",
            "\\midrule",
        ]

        for i, model in enumerate(models):
            size = f"{input_size[i]}" if input_size[i] else "---"
            row = f"{model} & {params_fmt[i]} & {gflops_fmt[i]} & {size} \\\\"
            lines.append(row)

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        content = "\n".join(lines)
        return self._save_table(content, "model_info")

    def generate_all(self) -> List[Path]:
        """
        Generate all LaTeX tables.

        Returns:
            List of paths to generated tables.
        """
        print(f"\nGenerating LaTeX tables in: {self.output_dir}")
        print("-" * 50)

        paths = [
            self.generate_main_results_table(),
            self.generate_speed_table(),
            self.generate_ap_by_size_table(),
            self.generate_model_info_table(),
        ]

        print("-" * 50)
        print(f"Generated {len(paths)} tables")

        return paths

