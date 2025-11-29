#!/usr/bin/env python3
"""
Thesis Figure and Table Generator

This script generates all publication-ready figures and LaTeX tables
for the Results & Discussion chapter of a thesis.

Usage:
    python scripts/generate_thesis_figures.py
    python scripts/generate_thesis_figures.py --input results/evaluation_results.json
    python scripts/generate_thesis_figures.py --format pdf --no-show
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import RESULTS_DIR
from src.results.results_manager import ResultsManager
from src.results.thesis_figures import ThesisFigureGenerator
from src.results.latex_tables import LaTeXTableGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate thesis figures and LaTeX tables from evaluation results"
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
        help="Output directory for figures and tables"
    )
    parser.add_argument(
        "--format",
        choices=["pdf", "png", "svg"],
        default="pdf",
        help="Output format for figures (default: pdf)"
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Generate only figures, skip tables"
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Generate only tables, skip figures"
    )
    return parser.parse_args()


def main():
    """Generate all thesis outputs."""
    args = parse_args()

    print("=" * 70)
    print("THESIS FIGURE AND TABLE GENERATOR")
    print("=" * 70)

    # Check input file
    if not args.input.exists():
        print(f"ERROR: Results file not found: {args.input}")
        print("\nTo generate results, run:")
        print("  python scripts/evaluate_all.py")
        sys.exit(1)

    # Load results
    print(f"\nLoading results from: {args.input}")
    manager = ResultsManager(args.output_dir)
    results = manager.load_json(args.input)

    if not results:
        print("ERROR: No results found in file.")
        sys.exit(1)

    print(f"Loaded {len(results)} model evaluations")

    # Print summary
    manager.print_summary()

    # Generate outputs
    all_paths = []

    if not args.tables_only:
        print("\n" + "=" * 70)
        print("GENERATING FIGURES")
        print("=" * 70)

        fig_gen = ThesisFigureGenerator(
            results=results,
            output_dir=args.output_dir / "figures",
            format=args.format
        )
        figure_paths = fig_gen.generate_all()
        all_paths.extend(figure_paths)

    if not args.figures_only:
        print("\n" + "=" * 70)
        print("GENERATING LATEX TABLES")
        print("=" * 70)

        table_gen = LaTeXTableGenerator(
            results=results,
            output_dir=args.output_dir / "tables"
        )
        table_paths = table_gen.generate_all()
        all_paths.extend(table_paths)

    # Summary
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nGenerated {len(all_paths)} files:")

    figures_dir = args.output_dir / "figures"
    tables_dir = args.output_dir / "tables"

    if figures_dir.exists():
        fig_files = list(figures_dir.glob(f"*.{args.format}"))
        if fig_files:
            print(f"\nFigures ({args.output_dir / 'figures'}):")
            for f in sorted(fig_files):
                print(f"  - {f.name}")

    if tables_dir.exists():
        tex_files = list(tables_dir.glob("*.tex"))
        if tex_files:
            print(f"\nLaTeX Tables ({args.output_dir / 'tables'}):")
            for f in sorted(tex_files):
                print(f"  - {f.name}")

    print("\n" + "-" * 70)
    print("Usage in LaTeX:")
    print("  \\input{tables/main_results.tex}")
    print("  \\includegraphics[width=\\textwidth]{figures/map_comparison.pdf}")
    print("-" * 70)


if __name__ == "__main__":
    main()

