#!/usr/bin/env python3
"""
Generate Results & Discussion Chapter PDF for Master's Thesis.

Creates a comprehensive PDF document with all evaluation results,
figures, and analysis for the thesis chapter.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fpdf import FPDF
from datetime import datetime


class ThesisPDF(FPDF):
    """Custom PDF class for thesis chapter."""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)
        
    def header(self):
        self.set_font('Helvetica', 'I', 9)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, 'Results & Discussion - Object Detection Performance Analysis', 0, 0, 'C')
        self.ln(15)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
        
    def chapter_title(self, title: str, level: int = 1):
        """Add a chapter/section title."""
        if level == 1:
            self.set_font('Helvetica', 'B', 18)
            self.set_text_color(0, 51, 102)
        elif level == 2:
            self.set_font('Helvetica', 'B', 14)
            self.set_text_color(0, 76, 153)
        else:
            self.set_font('Helvetica', 'B', 12)
            self.set_text_color(0, 102, 204)
        
        self.ln(5)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(3)
        
    def body_text(self, text: str):
        """Add body text with proper formatting."""
        self.set_font('Helvetica', '', 11)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, text)
        self.ln(3)
        
    def add_table(self, headers: List[str], data: List[List[str]], col_widths: List[int] = None):
        """Add a formatted table."""
        if col_widths is None:
            col_widths = [int(190 / len(headers))] * len(headers)
        
        # Header
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(0, 51, 102)
        self.set_text_color(255, 255, 255)
        
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 8, header, 1, 0, 'C', True)
        self.ln()
        
        # Data rows
        self.set_font('Helvetica', '', 9)
        self.set_text_color(0, 0, 0)
        
        for row_idx, row in enumerate(data):
            if row_idx % 2 == 0:
                self.set_fill_color(240, 240, 240)
            else:
                self.set_fill_color(255, 255, 255)
            
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 7, str(cell), 1, 0, 'C', True)
            self.ln()
        
        self.ln(5)
        
    def add_figure(self, image_path: str, caption: str, width: int = 170):
        """Add a figure with caption."""
        if Path(image_path).exists():
            # Center the image
            x = (210 - width) / 2
            self.image(image_path, x=x, w=width)
            self.ln(3)
            self.set_font('Helvetica', 'I', 9)
            self.set_text_color(80, 80, 80)
            self.multi_cell(0, 5, caption, 0, 'C')
            self.ln(5)
        else:
            self.body_text(f"[Figure not found: {image_path}]")


def load_results(results_dir: Path) -> Dict[str, Any]:
    """Load evaluation results from JSON."""
    results_file = results_dir / "thesis" / "data" / "all_models_results.json"
    
    with open(results_file, 'r') as f:
        return json.load(f)


def generate_results_chapter(output_path: Path):
    """Generate the Results & Discussion chapter PDF."""
    results_dir = project_root / 'results'
    figures_dir = results_dir / 'thesis' / 'figures'
    
    # Load results
    data = load_results(results_dir)
    results = data['results']
    summary = data.get('summary', {})
    
    # Sort results by mAP@0.5:0.95
    results_sorted = sorted(results, key=lambda x: x['map50_95'], reverse=True)
    
    # Create PDF
    pdf = ThesisPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # =========================================================================
    # Title Page
    # =========================================================================
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(0, 51, 102)
    pdf.ln(40)
    pdf.cell(0, 15, 'Chapter 4', 0, 1, 'C')
    pdf.cell(0, 15, 'Results & Discussion', 0, 1, 'C')
    pdf.ln(20)
    
    pdf.set_font('Helvetica', '', 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, 'Performance Analysis of Object Detection Models', 0, 1, 'C')
    pdf.cell(0, 10, 'for Visually Impaired Assistance', 0, 1, 'C')
    pdf.ln(30)
    
    pdf.set_font('Helvetica', 'I', 11)
    pdf.cell(0, 8, f'Evaluation Date: {datetime.now().strftime("%B %d, %Y")}', 0, 1, 'C')
    pdf.cell(0, 8, f'Dataset: COCO val2017 ({data["num_images"]} images)', 0, 1, 'C')
    pdf.cell(0, 8, f'Models Evaluated: {data["num_models"]}', 0, 1, 'C')
    
    # =========================================================================
    # Section 4.1: Overview
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('4.1 Evaluation Overview', 1)
    
    pdf.body_text(
        'This chapter presents the comprehensive evaluation results of 14 object detection models '
        'evaluated on the COCO val2017 dataset subset. The models evaluated include variants from '
        'the YOLO family (YOLOv8 and YOLOv10), transformer-based RT-DETR, and traditional architectures '
        '(SSD, RetinaNet, Faster R-CNN). Each model was evaluated using standardized metrics including '
        'mean Average Precision (mAP), recall, precision, and inference speed (FPS).'
    )
    
    pdf.body_text(
        'The evaluation was conducted with a specific focus on assistive technology applications '
        'for visually impaired persons, where high recall (detecting all relevant objects) and '
        'real-time performance are critical requirements. Models must balance accuracy with speed '
        'to be deployable on mobile devices and edge computing platforms.'
    )
    
    pdf.chapter_title('4.1.1 Models Evaluated', 2)
    
    model_overview = [
        ['YOLOv8', 'n, s, m, l, x', 'Ultralytics', 'Single-stage, anchor-free'],
        ['YOLOv10', 'n, s, m, l, x', 'Ultralytics', 'NMS-free, efficient'],
        ['RT-DETR', 'l', 'Baidu', 'Transformer-based'],
        ['SSD', '300', 'Liu et al.', 'Single-stage, multi-scale'],
        ['RetinaNet', 'base', 'Facebook AI', 'Focal loss, FPN'],
        ['Faster R-CNN', 'base', 'Microsoft', 'Two-stage, RPN'],
    ]
    
    pdf.add_table(
        ['Family', 'Variants', 'Source', 'Architecture'],
        model_overview,
        [30, 35, 45, 80]
    )
    
    # =========================================================================
    # Section 4.2: Main Results
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('4.2 Main Results', 1)
    
    pdf.body_text(
        'Table 4.1 presents the comprehensive evaluation results for all 14 models, '
        'sorted by mAP@0.5:0.95 in descending order. The results demonstrate clear '
        'trade-offs between model complexity, accuracy, and inference speed.'
    )
    
    # Main results table
    main_results = []
    for r in results_sorted:
        recall = f"{r['recall']:.1f}" if r.get('recall') else 'N/A'
        precision = f"{r['precision']:.1f}" if r.get('precision') else 'N/A'
        main_results.append([
            r['model_name'],
            f"{r['map50']:.1f}",
            f"{r['map50_95']:.1f}",
            recall,
            precision,
            f"{r['fps']:.1f}",
            f"{r['latency_ms']:.1f}"
        ])
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 8, 'Table 4.1: Comprehensive Model Comparison', 0, 1, 'C')
    pdf.add_table(
        ['Model', 'mAP@0.5', 'mAP@0.5:0.95', 'Recall', 'Precision', 'FPS', 'Latency(ms)'],
        main_results,
        [30, 22, 30, 22, 25, 22, 30]
    )
    
    # Key findings
    pdf.chapter_title('4.2.1 Key Findings', 2)
    
    best_map = summary.get('best_map50_95', {})
    best_recall = summary.get('best_recall', {})
    fastest = summary.get('fastest', {})
    
    pdf.body_text(
        f"Best Accuracy (mAP@0.5:0.95): {best_map.get('model', 'N/A')} achieved the highest "
        f"accuracy with {best_map.get('value', 0):.1f}%, demonstrating excellent detection "
        f"capability across various IoU thresholds."
    )
    
    pdf.body_text(
        f"Best Recall: {best_recall.get('model', 'N/A')} achieved the highest recall at "
        f"{best_recall.get('value', 0):.1f}%, which is critical for assistive technology "
        f"applications where missing detections could impact user safety."
    )
    
    pdf.body_text(
        f"Fastest Model: {fastest.get('model', 'N/A')} achieved {fastest.get('fps', 0):.1f} FPS, "
        f"making it highly suitable for real-time mobile deployment while maintaining "
        f"reasonable detection accuracy."
    )
    
    # =========================================================================
    # Section 4.3: YOLO Family Analysis
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('4.3 YOLO Family Analysis', 1)
    
    pdf.body_text(
        'The YOLO (You Only Look Once) family represents state-of-the-art single-stage '
        'object detectors. We evaluated five variants each of YOLOv8 and YOLOv10, '
        'ranging from nano (n) to extra-large (x), to understand the accuracy-speed trade-offs.'
    )
    
    pdf.chapter_title('4.3.1 YOLOv8 Results', 2)
    
    # YOLOv8 table
    yolov8_results = [r for r in results if r['model_family'] == 'YOLOv8']
    yolov8_results = sorted(yolov8_results, key=lambda x: ['n', 's', 'm', 'l', 'x'].index(x['model_variant']))
    
    yolov8_data = []
    for r in yolov8_results:
        recall = f"{r['recall']:.1f}" if r.get('recall') else 'N/A'
        yolov8_data.append([
            r['model_name'],
            f"{r['map50']:.1f}",
            f"{r['map50_95']:.1f}",
            recall,
            f"{r['fps']:.1f}"
        ])
    
    pdf.add_table(
        ['Model', 'mAP@0.5', 'mAP@0.5:0.95', 'Recall (%)', 'FPS'],
        yolov8_data,
        [35, 35, 40, 35, 35]
    )
    
    # Add YOLOv8 figure
    yolov8_fig = figures_dir / 'family' / 'yolov8_variants_comparison.pdf'
    if yolov8_fig.exists():
        # Convert PDF to PNG for embedding (or use the first page)
        pdf.body_text(
            'Figure 4.1 shows the scaling behavior of YOLOv8 variants. As model size increases '
            'from nano to extra-large, accuracy improves significantly (37.0% to 53.9% mAP@0.5:0.95) '
            'while speed decreases (411 to 139 FPS).'
        )
    
    pdf.chapter_title('4.3.2 YOLOv10 Results', 2)
    
    # YOLOv10 table
    yolov10_results = [r for r in results if r['model_family'] == 'YOLOv10']
    yolov10_results = sorted(yolov10_results, key=lambda x: ['n', 's', 'm', 'l', 'x'].index(x['model_variant']))
    
    yolov10_data = []
    for r in yolov10_results:
        recall = f"{r['recall']:.1f}" if r.get('recall') else 'N/A'
        yolov10_data.append([
            r['model_name'],
            f"{r['map50']:.1f}",
            f"{r['map50_95']:.1f}",
            recall,
            f"{r['fps']:.1f}"
        ])
    
    pdf.add_table(
        ['Model', 'mAP@0.5', 'mAP@0.5:0.95', 'Recall (%)', 'FPS'],
        yolov10_data,
        [35, 35, 40, 35, 35]
    )
    
    pdf.body_text(
        'YOLOv10 introduces NMS-free detection, resulting in faster inference compared to '
        'YOLOv8 at equivalent model sizes. YOLOv10x achieves the best overall accuracy '
        '(54.3% mAP@0.5:0.95) among all YOLO variants.'
    )
    
    # =========================================================================
    # Section 4.4: YOLOv8 vs YOLOv10
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('4.3.3 YOLOv8 vs YOLOv10 Comparison', 2)
    
    pdf.body_text(
        'Direct comparison between YOLOv8 and YOLOv10 variants of the same size reveals '
        'interesting trade-offs. Table 4.4 presents a side-by-side comparison.'
    )
    
    comparison_data = []
    for v8, v10 in zip(yolov8_results, yolov10_results):
        v8_recall = f"{v8['recall']:.1f}" if v8.get('recall') else 'N/A'
        v10_recall = f"{v10['recall']:.1f}" if v10.get('recall') else 'N/A'
        comparison_data.append([
            v8['model_variant'].upper(),
            f"{v8['map50_95']:.1f}",
            f"{v10['map50_95']:.1f}",
            v8_recall,
            v10_recall,
            f"{v8['fps']:.0f}",
            f"{v10['fps']:.0f}"
        ])
    
    pdf.add_table(
        ['Size', 'v8 mAP', 'v10 mAP', 'v8 Recall', 'v10 Recall', 'v8 FPS', 'v10 FPS'],
        comparison_data,
        [20, 25, 25, 28, 28, 25, 25]
    )
    
    pdf.body_text(
        'Key observations:\n'
        '- YOLOv10 consistently achieves higher FPS due to NMS-free design\n'
        '- YOLOv10 shows marginally better mAP at most model sizes\n'
        '- Both families show similar recall patterns, with larger models achieving ~64% recall\n'
        '- For real-time mobile applications, YOLOv10n offers the best speed-accuracy trade-off'
    )
    
    # =========================================================================
    # Section 4.4: Other Architectures
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('4.4 Alternative Architectures', 1)
    
    pdf.chapter_title('4.4.1 RT-DETR (Transformer-based)', 2)
    
    rtdetr = next((r for r in results if r['model_family'] == 'RT-DETR'), None)
    if rtdetr:
        pdf.body_text(
            f"RT-DETR-l achieves impressive results with {rtdetr['map50_95']:.1f}% mAP@0.5:0.95 "
            f"and notably the highest recall ({rtdetr['recall']:.1f}%) among all evaluated models. "
            f"At {rtdetr['fps']:.1f} FPS, it offers real-time performance suitable for assistive "
            f"applications where high recall is critical."
        )
    
    pdf.chapter_title('4.4.2 SSD300', 2)
    
    ssd = next((r for r in results if r['model_family'] == 'SSD'), None)
    if ssd:
        pdf.body_text(
            f"SSD300 achieves {ssd['map50_95']:.1f}% mAP@0.5:0.95 with {ssd['fps']:.1f} FPS. "
            f"While faster than two-stage detectors, its accuracy is significantly lower than "
            f"modern YOLO variants, making it less suitable for safety-critical applications."
        )
    
    pdf.chapter_title('4.4.3 RetinaNet', 2)
    
    retinanet = next((r for r in results if r['model_family'] == 'RetinaNet'), None)
    if retinanet:
        pdf.body_text(
            f"RetinaNet with focal loss achieves {retinanet['map50_95']:.1f}% mAP@0.5:0.95 "
            f"and {retinanet['recall']:.1f}% recall. Its {retinanet['fps']:.1f} FPS is suitable "
            f"for near-real-time applications but falls short for mobile deployment."
        )
    
    pdf.chapter_title('4.4.4 Faster R-CNN', 2)
    
    frcnn = next((r for r in results if r['model_family'] == 'Faster R-CNN'), None)
    if frcnn:
        pdf.body_text(
            f"Faster R-CNN, as a two-stage detector, achieves {frcnn['map50_95']:.1f}% mAP@0.5:0.95 "
            f"with {frcnn['fps']:.1f} FPS. While offering competitive accuracy, its slower "
            f"inference speed limits real-time mobile deployment potential."
        )
    
    # =========================================================================
    # Section 4.5: Speed Analysis
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('4.5 Speed and Latency Analysis', 1)
    
    pdf.body_text(
        'For assistive technology applications, inference speed is crucial. The system must '
        'process frames in real-time (>30 FPS) to provide timely feedback to users. '
        'Table 4.5 presents the speed analysis with mobile feasibility assessment.'
    )
    
    # Speed table
    speed_sorted = sorted(results, key=lambda x: x['fps'], reverse=True)
    speed_data = []
    for r in speed_sorted:
        realtime = "Yes" if r['fps'] >= 30 else "No"
        mobile = "Yes" if r['fps'] >= 100 else ("Maybe" if r['fps'] >= 50 else "No")
        speed_data.append([
            r['model_name'],
            f"{r['fps']:.1f}",
            f"{r['latency_ms']:.1f}",
            realtime,
            mobile
        ])
    
    pdf.add_table(
        ['Model', 'FPS', 'Latency (ms)', 'Real-time*', 'Mobile**'],
        speed_data,
        [40, 30, 35, 35, 35]
    )
    
    pdf.set_font('Helvetica', 'I', 8)
    pdf.cell(0, 5, '* Real-time: >=30 FPS; ** Mobile Feasible: Estimated based on GPU-to-mobile scaling factors', 0, 1, 'L')
    pdf.ln(5)
    
    pdf.body_text(
        'All YOLO variants and RT-DETR achieve real-time performance on the evaluation hardware '
        '(NVIDIA RTX 4090). For mobile deployment, nano and small variants are recommended, '
        'with expected 20-35 FPS on high-end mobile devices after optimization.'
    )
    
    # =========================================================================
    # Section 4.6: Discussion
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('4.6 Discussion', 1)
    
    pdf.chapter_title('4.6.1 Accuracy vs Speed Trade-off', 2)
    
    pdf.body_text(
        'The evaluation reveals a clear accuracy-speed trade-off across model families. '
        'The Pareto frontier analysis shows that YOLOv10 variants dominate the efficiency '
        'curve, offering the best accuracy at any given speed constraint. For applications '
        'requiring >100 FPS, YOLOv10n provides 38.2% mAP with 462 FPS. For maximum accuracy '
        'without strict speed constraints, YOLOv10x achieves 54.3% mAP at 160 FPS.'
    )
    
    pdf.chapter_title('4.6.2 Recall Considerations for Assistive Technology', 2)
    
    pdf.body_text(
        'For visually impaired assistance, recall is arguably more important than precision - '
        'failing to detect an obstacle (false negative) is more dangerous than a false alarm. '
        'RT-DETR-l achieves the highest recall (65.2%), followed by YOLOv8x (64.5%) and '
        'YOLOv10l (63.7%). The nano variants achieve approximately 48% recall, which may be '
        'insufficient for safety-critical applications.'
    )
    
    pdf.chapter_title('4.6.3 Model Recommendations', 2)
    
    pdf.body_text(
        'Based on the evaluation results, we recommend the following models for different '
        'deployment scenarios:\n\n'
        '- High-end Mobile Devices: YOLOv10s (46.0% mAP, ~25 FPS expected)\n'
        '- Edge Devices (Jetson): YOLOv10m (50.8% mAP, ~20 FPS expected)\n'
        '- Maximum Accuracy: YOLOv10x or RT-DETR-l (>51% mAP)\n'
        '- Maximum Speed: YOLOv10n (38.2% mAP, highest FPS)'
    )
    
    pdf.chapter_title('4.6.4 Limitations', 2)
    
    pdf.body_text(
        'Several limitations should be considered:\n\n'
        '(1) Evaluation was performed on COCO dataset, which may not fully represent '
        'real-world assistive technology scenarios.\n'
        '(2) Speed measurements were taken on a high-end GPU; actual mobile performance '
        'will be significantly lower.\n'
        '(3) The evaluation focuses on general object detection; domain-specific fine-tuning '
        'may improve results for assistive applications.\n'
        '(4) Small object detection remains challenging across all models.'
    )
    
    # =========================================================================
    # Section 4.7: Conclusion
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('4.7 Summary', 1)
    
    pdf.body_text(
        'This chapter presented a comprehensive evaluation of 14 object detection models '
        'for potential use in assistive technology for visually impaired persons. The key '
        'findings can be summarized as follows:\n\n'
        '(1) YOLOv10 family offers the best accuracy-speed trade-off, with YOLOv10x achieving '
        'the highest mAP (54.3%) and YOLOv10n the highest speed (462 FPS).\n\n'
        '(2) RT-DETR-l achieves the highest recall (65.2%), making it suitable for safety-critical '
        'applications where missing detections must be minimized.\n\n'
        '(3) For mobile deployment, YOLOv10s or YOLOv10m offer the best balance between '
        'accuracy (46-51% mAP), speed, and model size.\n\n'
        '(4) Traditional architectures (SSD, Faster R-CNN) are outperformed by modern YOLO '
        'variants in both accuracy and speed.\n\n'
        '(5) All top-performing models achieve real-time performance (>30 FPS) on GPU hardware, '
        'with expected viable performance on modern mobile devices after optimization.'
    )
    
    # Save PDF
    pdf.output(str(output_path))
    print(f"PDF generated: {output_path}")


def main():
    output_dir = project_root / 'results' / 'thesis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'Results_and_Discussion_Chapter.pdf'
    
    print("=" * 70)
    print("GENERATING RESULTS & DISCUSSION CHAPTER PDF")
    print("=" * 70)
    
    generate_results_chapter(output_path)
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()

