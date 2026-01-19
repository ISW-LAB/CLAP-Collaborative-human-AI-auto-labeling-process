"""
evaluator.py - Performance evaluation module
"""

import os
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from detector import ObjectDetector
from utils import load_yolo_labels, yolo_to_xyxy, calculate_iou

class PerformanceEvaluator:
    """Performance evaluation class"""

    def __init__(self, detector: ObjectDetector, image_dir: str, label_dir: str,
                 iou_threshold: float = 0.5):
        """
        Initialize performance evaluator - includes label directory availability check

        Args:
            detector: Object detector
            image_dir: Image directory
            label_dir: Ground truth label directory
            iou_threshold: IoU threshold
        """
        self.detector = detector
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.iou_threshold = iou_threshold

        # Check label directory availability (newly added)
        self.labels_available = (
            os.path.exists(label_dir) and
            len([f for f in os.listdir(label_dir) if f.lower().endswith('.txt')]) > 0
        )

    def evaluate(self, cycle: int, model_name: str) -> Dict[str, float]:
        """
        Perform performance evaluation - returns default values if labels are unavailable

        Args:
            cycle: Current cycle
            model_name: Model name

        Returns:
            Performance metrics dictionary
        """
        if not self.labels_available:
            print(f"Warning: Skipping performance evaluation for Cycle {cycle} due to missing labels.")
            return {
                'Cycle': cycle,
                'Model': model_name,
                'mAP50': -1.0,
                'Precision': -1.0,
                'Recall': -1.0,
                'F1-Score': -1.0,
                'Detected_Objects': 0,
                'Filtered_Objects': 0,
                'Labels_Available': False  # Newly added
            }

        # Execute existing evaluation logic
        image_files = [f for f in os.listdir(self.image_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        all_precisions = []
        all_recalls = []
        all_f1_scores = []

        total_detected_objects = 0
        total_filtered_objects = 0

        # Backup existing statistics
        original_stats = self.detector.get_stats()
        self.detector.reset_stats()

        for image_file in tqdm(image_files, desc=f"Evaluating (Cycle {cycle})", leave=False):
            image_path = os.path.join(self.image_dir, image_file)

            # Perform object detection
            detected_objects, filtered_objects, _, _ = self.detector.detect_and_classify(image_path, cycle)

            total_detected_objects += len(detected_objects)
            total_filtered_objects += len(filtered_objects)

            # Load ground truth labels
            gt_label_path = os.path.join(self.label_dir, os.path.splitext(image_file)[0] + '.txt')
            if os.path.exists(gt_label_path):
                gt_objects = load_yolo_labels(gt_label_path)

                # Calculate performance metrics
                precision, recall, f1 = self._calculate_metrics(detected_objects, gt_objects, image_path)

                all_precisions.append(precision)
                all_recalls.append(recall)
                all_f1_scores.append(f1)
            else:
                # If label file doesn't exist, record as 0
                all_precisions.append(0.0)
                all_recalls.append(0.0)
                all_f1_scores.append(0.0)

        # Restore statistics
        self.detector.stats = original_stats

        # Average performance
        avg_precision = np.mean(all_precisions)
        avg_recall = np.mean(all_recalls)
        avg_f1 = np.mean(all_f1_scores)

        # Filtering is not applied in Cycle 0
        filtered_count = total_filtered_objects if cycle > 0 else 0

        return {
            'Cycle': cycle,
            'Model': model_name,
            'mAP50': avg_precision,
            'Precision': avg_precision,
            'Recall': avg_recall,
            'F1-Score': avg_f1,
            'Detected_Objects': total_detected_objects,
            'Filtered_Objects': filtered_count,
            'Labels_Available': True  # Newly added
        }

    def _calculate_metrics(self, detected_objects: List, gt_objects: List, image_path: str) -> Tuple[float, float, float]:
        """Calculate performance metrics for individual image"""
        if len(gt_objects) == 0 and len(detected_objects) == 0:
            return 1.0, 1.0, 1.0
        elif len(gt_objects) == 0:
            return 0.0, 1.0, 0.0
        elif len(detected_objects) == 0:
            return 1.0, 0.0, 0.0

        # Image size information
        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        # Convert YOLO format to xyxy
        gt_boxes = []
        for gt_obj in gt_objects:
            cls_id, center_x, center_y, width, height = gt_obj
            x1, y1, x2, y2 = yolo_to_xyxy((cls_id, center_x, center_y, width, height), w, h)
            gt_boxes.append((x1, y1, x2, y2))

        pred_boxes = []
        for det_obj in detected_objects:
            cls_id, center_x, center_y, width, height = det_obj
            x1, y1, x2, y2 = yolo_to_xyxy((cls_id, center_x, center_y, width, height), w, h)
            pred_boxes.append((x1, y1, x2, y2))

        # Calculate True Positives
        tp = 0
        matched_gt = set()

        for pred_box in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue

                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= self.iou_threshold and best_gt_idx != -1:
                tp += 1
                matched_gt.add(best_gt_idx)

        # Calculate performance metrics
        precision = tp / len(pred_boxes)
        recall = tp / len(gt_boxes)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

class MetricsManager:
    """Performance metrics management class"""

    def __init__(self, output_dir: str):
        """
        Initialize metrics manager - adds Labels_Available to columns

        Args:
            output_dir: Output directory
        """
        self.output_dir = output_dir
        self.metrics_file = os.path.join(output_dir, "performance_metrics.csv")

        # Initialize metrics dataframe (adds Labels_Available to column structure)
        self.columns = [
            'Cycle', 'Model', 'mAP50', 'Precision', 'Recall', 'F1-Score',
            'Detected_Objects', 'Filtered_Objects', 'Labels_Available'  # Newly added
        ]

        # Load existing metrics or create new
        if os.path.exists(self.metrics_file):
            try:
                self.metrics_df = pd.read_csv(self.metrics_file)

                # Add Labels_Available column if it doesn't exist (backward compatibility)
                if 'Labels_Available' not in self.metrics_df.columns:
                    self.metrics_df['Labels_Available'] = True

                print(f"Loaded existing metrics: {len(self.metrics_df)} records")
            except Exception as e:
                print(f"Warning: Failed to load existing metrics: {e}")
                self.metrics_df = pd.DataFrame(columns=self.columns)
        else:
            self.metrics_df = pd.DataFrame(columns=self.columns)

    def add_metrics(self, metrics: Dict[str, float]):
        """Add new metrics (resolves FutureWarning)"""
        cycle = metrics['Cycle']
        model = metrics['Model']

        # Check and update existing entry
        existing_mask = (self.metrics_df['Cycle'] == cycle) & (self.metrics_df['Model'] == model)

        if existing_mask.any():
            # Update existing entry
            for key, value in metrics.items():
                self.metrics_df.loc[existing_mask, key] = value
            print(f"Updated metrics: Cycle {cycle}, Model {model}")
        else:
            # Add new entry - resolves FutureWarning
            new_row = pd.DataFrame([metrics])

            if self.metrics_df.empty:
                # If DataFrame is empty, assign directly
                self.metrics_df = new_row
            else:
                # If there's existing data, use concat
                self.metrics_df = pd.concat([self.metrics_df, new_row], ignore_index=True)

            print(f"Added new metrics: Cycle {cycle}, Model {model}")

        # Save to file
        self.save_metrics()

    def save_metrics(self):
        """Save metrics to file"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.metrics_df.to_csv(self.metrics_file, index=False)
        except Exception as e:
            print(f"Warning: Failed to save metrics: {e}")

    def get_best_performance(self, metric: str = 'mAP50') -> Optional[Dict]:
        """Return best performance result"""
        if len(self.metrics_df) == 0:
            return None

        try:
            # Consider only valid performance metrics (mAP50 >= 0)
            valid_metrics = self.metrics_df[self.metrics_df[metric] >= 0]
            if len(valid_metrics) == 0:
                return None

            best_idx = valid_metrics[metric].idxmax()
            return valid_metrics.iloc[best_idx].to_dict()
        except Exception as e:
            print(f"Warning: Failed to retrieve best performance: {e}")
            return None

    def get_latest_performance(self) -> Optional[Dict]:
        """Return latest performance metrics"""
        if self.metrics_df.empty:
            return None

        return self.metrics_df.iloc[-1].to_dict()

    def get_cycle_performance(self, cycle: int) -> pd.DataFrame:
        """Return performance for specific cycle"""
        return self.metrics_df[self.metrics_df['Cycle'] == cycle].copy()

    def get_model_performance(self, model_name: str) -> pd.DataFrame:
        """Return performance for specific model"""
        return self.metrics_df[self.metrics_df['Model'] == model_name].copy()

    def export_summary(self, summary_path: str):
        """Export performance summary - adds explanation when labels are missing"""
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("Active Learning Performance Summary\n")
                f.write("="*50 + "\n\n")

                if len(self.metrics_df) > 0:
                    # Check label availability
                    labels_available = self.metrics_df['Labels_Available'].iloc[0] if 'Labels_Available' in self.metrics_df.columns else True

                    if not labels_available:
                        f.write("Warning: Performance metrics (mAP, Precision, Recall) cannot be measured due to missing label data.\n")
                        f.write("Only detected object counts and filtering results are recorded.\n")
                        f.write("Performance metrics are displayed as -1.\n\n")

                    # Performance trend by cycle
                    f.write("Performance Trend by Cycle:\n")
                    f.write("-" * 30 + "\n")

                    for idx, row in self.metrics_df.iterrows():
                        cycle = int(row['Cycle'])
                        detected = row['Detected_Objects']
                        filtered = row.get('Filtered_Objects', 0)

                        f.write(f"Cycle {cycle}:\n")
                        f.write(f"  Detected Objects: {detected}\n")
                        f.write(f"  Filtered Objects: {filtered}\n")

                        if labels_available and row['mAP50'] >= 0:
                            mAP50 = row['mAP50']
                            precision = row['Precision']
                            recall = row['Recall']
                            f.write(f"  mAP50: {mAP50:.4f}\n")
                            f.write(f"  Precision: {precision:.4f}\n")
                            f.write(f"  Recall: {recall:.4f}\n")
                        else:
                            f.write("  Performance Metrics: Not measurable (no labels)\n")

                        f.write("\n")

                    # Best performance (only when labels are available and measurable)
                    if labels_available:
                        best_performance = self.get_best_performance()
                        if best_performance:
                            f.write(f"Best Performance:\n")
                            f.write("-" * 30 + "\n")
                            f.write(f"Cycle {int(best_performance['Cycle'])}: "
                                   f"mAP50={best_performance['mAP50']:.4f}\n")
                            f.write(f"Model: {best_performance['Model']}\n")
                            f.write(f"Precision: {best_performance['Precision']:.4f}\n")
                            f.write(f"Recall: {best_performance['Recall']:.4f}\n")
                            f.write(f"F1-Score: {best_performance['F1-Score']:.4f}\n")

                    # Performance improvement analysis
                    if len(self.metrics_df) > 1 and labels_available:
                        # Only analyze if there are valid performance metrics
                        valid_metrics = self.metrics_df[self.metrics_df['mAP50'] >= 0]
                        if len(valid_metrics) > 1:
                            first_cycle = valid_metrics.iloc[0]
                            last_cycle = valid_metrics.iloc[-1]

                            if first_cycle['Model'] == last_cycle['Model']:
                                initial_mAP = first_cycle['mAP50']
                                final_mAP = last_cycle['mAP50']
                                improvement = final_mAP - initial_mAP

                                f.write(f"\nPerformance Improvement Analysis ({first_cycle['Model']}):\n")
                                f.write("-" * 30 + "\n")
                                f.write(f"Initial mAP50 (Cycle {int(first_cycle['Cycle'])}): {initial_mAP:.4f}\n")
                                f.write(f"Final mAP50 (Cycle {int(last_cycle['Cycle'])}): {final_mAP:.4f}\n")
                                f.write(f"Improvement: {improvement:+.4f}\n")

                                if initial_mAP > 0:
                                    improvement_pct = (improvement / initial_mAP * 100)
                                    f.write(f"Improvement Ratio: {improvement_pct:+.2f}%\n")

                    # Best performance by model (if multiple models exist)
                    if 'Model' in self.metrics_df.columns:
                        unique_models = self.metrics_df['Model'].unique()
                        if len(unique_models) > 1:
                            f.write(f"\nBest Performance by Model:\n")
                            f.write("-" * 30 + "\n")

                            for model in unique_models:
                                model_data = self.metrics_df[
                                    (self.metrics_df['Model'] == model) &
                                    (self.metrics_df['mAP50'] >= 0)
                                ]
                                if len(model_data) > 0:
                                    best_idx = model_data['mAP50'].idxmax()
                                    best_row = model_data.loc[best_idx]
                                    f.write(f"{model}: mAP50={best_row['mAP50']:.4f} "
                                           f"(Cycle {int(best_row['Cycle'])})\n")
                else:
                    f.write("No evaluation data available.\n")

            print(f"Performance summary saved: {summary_path}")

        except Exception as e:
            print(f"Warning: Failed to export performance summary: {e}")

    def plot_performance_curve(self, save_path: str = None):
        """Generate performance curve graph"""
        if self.metrics_df.empty:
            print("Warning: No data available to generate graph.")
            return

        # Filter only valid performance metrics
        valid_metrics = self.metrics_df[self.metrics_df['mAP50'] >= 0]
        if len(valid_metrics) == 0:
            print("Warning: Cannot generate graph due to no valid performance metrics.")
            return

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))

            cycles = valid_metrics['Cycle']

            # Create subplots
            plt.subplot(2, 2, 1)
            plt.plot(cycles, valid_metrics['mAP50'], 'b-o', linewidth=2, markersize=6)
            plt.title('mAP50 Performance')
            plt.xlabel('Cycle')
            plt.ylabel('mAP50')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)

            plt.subplot(2, 2, 2)
            plt.plot(cycles, valid_metrics['Precision'], 'r-s', linewidth=2, markersize=6)
            plt.title('Precision')
            plt.xlabel('Cycle')
            plt.ylabel('Precision')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)

            plt.subplot(2, 2, 3)
            plt.plot(cycles, valid_metrics['Recall'], 'g-^', linewidth=2, markersize=6)
            plt.title('Recall')
            plt.xlabel('Cycle')
            plt.ylabel('Recall')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)

            plt.subplot(2, 2, 4)
            plt.plot(cycles, valid_metrics['mAP50'], 'b-o', label='mAP50', linewidth=2)
            plt.plot(cycles, valid_metrics['Precision'], 'r-s', label='Precision', linewidth=2)
            plt.plot(cycles, valid_metrics['Recall'], 'g-^', label='Recall', linewidth=2)
            plt.title('All Metrics')
            plt.xlabel('Cycle')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)

            plt.tight_layout()

            if save_path is None:
                save_path = os.path.join(self.output_dir, "performance_curve.png")

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Performance curve graph saved: {save_path}")

        except ImportError:
            print("Warning: Cannot generate graph because matplotlib is not installed.")
        except Exception as e:
            print(f"Warning: Error during graph generation: {e}")

class ComparisonAnalyzer:
    """Experimental results comparison analysis class"""

    def __init__(self, base_results_dir: str):
        """
        Initialize comparison analyzer

        Args:
            base_results_dir: Base results directory
        """
        self.base_results_dir = base_results_dir

    def compare_experiments(self, experiment_names: List[str]) -> pd.DataFrame:
        """Compare multiple experimental results"""
        all_metrics = []

        for exp_name in experiment_names:
            metrics_file = os.path.join(self.base_results_dir, exp_name, "performance_metrics.csv")
            if os.path.exists(metrics_file):
                try:
                    df = pd.read_csv(metrics_file)
                    df['Experiment'] = exp_name
                    all_metrics.append(df)
                    print(f"Loaded experiment data: {exp_name} ({len(df)} records)")
                except Exception as e:
                    print(f"Warning: Failed to load experiment data ({exp_name}): {e}")
            else:
                print(f"Warning: Metrics file not found: {metrics_file}")

        if all_metrics:
            combined_df = pd.concat(all_metrics, ignore_index=True)
            print(f"Combined {len(combined_df)} records successfully")
            return combined_df
        else:
            print("Warning: No experiment data to compare.")
            return pd.DataFrame()

    def generate_comparison_report(self, experiment_names: List[str], output_path: str):
        """Generate experiment comparison report"""
        comparison_df = self.compare_experiments(experiment_names)

        if len(comparison_df) == 0:
            print("Warning: No experiment data to compare.")
            return

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("Experiment Comparison Report\n")
                f.write("="*60 + "\n\n")

                # Check label availability
                labels_available = comparison_df.get('Labels_Available', pd.Series([True])).iloc[0]

                if not labels_available:
                    f.write("Warning: Some experiments cannot measure performance metrics due to missing label data.\n")
                    f.write("Only detected object counts and filtering results are compared.\n\n")

                # Best performance by experiment (only when labels are available)
                if labels_available:
                    f.write("Best Performance by Experiment (mAP50):\n")
                    f.write("-"*40 + "\n")

                    for exp_name in experiment_names:
                        exp_data = comparison_df[comparison_df['Experiment'] == exp_name]
                        valid_data = exp_data[exp_data['mAP50'] >= 0]

                        if len(valid_data) > 0:
                            best_idx = valid_data['mAP50'].idxmax()
                            best = valid_data.loc[best_idx]
                            f.write(f"{exp_name}:\n")
                            f.write(f"  Best mAP50: {best['mAP50']:.4f} (Cycle {best['Cycle']}, Model: {best['Model']})\n")
                            f.write(f"  Precision: {best['Precision']:.4f}\n")
                            f.write(f"  Recall: {best['Recall']:.4f}\n")
                            f.write(f"  F1-Score: {best['F1-Score']:.4f}\n\n")
                        else:
                            f.write(f"{exp_name}: No measurable performance metrics\n\n")

                # Detected object count comparison
                f.write("Detected Object Count Comparison by Experiment:\n")
                f.write("-"*40 + "\n")

                for exp_name in experiment_names:
                    exp_data = comparison_df[comparison_df['Experiment'] == exp_name]
                    if len(exp_data) > 0:
                        total_detected = exp_data['Detected_Objects'].sum()
                        total_filtered = exp_data['Filtered_Objects'].sum()
                        f.write(f"{exp_name}:\n")
                        f.write(f"  Total Detected Objects: {total_detected}\n")
                        f.write(f"  Total Filtered Objects: {total_filtered}\n\n")

                # Performance comparison by cycle (only when labels are available)
                if labels_available:
                    f.write("Average Performance Comparison by Cycle:\n")
                    f.write("-"*40 + "\n")

                    try:
                        valid_comparison = comparison_df[comparison_df['mAP50'] >= 0]
                        if len(valid_comparison) > 0:
                            cycle_comparison = valid_comparison.groupby(['Experiment', 'Cycle'])['mAP50'].mean().unstack(fill_value=0)
                            f.write(cycle_comparison.to_string())
                            f.write("\n\n")
                    except Exception as e:
                        f.write(f"Failed to generate cycle-by-cycle comparison: {e}\n\n")

                # Overall statistics (only when labels are available)
                if labels_available:
                    f.write("Overall Statistics Summary:\n")
                    f.write("-"*40 + "\n")

                    try:
                        valid_comparison = comparison_df[comparison_df['mAP50'] >= 0]
                        if len(valid_comparison) > 0:
                            stats = valid_comparison.groupby('Experiment')[['mAP50', 'Precision', 'Recall', 'F1-Score']].agg(['mean', 'std', 'max'])
                            f.write(stats.to_string())
                        else:
                            f.write("No measurable performance metrics available.\n")
                    except Exception as e:
                        f.write(f"Failed to generate statistics summary: {e}\n")

            print(f"Comparison report saved: {output_path}")

        except Exception as e:
            print(f"Warning: Failed to generate comparison report: {e}")
