"""
active_learning.py - Main Active Learning class with captioning classifier support and per-cycle time tracking
"""

import os
import cv2
import yaml
import shutil
import numpy as np
from tqdm import tqdm
from typing import Optional, List, Tuple, Dict
from ultralytics import YOLO
import time
import json

from config import ExperimentConfig
from utils import (set_seed, get_device, create_experiment_directory_structure,
                  get_image_files, save_yolo_labels, Timer, load_yolo_labels,
                  yolo_to_xyxy, calculate_iou)
from classifier import ObjectClassifier, ClassifierTrainer
from captioning_classifier import ImageCaptioningClassifier, CaptioningClassifierTrainer
from detector import ObjectDetector, CroppedObjectCollector
from evaluator import PerformanceEvaluator, MetricsManager

class YOLOActiveLearning:
    """Main YOLO Active Learning class with captioning classifier support and per-cycle time tracking"""

    def __init__(self, config: ExperimentConfig, model_path: str, classifier_path: Optional[str] = None):
        """
        Initialize Active Learning system

        Args:
            config: Experiment configuration
            model_path: YOLO model path
            classifier_path: Classifier model path (for using existing classifier)
        """
        self.config = config
        self.model_path = model_path
        self.classifier_path = classifier_path

        # Set seed
        set_seed(config.global_seed)

        # Set device
        self.device = get_device(config.gpu_num)

        # Model name
        self.model_name = os.path.splitext(os.path.basename(model_path))[0]

        # Create directory structure
        self.output_dir = os.path.join(config.output_dir, self.model_name)
        create_experiment_directory_structure(
            self.output_dir,
            config.max_cycles,
            config.use_classifier or config.use_captioning_classifier,
            config.enable_classifier_retraining and not config.use_captioning_classifier
        )

        # Initialize components
        self._initialize_components()

        # Timer
        self.timer = Timer()

        # Track current cycle
        self.current_cycle = 0

        # Variables for per-cycle time tracking
        self.cycle_times = {}
        self.cycle_start_time = None
        self.cycle_step_times = {}

        # Check classifier type
        classifier_type = "captioning" if config.use_captioning_classifier else "standard"
        print(f"Active Learning initialized - Model: {self.model_name}")
        print(f"Classifier type: {classifier_type}")
        print(f"Output directory: {self.output_dir}")

    def _start_cycle_timing(self, cycle: int):
        """Start cycle time measurement"""
        self.cycle_start_time = time.time()
        self.cycle_step_times[cycle] = {}
        print(f"⏱️ Cycle {cycle} time measurement started")

    def _end_cycle_timing(self, cycle: int):
        """End cycle time measurement and save"""
        if self.cycle_start_time is not None:
            cycle_duration = time.time() - self.cycle_start_time
            self.cycle_times[cycle] = cycle_duration

            # Save per-cycle time information to file
            self._save_cycle_time_info(cycle, cycle_duration)

            print(f"⏱️ Cycle {cycle} completed in: {cycle_duration/60:.2f} minutes")

            # Calculate cumulative average time
            if len(self.cycle_times) > 1:
                avg_time = sum(self.cycle_times.values()) / len(self.cycle_times)
                print(f"⏱️ Average cycle time: {avg_time/60:.2f} minutes (total {len(self.cycle_times)} cycles)")

    def _record_step_time(self, cycle: int, step_name: str, duration: float):
        """Record step time within a cycle"""
        if cycle not in self.cycle_step_times:
            self.cycle_step_times[cycle] = {}
        self.cycle_step_times[cycle][step_name] = duration

    def _save_cycle_time_info(self, cycle: int, total_duration: float):
        """Save per-cycle time information to file"""
        cycle_dir = os.path.join(self.output_dir, f"cycle_{cycle}")
        time_info_path = os.path.join(cycle_dir, "cycle_timing.json")

        # Construct time information
        time_info = {
            "cycle": cycle,
            "total_duration_seconds": total_duration,
            "total_duration_minutes": total_duration / 60,
            "start_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.cycle_start_time)),
            "end_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.cycle_start_time + total_duration)),
            "step_times": self.cycle_step_times.get(cycle, {}),
            "model_name": self.model_name,
            "classifier_type": "captioning" if self.config.use_captioning_classifier else "standard" if self.config.use_classifier else "none"
        }

        # Add step times in minutes as well
        if cycle in self.cycle_step_times:
            time_info["step_times_minutes"] = {
                step: duration/60 for step, duration in self.cycle_step_times[cycle].items()
            }

        # Save as JSON file
        try:
            with open(time_info_path, 'w', encoding='utf-8') as f:
                json.dump(time_info, f, indent=2, ensure_ascii=False)
            print(f"⏱️ Cycle {cycle} time information saved: {time_info_path}")
        except Exception as e:
            print(f"⚠️ Failed to save Cycle {cycle} time information: {e}")

    def _save_overall_timing_summary(self):
        """Save overall timing summary for all cycles"""
        summary_path = os.path.join(self.output_dir, "cycle_timing_summary.json")

        # Construct overall summary information
        summary = {
            "experiment_info": {
                "model_name": self.model_name,
                "classifier_type": "captioning" if self.config.use_captioning_classifier else "standard" if self.config.use_classifier else "none",
                "total_cycles": len(self.cycle_times),
                "experiment_date": time.strftime('%Y-%m-%d %H:%M:%S')
            },
            "cycle_times_seconds": self.cycle_times,
            "cycle_times_minutes": {cycle: duration/60 for cycle, duration in self.cycle_times.items()},
            "statistics": {
                "total_time_seconds": sum(self.cycle_times.values()),
                "total_time_minutes": sum(self.cycle_times.values()) / 60,
                "average_cycle_time_seconds": sum(self.cycle_times.values()) / len(self.cycle_times) if self.cycle_times else 0,
                "average_cycle_time_minutes": (sum(self.cycle_times.values()) / len(self.cycle_times)) / 60 if self.cycle_times else 0,
                "fastest_cycle": min(self.cycle_times.items(), key=lambda x: x[1]) if self.cycle_times else None,
                "slowest_cycle": max(self.cycle_times.items(), key=lambda x: x[1]) if self.cycle_times else None
            },
            "detailed_step_times": self.cycle_step_times
        }

        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"⏱️ Overall cycle time summary saved: {summary_path}")

            # Generate human-readable text summary as well
            self._save_human_readable_timing_summary()

        except Exception as e:
            print(f"⚠️ Failed to save overall time summary: {e}")

    def _save_human_readable_timing_summary(self):
        """Save human-readable time summary"""
        summary_path = os.path.join(self.output_dir, "cycle_timing_summary.txt")

        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("Active Learning Per-Cycle Time Summary\n")
                f.write("=" * 60 + "\n\n")

                f.write(f"Model: {self.model_name}\n")
                f.write(f"Classifier: {'captioning' if self.config.use_captioning_classifier else 'standard' if self.config.use_classifier else 'none'}\n")
                f.write(f"Total cycles: {len(self.cycle_times)}\n")
                f.write(f"Experiment date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                if self.cycle_times:
                    f.write("Per-cycle duration:\n")
                    f.write("-" * 30 + "\n")
                    for cycle in sorted(self.cycle_times.keys()):
                        duration = self.cycle_times[cycle]
                        f.write(f"Cycle {cycle:2d}: {duration/60:6.2f} min ({duration:7.1f} sec)\n")

                    f.write("\n")
                    f.write("Statistics:\n")
                    f.write("-" * 30 + "\n")
                    total_time = sum(self.cycle_times.values())
                    avg_time = total_time / len(self.cycle_times)
                    fastest = min(self.cycle_times.items(), key=lambda x: x[1])
                    slowest = max(self.cycle_times.items(), key=lambda x: x[1])

                    f.write(f"Total time: {total_time/60:.2f} min ({total_time:.1f} sec)\n")
                    f.write(f"Average cycle time: {avg_time/60:.2f} min ({avg_time:.1f} sec)\n")
                    f.write(f"Fastest cycle: Cycle {fastest[0]} ({fastest[1]/60:.2f} min)\n")
                    f.write(f"Slowest cycle: Cycle {slowest[0]} ({slowest[1]/60:.2f} min)\n")

                    # Step-wise time analysis (if available)
                    if self.cycle_step_times:
                        f.write("\nAverage step times:\n")
                        f.write("-" * 30 + "\n")

                        # Collect all step names
                        all_steps = set()
                        for cycle_steps in self.cycle_step_times.values():
                            all_steps.update(cycle_steps.keys())

                        for step in sorted(all_steps):
                            step_times = []
                            for cycle_steps in self.cycle_step_times.values():
                                if step in cycle_steps:
                                    step_times.append(cycle_steps[step])

                            if step_times:
                                avg_step_time = sum(step_times) / len(step_times)
                                f.write(f"{step}: {avg_step_time/60:.2f} min (average)\n")

                f.write("\n" + "=" * 60 + "\n")

            print(f"⏱️ Human-readable time summary saved: {summary_path}")

        except Exception as e:
            print(f"⚠️ Failed to save text time summary: {e}")

    def _initialize_components(self):
        """Initialize system components - with captioning classifier support"""
        # Initialize classifier model
        if self.config.use_captioning_classifier:
            # Use captioning classifier
            print("Initializing captioning classifier...")
            self.classifier = ImageCaptioningClassifier(
                target_keywords=self.config.target_keywords,
                model_type=self.config.captioning_model_type,
                device=self.device,
                conf_threshold=self.config.class_conf_threshold,
                gpu_num=self.config.gpu_num
            )
            print(f"✓ Captioning classifier initialized (keywords: {self.config.target_keywords})")

        elif self.config.use_classifier and self.classifier_path:
            # Use standard classifier
            print("Initializing standard classifier model...")
            self.classifier = ObjectClassifier(
                self.classifier_path,
                self.device,
                self.config.class_conf_threshold,
                self.config.gpu_num
            )
            print("✓ Standard classifier model initialized")
        else:
            self.classifier = None
            print("Not using classifier")

        # Initialize object detector
        self.detector = ObjectDetector(
            self.model_path,
            self.classifier,
            None,  # No SAHI inference
            self.config.conf_threshold,
            self.config.iou_threshold
        )

        # Initialize classifier trainer
        if self.config.use_captioning_classifier:
            # Captioning classifier trainer
            self.classifier_trainer = CaptioningClassifierTrainer(self.device)
            print("Captioning classifier trainer initialized (no retraining)")

        elif self.config.use_classifier and self.config.enable_classifier_retraining:
            # Standard classifier trainer
            self.classifier_trainer = ClassifierTrainer(
                self.device,
                self.config.max_samples_per_class,
                self.config.classifier_batch_size,
                self.config.classifier_epochs,
                self.config.classifier_learning_rate_new,
                self.config.classifier_learning_rate_finetune
            )
            print("Standard classifier trainer initialized")
        else:
            self.classifier_trainer = None

        # Initialize cropped object collector
        if ((self.config.use_classifier and self.config.enable_classifier_retraining) or
            self.config.use_captioning_classifier):
            self.object_collector = CroppedObjectCollector(self.detector)
        else:
            self.object_collector = None

        # Initialize performance evaluator
        self.evaluator = PerformanceEvaluator(
            self.detector,
            self.config.image_dir,
            self.config.label_dir,
            self.config.iou_threshold
        )

        # Initialize metrics manager
        self.metrics_manager = MetricsManager(self.output_dir)

        # Dataset directory
        self.dataset_dir = os.path.join(self.output_dir, "dataset")

    def load_classifier_for_cycle(self, cycle: int) -> bool:
        """
        Load classifier model for specific cycle - with captioning classifier support

        Args:
            cycle: Cycle to load

        Returns:
            Whether loading was successful
        """
        if not (self.config.use_classifier or self.config.use_captioning_classifier):
            return False

        # Captioning classifier doesn't retrain, so always use the same model
        if self.config.use_captioning_classifier:
            print(f"Cycle {cycle}: Using captioning classifier (no retraining)")
            return True

        # Standard classifier logic
        # Cycle 1 uses initial model
        if cycle == 1:
            print(f"Cycle 1: Using initial classifier model")
            return True

        # Model path for Cycle 2 and above
        cycle_model_path = os.path.join(
            self.output_dir, f"cycle_{cycle}", "classification_training", "best_classifier.pth"
        )

        if os.path.exists(cycle_model_path):
            print(f"Loading trained classifier for Cycle {cycle}: {cycle_model_path}")
            self.classifier = ObjectClassifier(
                cycle_model_path,
                self.device,
                self.config.class_conf_threshold,
                self.config.gpu_num
            )
            self.detector.update_classifier(self.classifier)
            return True
        else:
            print(f"⚠️ Cannot find model for Cycle {cycle}: {cycle_model_path}")
            return False

    def run_inference_cycle(self, cycle: int, collect_data: bool = False) -> Tuple[bool, Optional[Dict[str, int]]]:
        """
        Run inference cycle (unified version) - can run even without labels
        With captioning classifier support

        Args:
            cycle: Current cycle
            collect_data: Whether to collect classification data

        Returns:
            (success status, collected data dictionary)
        """
        cycle_dir = os.path.join(self.output_dir, f"cycle_{cycle}")
        detections_dir = os.path.join(cycle_dir, "detections")
        labels_dir = os.path.join(cycle_dir, "labels")

        if self.config.use_classifier or self.config.use_captioning_classifier:
            filtered_detections_dir = os.path.join(cycle_dir, "filtered_detections")
            filtered_labels_dir = os.path.join(cycle_dir, "filtered_labels")

        # Prepare classification data collection directory
        collected_data = {'class0': 0, 'class1': 0}
        if collect_data and (self.config.use_classifier or self.config.use_captioning_classifier):
            cropped_objects_dir = os.path.join(cycle_dir, "cropped_objects")
            class0_dir = os.path.join(cropped_objects_dir, "class0")
            class1_dir = os.path.join(cropped_objects_dir, "class1")
            os.makedirs(class0_dir, exist_ok=True)
            os.makedirs(class1_dir, exist_ok=True)
            print(f"Classification data collection directory: {cropped_objects_dir}")

        # Image file list
        image_files = get_image_files(self.config.image_dir)
        if not image_files:
            print(f"Cannot find images: {self.config.image_dir}")
            return False, None

        # Normalize image file paths
        normalized_image_files = []
        for img_file in image_files:
            if os.path.isabs(img_file):
                normalized_path = os.path.normpath(img_file)
            else:
                normalized_path = os.path.normpath(os.path.join(self.config.image_dir, img_file))

            if os.path.exists(normalized_path):
                normalized_image_files.append(normalized_path)
            else:
                base_name = os.path.basename(img_file)
                alternative_path = os.path.normpath(os.path.join(self.config.image_dir, base_name))
                if os.path.exists(alternative_path):
                    normalized_image_files.append(alternative_path)
                else:
                    print(f"⚠️ Image file not found: {img_file} -> {normalized_path}")

        if not normalized_image_files:
            print(f"Cannot find valid image files.")
            return False, None

        print(f"Images to process: {len(normalized_image_files)}/{len(image_files)}")
        image_files = normalized_image_files

        detected_objects_count = 0
        filtered_objects_count = 0

        # Variables for performance evaluation (only when labels are available)
        all_precisions = []
        all_recalls = []
        all_f1_scores = []

        # Modify progress description
        desc_parts = [f"Inference (cycle {cycle})"]
        if collect_data:
            desc_parts.append("+ data collection")
        if self.config.labels_available:
            desc_parts.append("+ performance eval")
        else:
            desc_parts.append("(no performance eval)")
        desc = " ".join(desc_parts)

        with tqdm(total=len(image_files), desc=desc, unit="img") as pbar:
            for image_path in image_files:
                # 1. Object detection and classification
                detected_objects, filtered_objects, img_all, img_filtered = self.detector.detect_and_classify(image_path, cycle)

                detected_objects_count += len(detected_objects)
                filtered_objects_count += len(filtered_objects)

                # 2. Save results
                base_name = os.path.basename(image_path)
                if img_all is not None and img_all.size > 0:
                    try:
                        cv2.imwrite(os.path.join(detections_dir, base_name), img_all)
                    except Exception as e:
                        print(f"⚠️ Failed to save image ({base_name}): {e}")

                if (self.config.use_classifier or self.config.use_captioning_classifier) and img_filtered is not None and img_filtered.size > 0:
                    try:
                        cv2.imwrite(os.path.join(filtered_detections_dir, base_name), img_filtered)
                    except Exception as e:
                        print(f"⚠️ Failed to save filtered image ({base_name}): {e}")

                # 3. Save labels
                label_name = os.path.splitext(base_name)[0] + '.txt'

                try:
                    if self.config.use_classifier or self.config.use_captioning_classifier:
                        all_objects = detected_objects + filtered_objects
                        save_yolo_labels(all_objects, os.path.join(labels_dir, label_name))
                        save_yolo_labels(detected_objects, os.path.join(filtered_labels_dir, label_name))
                    else:
                        save_yolo_labels(detected_objects, os.path.join(labels_dir, label_name))
                except Exception as e:
                    print(f"⚠️ Failed to save labels ({label_name}): {e}")

                # 4. Collect classification data (if needed)
                if collect_data and (self.config.use_classifier or self.config.use_captioning_classifier):
                    img = cv2.imread(image_path)
                    if img is not None and img.size > 0:
                        try:
                            collected_count = self._collect_objects_from_image(
                                img, image_path, cycle, class0_dir, class1_dir
                            )
                            collected_data['class0'] += collected_count['class0']
                            collected_data['class1'] += collected_count['class1']
                        except Exception as e:
                            print(f"⚠️ Failed to collect objects ({base_name}): {e}")

                # 5. Performance evaluation (only when labels are available)
                if self.config.labels_available:
                    try:
                        gt_label_path = os.path.join(self.config.label_dir, label_name)
                        if os.path.exists(gt_label_path):
                            gt_objects = load_yolo_labels(gt_label_path)
                            precision, recall, f1 = self._calculate_metrics(detected_objects, gt_objects, image_path)
                            all_precisions.append(precision)
                            all_recalls.append(recall)
                            all_f1_scores.append(f1)
                        else:
                            all_precisions.append(0.0)
                            all_recalls.append(0.0)
                            all_f1_scores.append(0.0)
                    except Exception as e:
                        print(f"⚠️ Performance evaluation failed ({base_name}): {e}")
                        all_precisions.append(0.0)
                        all_recalls.append(0.0)
                        all_f1_scores.append(0.0)

                # Update progress
                postfix = {'detected': detected_objects_count}
                if self.config.use_classifier or self.config.use_captioning_classifier:
                    postfix['filtered'] = filtered_objects_count
                if collect_data:
                    postfix['class0'] = collected_data['class0']
                    postfix['class1'] = collected_data['class1']

                pbar.set_postfix(postfix)
                pbar.update(1)

        # Print captioning classifier statistics
        if self.config.use_captioning_classifier and isinstance(self.classifier, ImageCaptioningClassifier):
            captioning_stats = self.classifier.get_stats()
            print(f"\nCaptioning classification statistics:")
            print(f"  - Total classifications: {captioning_stats['total_classifications']}")
            print(f"  - Positive classifications: {captioning_stats['positive_classifications']}")
            print(f"  - Negative classifications: {captioning_stats['negative_classifications']}")
            if captioning_stats['keyword_matches']:
                print(f"  - Keyword matches: {captioning_stats['keyword_matches']}")

            # Save captioning log
            captioning_log_path = os.path.join(cycle_dir, "captioning_classification_log.txt")
            self.classifier.export_classification_log(captioning_log_path)

        # Calculate and save performance metrics
        if self.config.labels_available and all_precisions:
            avg_precision = np.mean(all_precisions)
            avg_recall = np.mean(all_recalls)
            avg_f1 = np.mean(all_f1_scores)
        else:
            avg_precision = -1.0
            avg_recall = -1.0
            avg_f1 = -1.0

        # Save metrics
        metrics = {
            'Cycle': cycle,
            'Model': self.model_name,
            'mAP50': avg_precision,
            'Precision': avg_precision,
            'Recall': avg_recall,
            'F1-Score': avg_f1,
            'Detected_Objects': detected_objects_count,
            'Filtered_Objects': filtered_objects_count if cycle > 0 else 0,
            'Labels_Available': self.config.labels_available
        }

        self.metrics_manager.add_metrics(metrics)

        print(f"Cycle {cycle}: detected={detected_objects_count}, filtered={filtered_objects_count}")
        if self.config.labels_available:
            print(f"Performance: mAP50={avg_precision:.4f}, Precision={avg_precision:.4f}, Recall={avg_recall:.4f}")
        else:
            print("Performance: No labels - cannot measure")

        if collect_data:
            print(f"Collected data: class0={collected_data['class0']}, class1={collected_data['class1']}")
            return detected_objects_count > 0, collected_data
        else:
            return detected_objects_count > 0, None

    def _collect_objects_from_image(self, img: np.ndarray, image_path: str, cycle: int,
                                   class0_dir: str, class1_dir: str) -> Dict[str, int]:
        """Collect objects from a single image"""
        collected = {'class0': 0, 'class1': 0}

        # Object detection
        results = self.detector.model.predict(
            source=img,
            conf=self.detector.conf_threshold,
            iou=self.detector.iou_threshold,
            save=False,
            verbose=False
        )
        detected_boxes = []
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detected_boxes.append((x1, y1, x2, y2, conf))

        # Save objects
        for idx, box_data in enumerate(detected_boxes):
            x1, y1, x2, y2, conf = box_data

            # Validate and clip box coordinates
            if x1 >= x2 or y1 >= y2:
                continue

            h, w = img.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))

            obj_img = img[int(y1):int(y2), int(x1):int(x2)]

            # Validate object image
            if obj_img.size == 0 or obj_img.shape[0] < 10 or obj_img.shape[1] < 10:
                continue

            # Perform classification
            if self.detector.classifier:
                pred_class, class_conf = self.detector.classifier.classify(obj_img)
                if class_conf < 0.3:
                    continue
            else:
                pred_class = 0
                class_conf = conf

            # Generate filename and save
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            obj_filename = f"{base_name}_cycle{cycle}_obj{idx}_conf{class_conf:.3f}.jpg"

            try:
                if pred_class == 0:
                    obj_save_path = os.path.join(class0_dir, obj_filename)
                    success = cv2.imwrite(obj_save_path, obj_img)
                    if success:
                        collected['class0'] += 1
                else:
                    obj_save_path = os.path.join(class1_dir, obj_filename)
                    success = cv2.imwrite(obj_save_path, obj_img)
                    if success:
                        collected['class1'] += 1
            except Exception as e:
                continue

        return collected

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
        if img is None or img.size == 0:
            print(f"⚠️ Failed to load image: {image_path}")
            return 0.0, 0.0, 0.0

        h, w = img.shape[:2]

        # Convert YOLO format to xyxy
        gt_boxes = []
        for gt_obj in gt_objects:
            try:
                cls_id, center_x, center_y, width, height = gt_obj
                x1, y1, x2, y2 = yolo_to_xyxy((cls_id, center_x, center_y, width, height), w, h)
                gt_boxes.append((x1, y1, x2, y2))
            except Exception as e:
                print(f"⚠️ Failed to convert GT box: {e}")
                continue

        pred_boxes = []
        for det_obj in detected_objects:
            try:
                cls_id, center_x, center_y, width, height = det_obj
                x1, y1, x2, y2 = yolo_to_xyxy((cls_id, center_x, center_y, width, height), w, h)
                pred_boxes.append((x1, y1, x2, y2))
            except Exception as e:
                print(f"⚠️ Failed to convert prediction box: {e}")
                continue

        # Handle case when there are no converted boxes
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            return 1.0, 1.0, 1.0
        elif len(gt_boxes) == 0:
            return 0.0, 1.0, 0.0
        elif len(pred_boxes) == 0:
            return 1.0, 0.0, 0.0

        # Calculate True Positives
        tp = 0
        matched_gt = set()

        for pred_box in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue

                try:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                except Exception as e:
                    print(f"⚠️ Failed to calculate IoU: {e}")
                    continue

            if best_iou >= self.config.iou_threshold and best_gt_idx != -1:
                tp += 1
                matched_gt.add(best_gt_idx)

        # Calculate performance metrics
        precision = tp / len(pred_boxes) if len(pred_boxes) > 0 else 0.0
        recall = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    def collect_classification_data(self, cycle: int):
        """Collect classification training data (for standalone execution)"""
        if not ((self.config.use_classifier and self.config.enable_classifier_retraining) or
                self.config.use_captioning_classifier):
            return {'class0': 0, 'class1': 0}

        print(f"⚠️ Standalone data collection called - unified inference is recommended")

        image_files = get_image_files(self.config.image_dir)
        output_dir = os.path.join(self.output_dir, f"cycle_{cycle}", "cropped_objects")

        return self.object_collector.collect_objects(
            image_files,
            output_dir,
            cycle,
            self.config.manual_label_dir
        )

    def evaluate_performance(self, cycle: int):
        """Evaluate and save performance (for standalone execution)"""
        print(f"⚠️ Standalone performance evaluation called - unified inference is recommended")

        self.timer.start_phase(f"evaluation_cycle_{cycle}")

        metrics = self.evaluator.evaluate(cycle, self.model_name)
        self.metrics_manager.add_metrics(metrics)

        self.timer.end_phase()

        print(f"Cycle {cycle} performance: mAP50={metrics['mAP50']:.4f}, "
              f"Precision={metrics['Precision']:.4f}, Recall={metrics['Recall']:.4f}")

        return metrics

    def train_classifier(self, target_cycle: int) -> bool:
        """
        Train classifier model for specific cycle - with captioning classifier support

        Args:
            target_cycle: Target cycle where the trained model will be used

        Returns:
            Whether training was successful
        """
        if not ((self.config.use_classifier and self.config.enable_classifier_retraining) or
                self.config.use_captioning_classifier):
            return False

        # Determine data collection cycle
        data_cycle = target_cycle - 1
        cropped_data_dir = os.path.join(self.output_dir, f"cycle_{data_cycle}", "cropped_objects")

        # For captioning classifier
        if self.config.use_captioning_classifier:
            print(f"Captioning classifier for Target Cycle {target_cycle}:")
            print(f"  - No retraining (using pretrained model)")
            print(f"  - Keywords: {self.config.target_keywords}")

            # Create new captioning classifier (same settings)
            new_classifier = self.classifier_trainer.train_classifier(
                cropped_data_dir,
                None,  # Previous model path not needed
                self.config.manual_label_dir,
                target_cycle,
                target_keywords=self.config.target_keywords,
                model_type=self.config.captioning_model_type
            )

            if new_classifier:
                # Save configuration for target cycle
                classification_training_dir = os.path.join(self.output_dir, f"cycle_{target_cycle}", "classification_training")
                os.makedirs(classification_training_dir, exist_ok=True)
                config_save_path = os.path.join(classification_training_dir, "captioning_classifier_config.json")
                new_classifier.save_model(config_save_path)

                # Apply immediately if current cycle equals target cycle
                if target_cycle == self.current_cycle:
                    self.classifier = new_classifier
                    self.detector.update_classifier(new_classifier)
                    print(f"✓ Captioning classifier for Cycle {target_cycle} applied immediately")

                print(f"✓ Captioning classifier for Cycle {target_cycle} ready")
                return True

            return False

        # Standard classifier logic
        # Determine previous model path
        previous_model_path = None
        if target_cycle > 1:
            previous_model_path = os.path.join(
                self.output_dir, f"cycle_{target_cycle-1}", "classification_training", "best_classifier.pth"
            )
            if not os.path.exists(previous_model_path):
                previous_model_path = self.classifier_path
        else:
            previous_model_path = self.classifier_path

        print(f"Training standard classifier for Target Cycle {target_cycle}:")
        print(f"  - Data source: Cycle {data_cycle}")
        print(f"  - Previous model: {previous_model_path}")

        # Train classifier model
        new_classifier = self.classifier_trainer.train_classifier(
            cropped_data_dir,
            previous_model_path,
            self.config.manual_label_dir,
            target_cycle
        )

        if new_classifier:
            # Save model for target cycle
            classification_training_dir = os.path.join(self.output_dir, f"cycle_{target_cycle}", "classification_training")
            os.makedirs(classification_training_dir, exist_ok=True)
            model_save_path = os.path.join(classification_training_dir, "best_classifier.pth")
            new_classifier.save_model(model_save_path)

            # Apply immediately if current cycle equals target cycle
            if target_cycle == self.current_cycle:
                self.classifier = new_classifier
                self.detector.update_classifier(new_classifier)
                print(f"✓ Standard classifier for Cycle {target_cycle} applied immediately")

            print(f"✓ Standard classifier training for Cycle {target_cycle} completed")
            return True

        return False

    def prepare_yolo_dataset(self, cycle: int):
        """Prepare dataset for YOLO training"""
        # Determine labels directory
        if (self.config.use_classifier or self.config.use_captioning_classifier) and cycle > 1:
            labels_dir = os.path.join(self.output_dir, f"cycle_{cycle}", "filtered_labels")
        else:
            labels_dir = os.path.join(self.output_dir, f"cycle_{cycle}", "labels")

        image_files = [os.path.basename(f) for f in get_image_files(self.config.image_dir)]

        # Train/validation split
        if len(image_files) > 5:
            val_count = max(5, min(20, int(len(image_files) * 0.05)))
            val_files = image_files[:val_count]
            train_files = image_files[val_count:]
        else:
            val_files = image_files[:1]
            train_files = image_files[1:]

        # Clean existing dataset
        for split_dir in ["train", "val"]:
            for data_type in ["images", "labels"]:
                dir_path = os.path.join(self.dataset_dir, data_type, split_dir)
                if os.path.exists(dir_path):
                    for file in os.listdir(dir_path):
                        file_path = os.path.join(dir_path, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)

        # Copy dataset
        for split, files in [("train", train_files), ("val", val_files)]:
            for file in files:
                # Copy image
                src_img = os.path.join(self.config.image_dir, file)
                dst_img = os.path.join(self.dataset_dir, "images", split, file)
                shutil.copy(src_img, dst_img)

                # Copy label
                label_name = os.path.splitext(file)[0] + '.txt'
                src_label = os.path.join(labels_dir, label_name)

                if os.path.exists(src_label):
                    dst_label = os.path.join(self.dataset_dir, "labels", split, label_name)
                    shutil.copy(src_label, dst_label)

        # Generate YAML file
        dataset_yaml = {
            'path': os.path.abspath(self.dataset_dir),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,
            'names': ['object']
        }

        yaml_path = os.path.join(self.dataset_dir, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)

        print(f"Dataset prepared: train={len(train_files)}, val={len(val_files)}")
        return yaml_path

    def train_yolo_model(self, cycle: int) -> str:
        """Train YOLO model"""
        yaml_path = os.path.join(self.dataset_dir, 'dataset.yaml')
        results_dir = os.path.join(self.output_dir, f"cycle_{cycle}", "training")

        print(f"Starting YOLO model training - Cycle {cycle}")

        # Train model
        results = self.detector.model.train(
            data=yaml_path,
            epochs=self.config.yolo_epochs,
            imgsz=640,
            batch=self.config.yolo_batch_size,
            patience=self.config.yolo_patience,
            project=results_dir,
            name="yolo_model",
            device=self.device
        )

        # Trained model path
        trained_model_path = os.path.join(results_dir, "yolo_model", "weights", "best.pt")

        # Update model
        self.detector.update_yolo_model(trained_model_path)

        print(f"YOLO model training completed: {trained_model_path}")
        return trained_model_path

    def run(self, skip_cycle_0=False):
        """Run Active Learning process - with captioning classifier support and per-cycle time tracking"""
        print(f"\n{'='*60}")
        print(f"Starting Active Learning")
        print(f"Model: {self.model_name}")

        # Modify classifier info output
        if self.config.use_captioning_classifier:
            classifier_info = f"captioning({self.config.captioning_model_type})"
        elif self.config.use_classifier:
            classifier_info = f"standard(retrain={self.config.enable_classifier_retraining})"
        else:
            classifier_info = "none"

        print(f"Configuration: classifier={classifier_info}")
        print(f"Skip Cycle 0: {skip_cycle_0}")
        print(f"⏱️ Per-cycle time tracking: enabled")
        print(f"{'='*60}")

        self.timer.start()

        try:
            # Set correct starting cycle
            start_cycle = 1 if skip_cycle_0 else 0

            if not skip_cycle_0:
                # Cycle 0: Baseline measurement only
                self._start_cycle_timing(0)
                self.timer.start_phase("cycle_0")
                print(f"\n--- Cycle 0 - Baseline Performance Measurement ---")

                step_start = time.time()
                print("Step: Baseline inference and performance evaluation (no data collection)")
                success, _ = self.run_inference_cycle(0, collect_data=False)
                step_duration = time.time() - step_start
                self._record_step_time(0, "baseline_inference", step_duration)

                if not success:
                    raise Exception("No objects detected in initial inference.")

                self.timer.end_phase()
                self._end_cycle_timing(0)
                print("Cycle 0 completed: Baseline performance measured")
                start_cycle = 1
            else:
                print("\nSkipping Cycle 0 - Starting training cycles directly")

            # Run training cycles
            for cycle in range(start_cycle, self.config.max_cycles + 1):
                print(f"\n--- Cycle {cycle} ---")
                self.current_cycle = cycle
                self._start_cycle_timing(cycle)  # Start cycle time measurement

                cycle_inference_done = False

                # 0. Special handling for Cycle 1
                if cycle == 1:
                    step_start = time.time()
                    if self.config.use_captioning_classifier:
                        print("Step 0: Cycle 1 - Using captioning classifier (no retraining)")
                        print("✓ Skipping classifier training - using pretrained model as-is")
                    elif self.config.use_classifier:
                        print("Step 0: Cycle 1 - Using standard classifier")
                        print("✓ Skipping classifier training - using initial weights as-is")

                    if skip_cycle_0:
                        print("Step 0-1: skip_cycle_0=True - Running initial inference and data collection")
                        self.timer.start_phase("cycle_1_unified_inference")

                        collect_data = (self.config.use_classifier or self.config.use_captioning_classifier)
                        success, collected_data = self.run_inference_cycle(1, collect_data=collect_data)

                        if not success:
                            print(f"Cycle 1 initial inference failed")
                            continue

                        cycle_inference_done = True
                        self.timer.end_phase()

                    step_duration = time.time() - step_start
                    self._record_step_time(cycle, "initial_setup", step_duration)

                # 1. Load classifier model for this cycle (from Cycle 2 onwards)
                elif cycle >= 2 and (self.config.use_classifier or self.config.use_captioning_classifier):
                    step_start = time.time()
                    print(f"Step 1: Loading classifier model for Cycle {cycle}")
                    success = self.load_classifier_for_cycle(cycle)
                    if success:
                        print(f"✓ Classifier model for Cycle {cycle} loaded")
                    else:
                        print(f"⚠️ Failed to load classifier for Cycle {cycle} - using previous model")

                    step_duration = time.time() - step_start
                    self._record_step_time(cycle, "model_loading", step_duration)

                # 2. Unified inference: inference + performance evaluation + data collection
                if not cycle_inference_done:
                    step_start = time.time()
                    self.timer.start_phase(f"cycle_{cycle}_unified_inference")

                    collect_data = (cycle >= 1 and
                                (self.config.use_classifier or self.config.use_captioning_classifier))

                    action_desc = f"inference + performance eval"
                    if collect_data:
                        action_desc += " + data collection"

                    print(f"Step 2: Unified inference ({action_desc})")

                    success, collected_data = self.run_inference_cycle(cycle, collect_data=collect_data)

                    if not success:
                        print(f"Cycle {cycle}: No objects detected, skipping")
                        continue

                    self.timer.end_phase()
                    step_duration = time.time() - step_start
                    self._record_step_time(cycle, "unified_inference", step_duration)
                else:
                    print("Step 2: Cycle 1 unified inference already completed - skipping")

                # 3. Train classifier model for next cycle
                if (cycle >= 1 and
                    cycle < self.config.max_cycles):

                    step_start = time.time()

                    # Captioning classifier doesn't retrain
                    if self.config.use_captioning_classifier:
                        print(f"Step 3: Captioning classifier doesn't retrain - skipping")

                    # Standard classifier retrains only when enabled and has sufficient data
                    elif (self.config.use_classifier and
                          self.config.enable_classifier_retraining and
                          collected_data and
                          collected_data.get('class1', 0) >= 5):

                        self.timer.start_phase(f"cycle_{cycle}_classification")
                        print(f"Step 3: Training standard classifier for Cycle {cycle+1}")

                        training_success = self.train_classifier(cycle + 1)
                        if training_success:
                            print(f"✓ Standard classifier training for Cycle {cycle+1} completed")
                        else:
                            print(f"⚠️ Standard classifier training for Cycle {cycle+1} failed")

                        self.timer.end_phase()
                    else:
                        if cycle == self.config.max_cycles:
                            print(f"Last cycle - skipping classifier training")
                        elif collected_data and collected_data:
                            print(f"⚠️ Insufficient data - skipping classifier training for Cycle {cycle+1}")

                    step_duration = time.time() - step_start
                    self._record_step_time(cycle, "classifier_training", step_duration)

                # 4. YOLO model training
                step_start = time.time()
                self.timer.start_phase(f"cycle_{cycle}_yolo_training")
                print(f"Step 4: YOLO model training")
                self.prepare_yolo_dataset(cycle)
                self.train_yolo_model(cycle)
                self.timer.end_phase()
                step_duration = time.time() - step_start
                self._record_step_time(cycle, "yolo_training", step_duration)

                self._end_cycle_timing(cycle)  # End cycle time measurement
                print(f"Cycle {cycle} completed")

            # Experiment completed
            total_time = self.timer.end()
            print(f"\n{'='*60}")
            print(f"Active Learning completed!")
            print(f"Total time: {total_time/60:.1f} minutes")
            print(f"Results directory: {self.output_dir}")

            # Print per-cycle time summary
            if self.cycle_times:
                print(f"\n⏱️ Per-cycle time summary:")
                for cycle in sorted(self.cycle_times.keys()):
                    duration = self.cycle_times[cycle]
                    print(f"   Cycle {cycle}: {duration/60:.2f} minutes")

                avg_cycle_time = sum(self.cycle_times.values()) / len(self.cycle_times)
                print(f"   Average cycle time: {avg_cycle_time/60:.2f} minutes")

            # Final performance summary
            best_performance = self.metrics_manager.get_best_performance()
            if best_performance:
                print(f"Best performance: mAP50={best_performance['mAP50']:.4f} (Cycle {best_performance['Cycle']})")

            # Generate performance summary file
            summary_path = os.path.join(self.output_dir, "performance_summary.txt")
            self.metrics_manager.export_summary(summary_path)

            # Generate time summary file (existing + new per-cycle time)
            time_summary_path = os.path.join(self.output_dir, "time_summary.txt")
            with open(time_summary_path, 'w') as f:
                f.write("Execution Time Summary\n")
                f.write("="*40 + "\n")
                f.write(self.timer.get_summary())

            # Save per-cycle time summary
            self._save_overall_timing_summary()

            print(f"Summary file: {summary_path}")
            print(f"⏱️ Per-cycle time summary: {os.path.join(self.output_dir, 'cycle_timing_summary.txt')}")
            print(f"{'='*60}")

        except Exception as e:
            print(f"\nError occurred during Active Learning execution: {str(e)}")
            raise
