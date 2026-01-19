"""
detector.py - Integrated object detection and classification module (with captioning classifier support)
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional, Dict, Any, Union
from classifier import ObjectClassifier
from captioning_classifier import ImageCaptioningClassifier
from utils import xyxy_to_yolo

class ObjectDetector:
    """Integrated object detection and classification class (with captioning classifier support)"""

    def __init__(self, model_path: str,
                 classifier: Optional[Union[ObjectClassifier, ImageCaptioningClassifier]] = None,
                 conf_threshold: float = 0.25, iou_threshold: float = 0.5):
        """
        Initialize object detector

        Args:
            model_path: YOLO model path
            classifier: Classification model (traditional classifier or captioning classifier)
            conf_threshold: Detection confidence threshold
            iou_threshold: IoU threshold
        """
        self.model = YOLO(model_path)
        self.classifier = classifier
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.use_classifier = classifier is not None

        # Check classifier type
        self.classifier_type = "none"
        if isinstance(classifier, ImageCaptioningClassifier):
            self.classifier_type = "captioning"
        elif isinstance(classifier, ObjectClassifier):
            self.classifier_type = "traditional"

        # Statistics
        self.stats = {
            'total_detections': 0,
            'filtered_detections': 0,
            'classification_calls': 0
        }

        print(f"Object detector initialized:")
        print(f"  - Classifier type: {self.classifier_type}")

    def detect_and_classify(self, image_path: str, cycle: int = 0) -> Tuple[List, List, np.ndarray, Optional[np.ndarray]]:
        """
        Perform object detection and classification

        Args:
            image_path: Image path
            cycle: Current cycle

        Returns:
            (detected objects, filtered objects, visualization image 1, visualization image 2)
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return [], [], np.array([]), None

        h, w = img.shape[:2]

        # Object detection
        detected_boxes = self._detect_with_yolo(img)

        # Process results
        detected_objects = []
        filtered_objects = []

        img_with_all_boxes = img.copy()
        img_with_filtered_boxes = img.copy() if self.use_classifier else None

        # In Cycle 0, disable classification filtering to measure baseline
        # From Cycle 1 onwards, apply classification filtering to measure progressive improvement
        apply_filtering = cycle > 0 and self.use_classifier and self.classifier is not None

        # Filtering count for current image
        current_filtered_count = 0

        for box_data in detected_boxes:
            x1, y1, x2, y2 = box_data['xyxy']
            conf = box_data['conf']
            cls_id = box_data['cls']

            # Extract object image
            obj_img = img[int(y1):int(y2), int(x1):int(x2)]

            # Convert to YOLO format
            center_x, center_y, width, height = xyxy_to_yolo((x1, y1, x2, y2), w, h)

            # Apply classification filtering
            if apply_filtering and obj_img.size > 0:
                pred_class, class_conf = self.classifier.classify(obj_img)
                self.stats['classification_calls'] += 1

                if pred_class == 0:  # Positive object (keep)
                    detected_objects.append([cls_id, center_x, center_y, width, height])
                    # Visualize on all detections
                    self._draw_detection(img_with_all_boxes, x1, y1, x2, y2, conf, "Y", (0, 255, 0))
                    # Also visualize on filtered results
                    if img_with_filtered_boxes is not None:
                        # Label based on classifier type
                        label = "C0" if self.classifier_type == "traditional" else "K0"  # K0 = Keyword match
                        self._draw_detection(img_with_filtered_boxes, x1, y1, x2, y2, class_conf, label, (0, 255, 0))
                else:  # Negative object (filter out)
                    filtered_objects.append([cls_id, center_x, center_y, width, height])
                    # Visualize in red on all detections
                    label = "C1" if self.classifier_type == "traditional" else "K1"  # K1 = Keyword not match
                    self._draw_detection(img_with_all_boxes, x1, y1, x2, y2, class_conf, label, (0, 0, 255))
                    self.stats['filtered_detections'] += 1
                    current_filtered_count += 1
            else:
                # Cycle 0 or no classifier: keep all objects
                detected_objects.append([cls_id, center_x, center_y, width, height])
                # Visualize with YOLO label
                self._draw_detection(img_with_all_boxes, x1, y1, x2, y2, conf, "Y", (0, 255, 0))

        self.stats['total_detections'] += len(detected_boxes)

        return detected_objects, filtered_objects, img_with_all_boxes, img_with_filtered_boxes

    def _detect_with_yolo(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """Object detection using YOLO"""
        results = self.model.predict(
            source=img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=False,
            verbose=False
        )

        result = results[0]
        detected_boxes = []

        if len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()

                detected_boxes.append({
                    'xyxy': [x1, y1, x2, y2],
                    'conf': conf,
                    'cls': 0  # Single class
                })

        return detected_boxes

    def _draw_detection(self, img: np.ndarray, x1: float, y1: float, x2: float, y2: float,
                       conf: float, label: str, color: Tuple[int, int, int]):
        """Visualize detection results"""
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, f"{label}:{conf:.2f}",
                   (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def update_yolo_model(self, new_model_path: str):
        """Update YOLO model"""
        self.model = YOLO(new_model_path)

    def update_classifier(self, new_classifier: Union[ObjectClassifier, ImageCaptioningClassifier]):
        """Update classification model - with captioning classifier support"""
        self.classifier = new_classifier
        self.use_classifier = new_classifier is not None

        # Update classifier type
        if isinstance(new_classifier, ImageCaptioningClassifier):
            self.classifier_type = "captioning"
            print("Classifier updated: captioning classifier")
        elif isinstance(new_classifier, ObjectClassifier):
            self.classifier_type = "traditional"
            print("Classifier updated: traditional classifier")
        else:
            self.classifier_type = "none"
            print("Classifier updated: none")

    def get_stats(self) -> Dict[str, int]:
        """Return detection statistics"""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_detections': 0,
            'filtered_detections': 0,
            'classification_calls': 0
        }

    def get_config(self) -> Dict[str, Any]:
        """Return detector configuration - with captioning classifier info"""
        config = {
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'use_classifier': self.use_classifier,
            'classifier_type': self.classifier_type
        }

        if self.classifier:
            if self.classifier_type == "captioning":
                config['classifier_info'] = self.classifier.get_model_info()
            elif self.classifier_type == "traditional":
                config['classifier_info'] = self.classifier.get_model_info()

        return config

class CroppedObjectCollector:
    """Cropped object collection class (with captioning classifier support)"""

    def __init__(self, detector: ObjectDetector):
        self.detector = detector

    def collect_objects(self, image_files: List[str], output_dir: str, cycle: int,
                       manual_label_dir: Optional[str] = None) -> Dict[str, int]:
        """
        Collect cropped object images

        Args:
            image_files: List of image file paths
            output_dir: Output directory
            cycle: Current cycle
            manual_label_dir: Manual labeling directory

        Returns:
            Dictionary of collected object counts
        """
        import os
        from tqdm import tqdm
        import random

        class0_dir = os.path.join(output_dir, "class0")
        class1_dir = os.path.join(output_dir, "class1")
        os.makedirs(class0_dir, exist_ok=True)
        os.makedirs(class1_dir, exist_ok=True)

        # Print directory description based on classifier type
        if self.detector.classifier_type == "captioning":
            print(f"Captioning object collection directories:")
            print(f"  - class0 (keyword match): {class0_dir}")
            print(f"  - class1 (keyword not match): {class1_dir}")
        else:
            print(f"Object collection directories:")
            print(f"  - class0: {class0_dir}")
            print(f"  - class1: {class1_dir}")

        collected_objects = {'class0': 0, 'class1': 0}

        desc = f"Collecting objects (cycle {cycle})"
        with tqdm(total=len(image_files), desc=desc, unit="img") as pbar:
            for image_file in image_files:
                img = cv2.imread(image_file)
                if img is None:
                    pbar.update(1)
                    continue

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

                    # Validate box coordinates
                    if x1 >= x2 or y1 >= y2:
                        continue

                    # Clip to image boundaries
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

                        # Exclude very low confidence (ensure data quality)
                        # For captioning classifier, confidence is 0 or 1 due to keyword matching
                        if self.detector.classifier_type == "captioning":
                            # Captioning classifier is keyword-based, accept all results
                            pass
                        else:
                            # Traditional classifier uses confidence threshold
                            if class_conf < 0.3:
                                continue
                    else:
                        # Without classifier, treat all objects as class0 (positive)
                        pred_class = 0
                        class_conf = conf

                    # Generate filename
                    base_name = os.path.splitext(os.path.basename(image_file))[0]

                    # Construct filename based on classifier type
                    if self.detector.classifier_type == "captioning":
                        obj_filename = f"{base_name}_cycle{cycle}_obj{idx}_cap{class_conf:.3f}.jpg"
                    else:
                        obj_filename = f"{base_name}_cycle{cycle}_obj{idx}_conf{class_conf:.3f}.jpg"

                    # Save object
                    try:
                        if pred_class == 0:
                            obj_save_path = os.path.join(class0_dir, obj_filename)
                            success = cv2.imwrite(obj_save_path, obj_img)
                            if success:
                                collected_objects['class0'] += 1
                            else:
                                print(f"Warning: Failed to save image: {obj_save_path}")
                        else:
                            obj_save_path = os.path.join(class1_dir, obj_filename)
                            success = cv2.imwrite(obj_save_path, obj_img)
                            if success:
                                collected_objects['class1'] += 1
                            else:
                                print(f"Warning: Failed to save image: {obj_save_path}")
                    except Exception as e:
                        print(f"Warning: Error saving object: {e}")
                        continue

                pbar.update(1)
                pbar.set_postfix({
                    'class0': collected_objects['class0'],
                    'class1': collected_objects['class1']
                })

        # Print collection results
        if self.detector.classifier_type == "captioning":
            print(f"\nCaptioning object collection completed:")
            print(f"  - Keyword match (class0): {collected_objects['class0']} objects")
            print(f"  - Keyword not match (class1): {collected_objects['class1']} objects")
        else:
            print(f"\nObject collection completed:")
            print(f"  - class0: {collected_objects['class0']} objects")
            print(f"  - class1: {collected_objects['class1']} objects")

        # Verify collected files
        class0_files = len([f for f in os.listdir(class0_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        class1_files = len([f for f in os.listdir(class1_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

        print(f"Directory verification:")
        print(f"  - class0 actual files: {class0_files} objects")
        print(f"  - class1 actual files: {class1_files} objects")

        # Update with actual file counts
        collected_objects['class0'] = class0_files
        collected_objects['class1'] = class1_files

        return collected_objects
