"""
utils.py - Common Utility Functions Module
"""

import os
import random
import numpy as np
import torch
import time
import glob
from typing import List, Tuple

def set_seed(seed: int = 42):
    """Set global seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(gpu_num: int = 0) -> torch.device:
    """Get GPU device"""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_num}")
    else:
        return torch.device("cpu")

def create_directories(paths: List[str]):
    """Create list of directories"""
    for path in paths:
        os.makedirs(path, exist_ok=True)

def get_image_files(directory: str) -> List[str]:
    """Get list of image files"""
    if not os.path.exists(directory):
        return []

    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    files = []

    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        files.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}")))

    return sorted(files)

def get_model_files(directory: str, extension: str = ".pt") -> List[str]:
    """Get list of model files"""
    if not os.path.exists(directory):
        return []

    return sorted(glob.glob(os.path.join(directory, f"*{extension}")))

def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Calculate IoU of two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area

def yolo_to_xyxy(yolo_box: Tuple[float, float, float, float], img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """Convert YOLO format to xyxy format"""
    cls_id, center_x, center_y, width, height = yolo_box

    x1 = int((center_x - width/2) * img_width)
    y1 = int((center_y - height/2) * img_height)
    x2 = int((center_x + width/2) * img_width)
    y2 = int((center_y + height/2) * img_height)

    return x1, y1, x2, y2

def xyxy_to_yolo(xyxy_box: Tuple[int, int, int, int], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """Convert xyxy format to YOLO format"""
    x1, y1, x2, y2 = xyxy_box

    center_x = ((x1 + x2) / 2) / img_width
    center_y = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    return center_x, center_y, width, height

def load_yolo_labels(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """Load YOLO label file"""
    if not os.path.exists(label_path):
        return []

    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) >= 5:
                cls_id = int(values[0])
                center_x = float(values[1])
                center_y = float(values[2])
                width = float(values[3])
                height = float(values[4])
                labels.append((cls_id, center_x, center_y, width, height))

    return labels

def save_yolo_labels(labels: List[Tuple[int, float, float, float, float]], label_path: str):
    """Save YOLO label file"""
    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    with open(label_path, 'w') as f:
        for label in labels:
            line = ' '.join([str(x) for x in label])
            f.write(line + '\n')

def format_time(seconds: float) -> str:
    """Convert time to human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def format_bytes(bytes_size: int) -> str:
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}TB"

def get_system_info() -> dict:
    """Get system information"""
    import psutil
    import platform

    info = {
        'platform': platform.platform(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': format_bytes(psutil.virtual_memory().total),
        'memory_available': format_bytes(psutil.virtual_memory().available),
        'gpu_available': torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = format_bytes(torch.cuda.get_device_properties(0).total_memory)

    return info

def check_dependencies():
    """Check required libraries"""
    required_packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('ultralytics', 'ultralytics'),
        ('cv2', 'opencv-python'),
        ('PIL', 'pillow'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('tqdm', 'tqdm'),
        ('sklearn', 'scikit-learn'),
        ('yaml', 'pyyaml')
    ]

    missing_required = []

    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_required.append(package_name)

    if missing_required:
        raise ImportError(f"Required packages are missing: {missing_required}")

    return True

def create_experiment_directory_structure(base_dir: str, max_cycles: int, use_classifier: bool, enable_classifier_retraining: bool):
    """Create experiment directory structure"""
    directories = [base_dir]

    # Cycle directories
    for cycle in range(max_cycles + 1):
        cycle_dir = os.path.join(base_dir, f"cycle_{cycle}")
        directories.extend([
            cycle_dir,
            os.path.join(cycle_dir, "detections"),
            os.path.join(cycle_dir, "labels")
        ])

        if use_classifier:
            directories.extend([
                os.path.join(cycle_dir, "filtered_detections"),
                os.path.join(cycle_dir, "filtered_labels")
            ])

            if enable_classifier_retraining:
                directories.extend([
                    os.path.join(cycle_dir, "cropped_objects"),
                    os.path.join(cycle_dir, "cropped_objects", "class0"),
                    os.path.join(cycle_dir, "cropped_objects", "class1")
                ])

        if cycle > 0:
            directories.append(os.path.join(cycle_dir, "training"))

            if use_classifier and enable_classifier_retraining:
                directories.append(os.path.join(cycle_dir, "classification_training"))

    # Dataset directory
    dataset_dir = os.path.join(base_dir, "dataset")
    directories.extend([
        dataset_dir,
        os.path.join(dataset_dir, "images", "train"),
        os.path.join(dataset_dir, "images", "val"),
        os.path.join(dataset_dir, "labels", "train"),
        os.path.join(dataset_dir, "labels", "val")
    ])

    # Log directory
    directories.extend([
        os.path.join(base_dir, "logs"),
        os.path.join(base_dir, "configs")
    ])

    create_directories(directories)
    return directories

def log_experiment_start(config, log_dir: str):
    """Log experiment start"""
    log_file = os.path.join(log_dir, f"experiment_log_{int(time.time())}.txt")

    with open(log_file, 'w') as f:
        f.write(f"Experiment Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n" + config.get_summary() + "\n")
        f.write("\nSystem Information:\n")

        system_info = get_system_info()
        for key, value in system_info.items():
            f.write(f"  {key}: {value}\n")

    return log_file

class Timer:
    """Experiment time measurement class"""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.phase_times = {}
        self.current_phase = None

    def start(self):
        """Start timer"""
        self.start_time = time.time()
        return self

    def end(self):
        """End timer"""
        self.end_time = time.time()
        if self.current_phase:
            self.end_phase()
        return self.get_total_time()

    def start_phase(self, phase_name: str):
        """Start phase"""
        if self.current_phase:
            self.end_phase()
        self.current_phase = phase_name
        self.phase_times[phase_name] = {'start': time.time()}

    def end_phase(self):
        """End current phase"""
        if self.current_phase and self.current_phase in self.phase_times:
            self.phase_times[self.current_phase]['end'] = time.time()
            self.phase_times[self.current_phase]['duration'] = (
                self.phase_times[self.current_phase]['end'] -
                self.phase_times[self.current_phase]['start']
            )
        self.current_phase = None

    def get_total_time(self) -> float:
        """Get total execution time"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return 0.0

    def get_phase_time(self, phase_name: str) -> float:
        """Get specific phase time"""
        if phase_name in self.phase_times and 'duration' in self.phase_times[phase_name]:
            return self.phase_times[phase_name]['duration']
        return 0.0

    def get_summary(self) -> str:
        """Get time summary"""
        total_time = self.get_total_time()
        summary = [f"Total Execution Time: {format_time(total_time)}"]

        for phase_name, phase_data in self.phase_times.items():
            if 'duration' in phase_data:
                duration = phase_data['duration']
                percentage = (duration / total_time * 100) if total_time > 0 else 0
                summary.append(f"  {phase_name}: {format_time(duration)} ({percentage:.1f}%)")

        return "\n".join(summary)

def check_label_directory(label_dir: str, image_dir: str) -> dict:
    """
    Check label directory status

    Args:
        label_dir: Label directory path
        image_dir: Image directory path

    Returns:
        Label directory status information dictionary
    """
    status = {
        'exists': False,
        'label_count': 0,
        'image_count': 0,
        'matched_count': 0,
        'available': False
    }

    try:
        # Check image files
        image_files = get_image_files(image_dir)
        status['image_count'] = len(image_files)

        # Check label directory existence
        if not os.path.exists(label_dir):
            return status

        status['exists'] = True

        # Check label files
        label_files = [f for f in os.listdir(label_dir) if f.lower().endswith('.txt')]
        status['label_count'] = len(label_files)

        # Check matched file count
        image_basenames = {os.path.splitext(os.path.basename(f))[0] for f in image_files}
        label_basenames = {os.path.splitext(f)[0] for f in label_files}

        matched = image_basenames.intersection(label_basenames)
        status['matched_count'] = len(matched)

        # Determine availability (at least 1 match)
        status['available'] = status['matched_count'] > 0

    except Exception as e:
        print(f"⚠️ Error checking label directory: {e}")

    return status

def safe_load_yolo_labels(label_path: str) -> list:
    """
    Safe YOLO label loading (no error even if file does not exist)

    Args:
        label_path: Label file path

    Returns:
        Label list (empty list if file does not exist)
    """
    if not os.path.exists(label_path):
        return []

    try:
        return load_yolo_labels(label_path)
    except Exception as e:
        print(f"⚠️ Failed to load label file: {label_path} - {e}")
        return []
