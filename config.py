"""
config.py - Active Learning Experiment Configuration Module (VIT-GPT2 Captioning Classifier Support)
"""

import os
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ExperimentConfig:
    """Experiment configuration class"""

    # Path settings
    models_dir: str = "./models/yolo"
    classifiers_dir: str = "./models/classifiers"
    image_dir: str = "./data/images"
    label_dir: str = "./data/labels"
    output_dir: str = "./results"
    manual_label_dir: Optional[str] = None

    # Hardware settings
    gpu_num: int = 0

    # Training parameters
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    class_conf_threshold: float = 0.5
    max_cycles: int = 10
    max_samples_per_class: int = 100

    # Feature activation settings
    use_classifier: bool = True
    enable_classifier_retraining: bool = False

    # Captioning classifier settings (VIT-GPT2 support added)
    use_captioning_classifier: bool = False  # Whether to use captioning classifier
    captioning_model_type: str = "blip"      # Captioning model type ("blip", "blip2", "instructblip", "vit-gpt2")
    target_keywords: List[str] = None        # Positive object keyword list

    # Training settings
    yolo_epochs: int = 50
    yolo_batch_size: int = 16
    yolo_patience: int = 10
    classifier_epochs: int = 15
    classifier_batch_size: int = 16
    classifier_learning_rate_new: float = 0.001
    classifier_learning_rate_finetune: float = 0.0001

    # Seed settings
    global_seed: int = 42

    # Label availability
    labels_available: bool = True

    def __post_init__(self):
        """Post-initialization processing"""
        # Set default value for target_keywords
        if self.target_keywords is None:
            self.target_keywords = ['car']

    def validate(self):
        """Configuration validation - VIT-GPT2 captioning classifier validation added"""
        errors = []

        # Check path existence
        if not os.path.exists(self.models_dir):
            errors.append(f"YOLO model directory does not exist: {self.models_dir}")

        if not os.path.exists(self.image_dir):
            errors.append(f"Image directory does not exist: {self.image_dir}")

        # Classifier configuration validation
        if self.use_classifier and self.use_captioning_classifier:
            errors.append("Cannot use both traditional classifier and captioning classifier. Choose one.")

        if self.use_classifier and not self.use_captioning_classifier:
            if not os.path.exists(self.classifiers_dir):
                errors.append(f"Classifier model directory does not exist: {self.classifiers_dir}")

        if self.use_captioning_classifier:
            # Captioning model type validation (VIT-GPT2 added)
            valid_models = ["blip", "blip2", "instructblip", "vit-gpt2"]
            if self.captioning_model_type not in valid_models:
                errors.append(f"Captioning model type must be one of {valid_models}")

            # Target keyword validation
            if not self.target_keywords or len(self.target_keywords) == 0:
                errors.append("target_keywords is required when using captioning classifier")

            # Retraining setting validation
            if self.enable_classifier_retraining:
                print("⚠️ Captioning classifier does not support retraining. enable_classifier_retraining will be ignored.")

        # Parameter range validation
        if not 0.0 <= self.conf_threshold <= 1.0:
            errors.append("conf_threshold must be between 0.0 and 1.0")

        if not 0.0 <= self.iou_threshold <= 1.0:
            errors.append("iou_threshold must be between 0.0 and 1.0")

        if not 0.0 <= self.class_conf_threshold <= 1.0:
            errors.append("class_conf_threshold must be between 0.0 and 1.0")

        if self.max_cycles <= 0:
            errors.append("max_cycles must be positive")

        if self.max_samples_per_class <= 0:
            errors.append("max_samples_per_class must be positive")

        if errors:
            raise ValueError("\n".join(errors))

    def get_summary(self):
        """Return configuration summary - VIT-GPT2 captioning classifier information added"""
        summary = []
        summary.append("="*80)
        summary.append("Experiment Configuration Summary")
        summary.append("="*80)
        summary.append(f"YOLO Model Directory: {self.models_dir}")
        summary.append(f"Image Directory: {self.image_dir}")

        # Display label directory status
        label_status = "Available" if self.labels_available else "Unavailable (Performance evaluation excluded)"
        summary.append(f"Label Directory: {self.label_dir} ({label_status})")

        summary.append(f"Result Save Directory: {self.output_dir}")
        summary.append(f"Maximum Training Cycles: {self.max_cycles}")
        summary.append(f"GPU Number: {self.gpu_num}")
        summary.append("")

        # Classifier settings (VIT-GPT2 support added)
        summary.append("Classifier Settings:")
        if self.use_captioning_classifier:
            summary.append(f"  - Classification Method: Image Captioning")
            summary.append(f"  - Captioning Model: {self.captioning_model_type}")
            summary.append(f"  - Target Keywords: {', '.join(self.target_keywords)}")
            summary.append(f"  - Retraining: Not supported (using pretrained model)")
        elif self.use_classifier:
            summary.append(f"  - Classification Method: Traditional Classifier")
            summary.append(f"  - Classifier Model Directory: {self.classifiers_dir}")
            summary.append(f"  - Classifier Model Retraining: {self.enable_classifier_retraining}")
            summary.append(f"  - Max Samples per Class: {self.max_samples_per_class}")
        else:
            summary.append(f"  - Classification Method: Not used")

        summary.append("")
        summary.append("Other Feature Settings:")
        summary.append(f"  - Performance Evaluation: {self.labels_available}")
        summary.append("")
        summary.append("Threshold Settings:")
        summary.append(f"  - Detection Confidence: {self.conf_threshold}")
        summary.append(f"  - IoU Threshold: {self.iou_threshold}")
        if self.use_classifier or self.use_captioning_classifier:
            summary.append(f"  - Classification Confidence: {self.class_conf_threshold}")
        summary.append("="*80)
        return "\n".join(summary)

def load_config_from_dict(config_dict):
    """Load configuration from dictionary"""
    return ExperimentConfig(**config_dict)

def save_config_to_file(config: ExperimentConfig, filepath: str):
    """Save configuration to file"""
    import json
    from dataclasses import asdict

    config_dict = asdict(config)
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)

def load_config_from_file(filepath: str) -> ExperimentConfig:
    """Load configuration from file"""
    import json

    with open(filepath, 'r') as f:
        config_dict = json.load(f)

    return ExperimentConfig(**config_dict)
