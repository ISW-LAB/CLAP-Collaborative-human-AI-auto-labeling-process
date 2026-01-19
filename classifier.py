"""
classifier.py - Object Classification Model Module
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple, Optional
from utils import set_seed, get_image_files
import numpy as np

class ClassificationDataset(Dataset):
    """Dataset for classification model training"""

    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label

class ObjectClassifier:
    """Object classification model class"""

    def __init__(self, model_path: Optional[str] = None, device: Optional[torch.device] = None,
                 conf_threshold: float = 0.5, gpu_num: int = 0):
        """
        Initialize classification model

        Args:
            model_path: Pretrained model path (required)
            device: Computing device
            conf_threshold: Classification confidence threshold
            gpu_num: GPU number
        """
        if device is None:
            self.device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.conf_threshold = conf_threshold
        self.classifier_structure = "sequential"

        # Always create model structure without weights
        self.model = models.densenet121(weights=None)

        # Check model path and load
        if model_path and os.path.exists(model_path):
            print(f"Loading user-provided classification model: {model_path}")
            self._load_weights(model_path)

        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_weights(self, model_path: str):
        """Load model weights"""
        try:
            state_dict = torch.load(model_path, map_location=self.device)

            # Automatically detect classifier structure
            if 'classifier.1.weight' in state_dict:
                self.classifier_structure = "sequential"
            elif 'classifier.weight' in state_dict:
                self.classifier_structure = "linear"

            self._setup_classifier()
            self.model.load_state_dict(state_dict)
            print(f"✓ Classification model loaded successfully: {self.classifier_structure} structure")

        except RuntimeError as e:
            print(f"⚠️ Model loading failed, retrying after structure change: {e}")
            # Retry after structure change
            self.classifier_structure = "linear" if self.classifier_structure == "sequential" else "sequential"
            self._setup_classifier()

            try:
                new_state_dict = {}
                for key, value in state_dict.items():
                    if 'module.' in key:
                        key = key.replace('module.', '')
                    new_state_dict[key] = value
                self.model.load_state_dict(new_state_dict)
                print(f"✓ Classification model loaded successfully (structure changed): {self.classifier_structure}")
            except Exception as e2:
                print(f"✗ Model loading finally failed: {e2}")
                print("⚠️ Initializing with default sequential structure")
                self.classifier_structure = "sequential"
                self._setup_classifier()
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            print("⚠️ Initializing with default sequential structure")
            self.classifier_structure = "sequential"
            self._setup_classifier()

    def _setup_classifier(self):
        """Set up classifier structure"""
        num_features = self.model.classifier.in_features

        if self.classifier_structure == "sequential":
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 2)
            )
        else:
            self.model.classifier = nn.Linear(num_features, 2)

        print(f"Classifier head setup: {self.classifier_structure} ({num_features} → 2)")

    def classify(self, image: np.ndarray) -> Tuple[int, float]:
        """
        Classify object image

        Args:
            image: OpenCV format image (BGR)

        Returns:
            (predicted class, confidence) tuple
        """
        if image.shape[0] < 10 or image.shape[1] < 10:
            return 1, 0.0

        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                conf, predicted = torch.max(probabilities, 1)

            return predicted.item(), conf.item()
        except Exception as e:
            print(f"⚠️ Error during classification: {e}")
            return 1, 0.0

    def save_model(self, save_path: str):
        """Save model"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Classification model saved: {save_path}")

    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            'structure': self.classifier_structure,
            'device': str(self.device),
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'conf_threshold': self.conf_threshold
        }

class ClassifierTrainer:
    """Classification model training management class"""

    def __init__(self, device: torch.device, max_samples_per_class: int = 100,
                 batch_size: int = 16, num_epochs: int = 15,
                 lr_new: float = 0.001, lr_finetune: float = 0.0001):
        """
        Initialize classification model trainer

        Args:
            device: Computing device
            max_samples_per_class: Maximum samples per class
            batch_size: Batch size
            num_epochs: Number of training epochs
            lr_new: Learning rate for new model
            lr_finetune: Learning rate for fine-tuning
        """
        self.device = device
        self.max_samples_per_class = max_samples_per_class
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr_new = lr_new
        self.lr_finetune = lr_finetune

    def train_classifier(self, cropped_data_dir: str, previous_model_path: Optional[str] = None,
                        manual_label_dir: Optional[str] = None, cycle: int = 1) -> Optional[ObjectClassifier]:
        """
        Train classification model

        Args:
            cropped_data_dir: Cropped object image directory
            previous_model_path: Previous model path
            manual_label_dir: Manual labeling data directory
            cycle: Current cycle

        Returns:
            Trained classification model or None
        """
        set_seed()

        class0_dir = os.path.join(cropped_data_dir, "class0")
        class1_dir = os.path.join(cropped_data_dir, "class1")

        # Check directory existence and create
        if not os.path.exists(class0_dir):
            os.makedirs(class0_dir, exist_ok=True)
            print(f"⚠️ class0 directory does not exist, created: {class0_dir}")

        if not os.path.exists(class1_dir):
            os.makedirs(class1_dir, exist_ok=True)
            print(f"⚠️ class1 directory does not exist, created: {class1_dir}")

        # Collect data
        class0_paths = get_image_files(class0_dir)
        class1_paths = get_image_files(class1_dir)

        print(f"Cropped data check: class0={len(class0_paths)}, class1={len(class1_paths)}")

        # Add manual labeling data
        if cycle <= 2 and manual_label_dir:
            manual_class0 = os.path.join(manual_label_dir, "class0")
            manual_class1 = os.path.join(manual_label_dir, "class1")

            if os.path.exists(manual_class0):
                manual_class0_files = get_image_files(manual_class0)
                class0_paths.extend(manual_class0_files)
                print(f"Manual labeling class0 data added: {len(manual_class0_files)} files")

            if os.path.exists(manual_class1):
                manual_class1_files = get_image_files(manual_class1)
                class1_paths.extend(manual_class1_files)
                print(f"Manual labeling class1 data added: {len(manual_class1_files)} files")

        print(f"Final training data: class0={len(class0_paths)}, class1={len(class1_paths)}")

        # Check data shortage
        if len(class0_paths) < 5 or len(class1_paths) < 5:
            print(f"⚠️ Insufficient training data: class0={len(class0_paths)}, class1={len(class1_paths)}")
            print(f"⚠️ At least 5 samples per class are required.")
            return None

        # Sampling and balance adjustment
        class0_paths, class1_paths = self._balance_data(class0_paths, class1_paths)

        # Prepare dataset
        all_paths = class0_paths + class1_paths
        all_labels = [0] * len(class0_paths) + [1] * len(class1_paths)

        combined = list(zip(all_paths, all_labels))
        random.shuffle(combined)
        all_paths, all_labels = zip(*combined)

        # Train/validation split
        split_idx = int(len(all_paths) * 0.8)
        train_paths, val_paths = all_paths[:split_idx], all_paths[split_idx:]
        train_labels, val_labels = all_labels[:split_idx], all_labels[split_idx:]

        print(f"Data split: train={len(train_paths)}, validation={len(val_paths)}")

        # Model initialization
        is_fine_tuning = previous_model_path and os.path.exists(previous_model_path)

        if is_fine_tuning:
            print(f"Fine-tuning from previous model: {previous_model_path}")
            classifier_model = ObjectClassifier(
                model_path=previous_model_path,
                device=self.device
            )
        else:
            print("⚠️ Previous model not found or does not exist.")
            if previous_model_path:
                print(f"⚠️ Model not found: {previous_model_path}")

            # Initial weights needed for first cycle or when no previous model
            raise ValueError(
                f"Initial weights required for classification model training. "
                f"Please provide a pretrained .pth file in classifiers_dir."
            )

        # Execute training
        return self._train_model(classifier_model, train_paths, train_labels, val_paths, val_labels, is_fine_tuning)

    def _balance_data(self, class0_paths: List[str], class1_paths: List[str]) -> Tuple[List[str], List[str]]:
        """Balance data"""
        # Limit maximum samples
        if len(class0_paths) > self.max_samples_per_class:
            class0_paths = random.sample(class0_paths, self.max_samples_per_class)

        if len(class1_paths) > self.max_samples_per_class:
            class1_paths = random.sample(class1_paths, self.max_samples_per_class)

        # Undersampling (when imbalance ratio is 1.5 or more)
        min_samples = min(len(class0_paths), len(class1_paths))
        imbalance_ratio = max(len(class0_paths), len(class1_paths)) / min_samples if min_samples > 0 else 1.0

        print(f"Data imbalance ratio: {imbalance_ratio:.2f}")

        if imbalance_ratio >= 1.5:
            print("Applying data balance adjustment")
            if len(class0_paths) > len(class1_paths):
                class0_paths = random.sample(class0_paths, min_samples)
            else:
                class1_paths = random.sample(class1_paths, min_samples)

        print(f"After balance adjustment: class0={len(class0_paths)}, class1={len(class1_paths)}")
        return class0_paths, class1_paths

    def _train_model(self, classifier_model: ObjectClassifier, train_paths: List[str], train_labels: List[int],
                    val_paths: List[str], val_labels: List[int], is_fine_tuning: bool) -> Optional[ObjectClassifier]:
        """Execute model training"""
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = ClassificationDataset(train_paths, train_labels, transform=data_transform)
        val_dataset = ClassificationDataset(val_paths, val_labels, transform=data_transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=2)

        criterion = nn.CrossEntropyLoss()
        learning_rate = self.lr_finetune if is_fine_tuning else self.lr_new
        optimizer = optim.Adam(classifier_model.model.parameters(), lr=learning_rate)

        print(f"Training settings: {'Fine-tuning' if is_fine_tuning else 'New training'}, lr={learning_rate}")

        best_acc = 0.0
        best_model_state = None

        classifier_model.model.train()

        for epoch in range(self.num_epochs):
            # Training phase
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = classifier_model.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_acc = running_corrects.double() / len(train_dataset)

            # Validation phase
            classifier_model.model.eval()
            val_running_corrects = 0

            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    val_inputs = val_inputs.to(self.device)
                    val_labels = val_labels.to(self.device)

                    val_outputs = classifier_model.model(val_inputs)
                    _, val_preds = torch.max(val_outputs, 1)
                    val_running_corrects += torch.sum(val_preds == val_labels.data)

            val_epoch_acc = val_running_corrects.double() / len(val_dataset)

            if val_epoch_acc > best_acc:
                best_acc = val_epoch_acc
                best_model_state = classifier_model.model.state_dict().copy()

            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}: "
                      f"Train Acc: {epoch_acc:.4f}, Val Acc: {val_epoch_acc:.4f}")

            classifier_model.model.train()

        if best_model_state:
            classifier_model.model.load_state_dict(best_model_state)
            classifier_model.model.eval()
            print(f"Classification model training complete: Best validation accuracy {best_acc:.4f}")
            return classifier_model
        else:
            print("⚠️ Classification model training failed")
            return None
