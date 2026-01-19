"""
captioning_classifier.py - Image Captioning-based Object Classification Module (VIT-GPT2 Support Added)
"""

import os
import cv2
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import re

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False
    print("⚠️ transformers is not installed. Please run: pip install transformers")

try:
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    VIT_GPT2_AVAILABLE = True
except ImportError:
    VIT_GPT2_AVAILABLE = False
    print("⚠️ transformers for VIT-GPT2 is not installed. Please run: pip install transformers")

try:
    from lavis.models import load_model_and_preprocess
    LAVIS_AVAILABLE = True
except ImportError:
    LAVIS_AVAILABLE = False
    print("⚠️ LAVIS is not installed. Please run: pip install salesforce-lavis")

class ImageCaptioningClassifier:
    """Image captioning-based object classifier (VIT-GPT2 support added)"""

    def __init__(self, target_keywords: List[str], model_type: str = "blip",
                 device: Optional[torch.device] = None, conf_threshold: float = 0.5,
                 gpu_num: int = 0):
        """
        Initialize image captioning classifier

        Args:
            target_keywords: Keyword list for positive object classification (e.g., ['car', 'vehicle', 'automobile'])
            model_type: Captioning model type ('blip', 'blip2', 'instructblip', 'vit-gpt2')
            device: Computing device
            conf_threshold: Classification confidence threshold (0 or 1 for keyword matching)
            gpu_num: GPU number
        """
        if device is None:
            self.device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.target_keywords = [keyword.lower().strip() for keyword in target_keywords]
        self.model_type = model_type.lower()
        self.conf_threshold = conf_threshold
        self.gpu_num = gpu_num

        # Statistics tracking
        self.stats = {
            'total_classifications': 0,
            'positive_classifications': 0,
            'negative_classifications': 0,
            'keyword_matches': {},
            'generated_captions': []
        }

        # Model initialization
        self._initialize_model()

        print(f"Image captioning classifier initialization complete:")
        print(f"  - Model: {self.model_type}")
        print(f"  - Target keywords: {self.target_keywords}")
        print(f"  - Device: {self.device}")

    def _initialize_model(self):
        """Initialize captioning model (VIT-GPT2 support added)"""
        try:
            if self.model_type == "blip":
                if not BLIP_AVAILABLE:
                    raise ImportError("transformers library is required")

                print("Loading BLIP model...")
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self.model.to(self.device)
                self.model.eval()
                print("✓ BLIP model loaded successfully")

            elif self.model_type == "vit-gpt2":
                if not VIT_GPT2_AVAILABLE:
                    raise ImportError("transformers library for VIT-GPT2 is required")

                print("Loading VIT-GPT2 model...")
                model_name = "nlpconnect/vit-gpt2-image-captioning"
                self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
                self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)

                self.model.to(self.device)
                self.model.eval()
                print("✓ VIT-GPT2 model loaded successfully")

            elif self.model_type in ["blip2", "instructblip"]:
                if not LAVIS_AVAILABLE:
                    raise ImportError("LAVIS library is required")

                model_name = "blip2_opt" if self.model_type == "blip2" else "blip2_vicuna_instruct"
                print(f"Loading {self.model_type.upper()} model...")
                self.model, self.vis_processors, _ = load_model_and_preprocess(
                    name=model_name,
                    model_type="pretrain_opt2.7b" if self.model_type == "blip2" else "vicuna7b",
                    is_eval=True,
                    device=self.device
                )
                print(f"✓ {self.model_type.upper()} model loaded successfully")

            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        except Exception as e:
            print(f"✗ Model initialization failed: {e}")
            print("⚠️ Falling back to dummy classifier.")
            self.model = None
            self.processor = None
            self.feature_extractor = None
            self.tokenizer = None

    def generate_caption(self, image: np.ndarray) -> str:
        """Generate caption from image (VIT-GPT2 support added)"""
        if self.model is None:
            return "dummy caption with car"  # Dummy caption for testing

        try:
            # Convert OpenCV image to PIL image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)

            # Check image size and resize
            if pil_image.size[0] < 10 or pil_image.size[1] < 10:
                return ""

            # Generate caption based on model type
            if self.model_type == "blip":
                return self._generate_blip_caption(pil_image)
            elif self.model_type == "vit-gpt2":
                return self._generate_vit_gpt2_caption(pil_image)
            elif self.model_type in ["blip2", "instructblip"]:
                return self._generate_lavis_caption(pil_image)
            else:
                return ""

        except Exception as e:
            print(f"⚠️ Error during caption generation: {e}")
            return ""

    def _generate_blip_caption(self, pil_image: Image.Image) -> str:
        """Generate caption using BLIP model"""
        try:
            inputs = self.processor(pil_image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=50, num_beams=5)

            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption.lower().strip()

        except Exception as e:
            print(f"⚠️ BLIP caption generation error: {e}")
            return ""

    def _generate_vit_gpt2_caption(self, pil_image: Image.Image) -> str:
        """Generate caption using VIT-GPT2 model"""
        try:
            # Image preprocessing
            pixel_values = self.feature_extractor(images=pil_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            # Caption generation
            with torch.no_grad():
                output_ids = self.model.generate(
                    pixel_values,
                    max_length=50,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    length_penalty=2.0
                )

            # Decoding
            caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return caption.lower().strip()

        except Exception as e:
            print(f"⚠️ VIT-GPT2 caption generation error: {e}")
            return ""

    def _generate_lavis_caption(self, pil_image: Image.Image) -> str:
        """Generate caption using LAVIS model"""
        try:
            image_tensor = self.vis_processors["eval"](pil_image).unsqueeze(0).to(self.device)

            if self.model_type == "instructblip":
                # InstructBLIP uses prompts
                prompt = "Describe this image in detail."
                caption = self.model.generate({"image": image_tensor, "prompt": prompt})[0]
            else:
                # BLIP2 uses image only
                caption = self.model.generate({"image": image_tensor})[0]

            return caption.lower().strip()

        except Exception as e:
            print(f"⚠️ LAVIS caption generation error: {e}")
            return ""

    def classify(self, image: np.ndarray) -> Tuple[int, float]:
        """
        Perform image classification

        Args:
            image: OpenCV format image (BGR)

        Returns:
            (predicted class, confidence) tuple
            - Class: 0 (positive, contains keyword), 1 (negative, no keyword)
            - Confidence: 1.0 or 0.0 for keyword matching
        """
        self.stats['total_classifications'] += 1

        if image.shape[0] < 10 or image.shape[1] < 10:
            self.stats['negative_classifications'] += 1
            return 1, 0.0

        try:
            # Generate caption
            caption = self.generate_caption(image)

            if not caption:
                self.stats['negative_classifications'] += 1
                return 1, 0.0

            # Save caption to statistics (keep only last 100)
            self.stats['generated_captions'].append(caption)
            if len(self.stats['generated_captions']) > 100:
                self.stats['generated_captions'] = self.stats['generated_captions'][-100:]

            # Check keyword matching
            is_positive, matched_keywords = self._check_keywords(caption)

            # Update matched keyword statistics
            for keyword in matched_keywords:
                if keyword not in self.stats['keyword_matches']:
                    self.stats['keyword_matches'][keyword] = 0
                self.stats['keyword_matches'][keyword] += 1

            if is_positive:
                self.stats['positive_classifications'] += 1
                return 0, 1.0  # Positive object
            else:
                self.stats['negative_classifications'] += 1
                return 1, 0.0  # Negative object

        except Exception as e:
            print(f"⚠️ Error during image classification: {e}")
            self.stats['negative_classifications'] += 1
            return 1, 0.0

    def _check_keywords(self, caption: str) -> Tuple[bool, List[str]]:
        """
        Check target keywords in caption

        Args:
            caption: Generated caption

        Returns:
            (keyword presence, matched keyword list)
        """
        caption_lower = caption.lower()
        matched_keywords = []

        for keyword in self.target_keywords:
            # Exact matching considering word boundaries
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, caption_lower):
                matched_keywords.append(keyword)

        return len(matched_keywords) > 0, matched_keywords

    def update_keywords(self, new_keywords: List[str]):
        """Update target keywords"""
        self.target_keywords = [keyword.lower().strip() for keyword in new_keywords]
        print(f"Keywords updated: {self.target_keywords}")

    def get_stats(self) -> Dict[str, Any]:
        """Return classification statistics"""
        stats = self.stats.copy()

        # Calculate ratios
        if stats['total_classifications'] > 0:
            stats['positive_ratio'] = stats['positive_classifications'] / stats['total_classifications']
            stats['negative_ratio'] = stats['negative_classifications'] / stats['total_classifications']
        else:
            stats['positive_ratio'] = 0.0
            stats['negative_ratio'] = 0.0

        return stats

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_classifications': 0,
            'positive_classifications': 0,
            'negative_classifications': 0,
            'keyword_matches': {},
            'generated_captions': []
        }

    def get_recent_captions(self, num_captions: int = 10) -> List[str]:
        """Return recently generated captions"""
        return self.stats['generated_captions'][-num_captions:]

    def export_classification_log(self, output_path: str):
        """Export classification log"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("Image Captioning Classification Results Log\n")
                f.write("="*50 + "\n\n")

                f.write(f"Model Information:\n")
                f.write(f"  - Model Type: {self.model_type}\n")
                f.write(f"  - Target Keywords: {', '.join(self.target_keywords)}\n")
                f.write(f"  - Device: {self.device}\n\n")

                stats = self.get_stats()
                f.write(f"Classification Statistics:\n")
                f.write(f"  - Total Classifications: {stats['total_classifications']}\n")
                f.write(f"  - Positive Classifications: {stats['positive_classifications']} ({stats['positive_ratio']:.2%})\n")
                f.write(f"  - Negative Classifications: {stats['negative_classifications']} ({stats['negative_ratio']:.2%})\n\n")

                if stats['keyword_matches']:
                    f.write(f"Keyword Matching Statistics:\n")
                    for keyword, count in sorted(stats['keyword_matches'].items(), key=lambda x: x[1], reverse=True):
                        f.write(f"  - '{keyword}': {count} times\n")
                    f.write("\n")

                if stats['generated_captions']:
                    f.write(f"Recently Generated Captions (max 20):\n")
                    f.write("-" * 30 + "\n")
                    recent_captions = stats['generated_captions'][-20:]
                    for i, caption in enumerate(recent_captions, 1):
                        f.write(f"{i:2d}. {caption}\n")

            print(f"Classification log saved: {output_path}")

        except Exception as e:
            print(f"⚠️ Failed to save classification log: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Return model information"""
        return {
            'model_type': self.model_type,
            'target_keywords': self.target_keywords,
            'device': str(self.device),
            'conf_threshold': self.conf_threshold,
            'model_available': self.model is not None,
            'classification_method': 'keyword_matching'
        }

    def save_model(self, save_path: str):
        """Save model configuration (captioning models are pretrained, so only save configuration)"""
        import json

        config = {
            'model_type': self.model_type,
            'target_keywords': self.target_keywords,
            'conf_threshold': self.conf_threshold,
            'gpu_num': self.gpu_num
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save as JSON configuration file
        config_path = save_path.replace('.pth', '_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        print(f"Captioning classifier configuration saved: {config_path}")

class CaptioningClassifierTrainer:
    """Captioning classifier trainer (for configuration management)"""

    def __init__(self, device: torch.device, **kwargs):
        """
        Initialize captioning classifier trainer
        (No actual training, only configuration management)
        """
        self.device = device
        print("Captioning classifier uses pretrained models, no separate training required.")

    def train_classifier(self, cropped_data_dir: str, previous_model_path: Optional[str] = None,
                        manual_label_dir: Optional[str] = None, cycle: int = 1,
                        target_keywords: List[str] = None, model_type: str = "blip") -> Optional[ImageCaptioningClassifier]:
        """
        "Train" captioning classifier (actually creates new classifier with configuration)

        Args:
            cropped_data_dir: Cropped object image directory (for statistics collection)
            previous_model_path: Previous model path (for configuration reference)
            manual_label_dir: Manual labeling data directory (unused)
            cycle: Current cycle
            target_keywords: Target keyword list
            model_type: Captioning model type

        Returns:
            Newly created captioning classifier
        """

        if target_keywords is None:
            target_keywords = ['car', 'vehicle', 'automobile']

        print(f"Creating captioning classifier for Cycle {cycle}:")
        print(f"  - Model Type: {model_type}")
        print(f"  - Target Keywords: {target_keywords}")

        # Data directory statistics (optional)
        if os.path.exists(cropped_data_dir):
            class0_dir = os.path.join(cropped_data_dir, "class0")
            class1_dir = os.path.join(cropped_data_dir, "class1")

            class0_count = len([f for f in os.listdir(class0_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]) if os.path.exists(class0_dir) else 0
            class1_count = len([f for f in os.listdir(class1_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]) if os.path.exists(class1_dir) else 0

            print(f"  - Collected Data: class0={class0_count}, class1={class1_count}")

        try:
            # Create new captioning classifier
            new_classifier = ImageCaptioningClassifier(
                target_keywords=target_keywords,
                model_type=model_type,
                device=self.device
            )

            print(f"✓ Captioning classifier for Cycle {cycle} created successfully")
            return new_classifier

        except Exception as e:
            print(f"✗ Captioning classifier creation failed: {e}")
            return None

def load_captioning_classifier_from_config(config_path: str, device: Optional[torch.device] = None) -> Optional[ImageCaptioningClassifier]:
    """Load captioning classifier from configuration file"""
    import json

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        classifier = ImageCaptioningClassifier(
            target_keywords=config['target_keywords'],
            model_type=config['model_type'],
            device=device,
            conf_threshold=config.get('conf_threshold', 0.5),
            gpu_num=config.get('gpu_num', 0)
        )

        print(f"Captioning classifier configuration loaded: {config_path}")
        return classifier

    except Exception as e:
        print(f"⚠️ Failed to load captioning classifier configuration: {e}")
        return None
