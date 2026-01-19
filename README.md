# YOLO Active Learning with Classification System

A YOLO-based Active Learning system that supports both traditional classification models and image captioning-based classifiers (BLIP, BLIP2, InstructBLIP, VIT-GPT2) for iterative object detection improvement.

## 🌟 Key Features

- **YOLO-based Active Learning**: Iterative training for performance improvement
- **Dual Classification Methods**:
  - Traditional CNN-based classifiers (DenseNet121)
  - Image captioning classifiers (BLIP, BLIP2, InstructBLIP, VIT-GPT2)
- **Modular Design**: Each component can be used independently
- **Automated Experiments**: Fully automated execution without user intervention
- **Performance Tracking**: Comprehensive metrics and visualization
- **Cycle Timing**: Detailed timing information for each training cycle

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- Minimum 8GB RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/yolo-active-learning.git
cd yolo-active-learning

# Install dependencies
pip install -r requirements.txt

# Optional: Install transformers for captioning classifiers
pip install transformers>=4.30.0
```

## 🗂️ Project Structure

```
project/
├── config.py                    # Experiment configuration management
├── utils.py                     # Common utility functions
├── classifier.py                # Traditional classification model
├── captioning_classifier.py     # Image captioning-based classifier
├── detector.py                  # Object detection module
├── evaluator.py                 # Performance evaluation module
├── active_learning.py           # Main Active Learning class
├── main.py                      # Experiment execution script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 📁 Data Preparation

### Directory Structure
```
your_project/
├── models/
│   ├── yolo/
│   │   └── *.pt                 # YOLO model files
│   └── classifiers/
│       └── *.pth                # Classification model files (optional)
├── data/
│   ├── images/
│   │   └── *.jpg                # Training images
│   └── labels/
│       └── *.txt                # YOLO format labels (optional)
└── results/                     # Output directory (auto-created)
```

### Label Format (YOLO)
```
class_id center_x center_y width height
0 0.5 0.5 0.3 0.4
```

## 🚀 Quick Start

### 1. Basic Experiment

Edit `main.py` to configure your experiment:

```python
# Set your data paths
models_dir = "./models/yolo"
classifiers_dir = "./models/classifiers"
image_dir = "./data/images"
label_dir = "./data/labels"
output_dir = "./results"

# Configure basic parameters
conf_threshold = 0.25
max_cycles = 10
gpu_num = 0

# Choose classifier type
use_classifier = False  # Traditional classifier
use_captioning_classifier = True  # Captioning classifier
```

Run the experiment:
```bash
python main.py
```

### 2. Using Captioning Classifier

Configure the captioning classifier in `main.py`:

```python
# Captioning classifier settings
use_captioning_classifier = True
captioning_model_type = "vit-gpt2"  # Options: "blip", "blip2", "instructblip", "vit-gpt2"
target_keywords = ["car", "vehicle", "truck", "bus", "van"]
```

Supported captioning models:
- **BLIP**: Balanced performance, general-purpose
- **BLIP2**: High performance, larger model
- **InstructBLIP**: Instruction-based captioning
- **VIT-GPT2**: Vision Transformer + GPT-2, good natural language generation

### 3. Using Traditional Classifier

```python
# Traditional classifier settings
use_classifier = True
enable_classifier_retraining = False  # Set to True for retraining each cycle

# Classifier training parameters
classifier_epochs = 20
classifier_batch_size = 16
max_samples_per_class = 500
```

## ⚙️ Configuration

### Main Parameters

```python
from config import ExperimentConfig

config = ExperimentConfig(
    # Path settings
    models_dir="./models/yolo",
    classifiers_dir="./models/classifiers",
    image_dir="./data/images",
    label_dir="./data/labels",
    output_dir="./results",

    # Hardware settings
    gpu_num=0,

    # Detection parameters
    conf_threshold=0.25,
    iou_threshold=0.5,
    class_conf_threshold=0.5,
    max_cycles=10,

    # Classifier settings
    use_classifier=False,
    enable_classifier_retraining=False,
    use_captioning_classifier=True,
    captioning_model_type="vit-gpt2",
    target_keywords=['car', 'vehicle'],

    # Training settings
    yolo_epochs=50,
    yolo_batch_size=16,
    yolo_patience=10,

    # Seed for reproducibility
    global_seed=42
)
```

## 📊 Results and Analysis

### Output Structure
```
results/
├── model_name/
│   ├── cycle_0/
│   │   ├── detections/              # Detection result images
│   │   ├── labels/                  # Generated labels
│   │   └── cycle_timing.json        # Cycle timing information
│   ├── cycle_1/
│   │   ├── training/                # YOLO training results
│   │   ├── classification_training/ # Classifier training results
│   │   └── cropped_objects/         # Cropped object images
│   ├── performance_metrics.csv      # Performance metrics
│   ├── performance_summary.txt      # Performance summary
│   ├── cycle_timing_summary.json    # Overall timing summary
│   └── cycle_timing_summary.txt     # Human-readable timing
```

### Performance Metrics
- **mAP50**: Mean Average Precision @ IoU 0.5
- **Precision**: Detection precision
- **Recall**: Detection recall
- **F1-Score**: F1 score
- **Detected_Objects**: Number of detected objects
- **Filtered_Objects**: Number of filtered objects by classifier

### Timing Information
Each cycle's timing is recorded in JSON format:
```json
{
  "cycle": 1,
  "total_duration_minutes": 15.5,
  "step_times": {
    "detection": 120.5,
    "classification": 45.2,
    "training": 800.3
  }
}
```

## 🔧 Advanced Usage

### 1. Skip Cycle 0 (Baseline)

To skip the baseline measurement and start directly from Cycle 1:

```python
skip_cycle_0 = True  # Set in main.py
```

### 2. Custom Classifier

```python
from classifier import ObjectClassifier

# Load pretrained classifier
classifier = ObjectClassifier("path/to/model.pth")

# Classify object
pred_class, confidence = classifier.classify(cropped_image)
```

### 3. Programmatic Usage

```python
from active_learning import YOLOActiveLearning
from config import ExperimentConfig

# Configure experiment
config = ExperimentConfig(
    models_dir="./models/yolo",
    image_dir="./data/images",
    use_captioning_classifier=True,
    captioning_model_type="vit-gpt2",
    target_keywords=["car", "vehicle"]
)

# Run active learning
al = YOLOActiveLearning(
    config=config,
    model_path="./models/yolo/yolov8n.pt",
    classifier_path=None  # Not needed for captioning classifier
)

al.run(skip_cycle_0=False)
```

## 🐛 Troubleshooting

### Common Issues

#### 1. GPU Memory Error
```python
# Reduce batch size in main.py
yolo_batch_size = 8
classifier_batch_size = 8
```

#### 2. Classifier Loading Failure
- Ensure model structure matches the weights
- The system automatically attempts structure adjustment

#### 3. No Detections
- Lower the `conf_threshold` (e.g., 0.1)
- Verify the YOLO model is suitable for your dataset

#### 4. Transformers Not Found
```bash
# Install transformers for captioning classifiers
pip install transformers>=4.30.0
```

### Checking Logs
```bash
# View error logs
cat results/model_name/error_logs/error.log

# View experiment logs
cat results/model_name/logs/experiment_log_*.txt
```

## 📈 Experiment Design Examples

### 1. Comparing Classification Methods

Run experiments with different classifiers:
```python
# Experiment 1: No classifier (baseline)
use_classifier = False
use_captioning_classifier = False

# Experiment 2: Traditional classifier
use_classifier = True
use_captioning_classifier = False

# Experiment 3: Captioning classifier
use_classifier = False
use_captioning_classifier = True
```

### 2. Testing Different Captioning Models

```python
# Test each captioning model
models = ["blip", "vit-gpt2", "blip2", "instructblip"]
for model_type in models:
    captioning_model_type = model_type
    # Run experiment
```

### 3. Keyword Sensitivity Analysis

```python
# Test different keyword sets
keyword_sets = [
    ["car"],
    ["car", "vehicle"],
    ["car", "vehicle", "truck", "bus", "van"]
]
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{yolo-active-learning,
  author = {Your Name},
  title = {YOLO Active Learning with Classification System},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/yolo-active-learning}
}
```

## 📄 License

This project is created for research purposes. Please specify your license.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 🙏 Acknowledgments

- YOLO: [Ultralytics](https://github.com/ultralytics/ultralytics)
- BLIP: [Salesforce](https://github.com/salesforce/BLIP)
- VIT-GPT2: [NLP Connect](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)

## 📚 Additional Resources

- [YOLO Documentation](https://docs.ultralytics.com/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)
