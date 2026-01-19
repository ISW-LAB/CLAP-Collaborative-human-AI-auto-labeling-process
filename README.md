# CLAP: Vision-Language Model Guided Auto-Labeling for Object Detection

**Official PyTorch Implementation**

A novel human-in-the-loop framework integrating YOLO-based object detection with Vision-Language Models (specifically ViT-GPT2) to minimize annotation costs and facilitate robust cross-domain adaptation through semantic filtering.

---

## 🎯 Motivation: The Domain Shift Challenge

Standard object detection models pre-trained on general large-scale datasets (Source Domain, e.g., COCO) often suffer from significant performance degradation when applied to specific target domains (e.g., autonomous driving views in BDD100k or VisDrone). This is due to **domain shift**—differences in image style, lighting, viewpoint, and object appearance.

As illustrated below, directly transferring a model trained on diverse COCO images to specific driving scenarios presents challenges. Our system aims to adapt the model to the target domain iteratively with minimal human intervention.

<img width="1612" height="1125" alt="Motivation (1)" src="https://github.com/user-attachments/assets/7d7cf296-4ff6-42b0-b0f5-76babb09be46" />
*Figure 1: Illustration of domain shift. The visual gap between the Source Domain (COCO) and Target Domain (BDD100k) leads to poor generalization in standard transfer learning.*

---

## 🛠️ Methodology: Semantic Filtering with VLMs

To address domain shift and improve detection performance without extensive manual re-labeling, we propose an iterative **Auto-Labeling Framework**.

Unlike traditional pseudo-labeling that relies solely on confidence scores (which are often unreliable in new domains), our system employs a **Vision-Language Model (VLM)** acting as a semantic filter. The pipeline operates as follows:

1.  **Proposal Generation:** The detection model (YOLO) proposes object bounding boxes.
2.  **Caption Generation:** Each cropped proposal is fed into a VLM (ViT-GPT2) to generate a natural language description.
3.  **Semantic Validation:** The generated captions are cross-referenced against user-defined **Target Keywords**. Only objects whose descriptions match the semantic intent are retained as pseudo-labels.
4.  **Iterative Retraining:** The filtered pseudo-labels are used to retrain the detector, progressively improving its performance.

<img width="3543" height="1840" alt="Overview_edit (1)" src="https://github.com/user-attachments/assets/d295ddf8-e6b1-4518-a28f-d7b7e110e35f" />
*Figure 2: The proposed CLAP framework. (a) The detector generates proposals on unlabeled data. (b) A VLM (ViT-GPT2) generates captions for each proposal. (c) A keyword matching mechanism filters out false positives (red) and retains true positives (green) for retraining.*

---

## 🌟 Key Features

- **VLM-Driven Semantic Filtering**: Utilizes **ViT-GPT2** to semantically validate pseudo-labels, effectively removing false positives caused by domain shift.
- **Human-in-the-Loop Optimization**: Reduces human intervention to a simple set of **target keywords**, eliminating the need for box-level manual annotation.
- **Iterative Self-Training**: Fully automated loop of *Detection → Filtering → Retraining* to progressively adapt to the target domain.
- **Robust Performance**: Achieves superior F1-scores compared to standard transfer learning and confidence-based pseudo-labeling.
- **Detailed Analytics**: Provides comprehensive logs on timing, filtering ratios, and per-cycle performance metrics.

---

## 📋 Requirements

### System Prerequisites
- Python 3.8+
- CUDA-capable GPU (Minimum 8GB VRAM recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/ISW-LAB/CLAP-Collaborative-human-AI-auto-labeling-process.git
cd clap-autolabeling

# Install dependencies
pip install -r requirements.txt

# Install Transformers for VLM support
pip install transformers>=4.30.0
```

---

## 🗂️ Data Preparation

To run the CLAP framework, organize your dataset and models as follows:

### Directory Structure
```
project_root/
├── models/
│   └── yolo/
│       └── yolov8n.pt           # Pre-trained YOLO weights (or your custom model)
├── data/
│   ├── images/
│   │   └── *.jpg                # Unlabeled target domain images
│   └── labels/
│       └── *.txt                # (Optional) Ground truth labels for evaluation
└── results/                     # Output directory (automatically created)
```

### Label Format
The system uses the standard YOLO label format (`.txt` files):
```
<class_id> <center_x> <center_y> <width> <height>
0 0.5 0.5 0.3 0.4
```

---

## 🚀 Quick Start

### 1. Basic Experiment Setup

Edit the `main.py` file to configure your experiment parameters and **Target Keywords**.

```python
from config import ExperimentConfig
from active_learning import YOLOActiveLearning

# Define semantic keywords for your target class (e.g., Vehicle)
# These keywords act as the semantic filter for the VLM.
target_keywords = ["car", "vehicle", "truck", "bus", "van"]

config = ExperimentConfig(
    # Path settings
    models_dir="./models/yolo",
    image_dir="./data/images",
    output_dir="./results",

    # VLM (Semantic Filter) Settings
    use_captioning_classifier=True,
    captioning_model_type="vit-gpt2",  # Recommended for efficiency
    target_keywords=target_keywords,

    # Active Learning Cycle Settings
    conf_threshold=0.25,     # Initial detection confidence
    max_cycles=10,           # Number of self-training iterations
    gpu_num=0
)

# Initialize and Run
al = YOLOActiveLearning(config=config, model_path="./models/yolo/yolov8n.pt")
al.run()
```

### 2. Running the Experiment

Execute the main script to start the iterative auto-labeling process:
```bash
python main.py
```

---

## ⚙️ Configuration Parameters

Key parameters in `config.py` that control the CLAP framework:

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `use_captioning_classifier` | Enable VLM-based semantic filtering (Core of CLAP). | `True` |
| `target_keywords` | List of words the VLM looks for to validate a detection. | `['car']` |
| `conf_threshold` | Confidence threshold for the YOLO detector. | `0.25` |
| `max_cycles` | Total number of retraining iterations. | `10` |
| `yolo_epochs` | Number of epochs for retraining YOLO in each cycle. | `50` |
| `yolo_batch_size` | Batch size for YOLO training. | `16` |

---

## 📊 Results and Analysis

### Output Structure
After running the experiment, the `results/` directory will contain:
```
results/experiment_name/
├── cycle_0/
│   ├── detections/      # Visualized detection results
│   ├── labels/          # Generated pseudo-labels
│   └── logs/            # Cycle-specific logs
├── cycle_X/
│   ├── training/        # YOLO retraining weights (best.pt)
│   └── ...
├── performance_metrics.csv  # CSV tracking Precision, Recall, F1 per cycle
└── comparison_plot.png      # Visualization of performance trends
```

### Performance Metrics
The system automatically tracks metrics across cycles. As shown in the paper, CLAP significantly outperforms baselines:

| Method | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Baseline (Transfer)** | 0.780 | 0.388 | 0.519 |
| **Pseudo-Labeling** | 0.735 | 0.558 | 0.634 |
| **Proposed (CLAP)** | **0.835** | **0.525** | **0.872** |

*(Results based on Cycle 10 VisDrone adaptation)*

### Visual Comparison

The effectiveness of the semantic filtering approach is demonstrated visually across training cycles. Compared to baseline methods (Pseudo-Labeling) and supervised filters (DenseNet), our method significantly reduces false positives (red boxes) and progressively improves recall for true targets (green boxes).

<img width="1591" height="2126" alt="detection_result_by_cycle_135" src="https://github.com/user-attachments/assets/3a58fb14-eeea-4594-8d42-f3fed1cea9e2" />
*Figure 3: Qualitative comparison. (a) Pseudo-labeling accumulates errors over cycles. (c) CLAP maintains high precision by filtering non-target objects using VLM captions.*

---

## 🔧 Advanced Usage

### Skipping the Baseline (Cycle 0)
If you want to skip the initial inference cycle and start training immediately:
```python
al.run(skip_cycle_0=True)
```

### Customizing the VLM
While `vit-gpt2` is the default, the modular design supports other Hugging Face models. You can extend the `CaptioningClassifier` class in `classifier.py` to support models like Git or BLIP if higher capacity is needed (though ViT-GPT2 is optimized for speed/accuracy trade-offs in this framework).

---

## 🐛 Troubleshooting

**1. CUDA/GPU Memory Errors**
* **Solution:** Reduce the batch size in `main.py`.
    ```python
    yolo_batch_size = 8  # Decrease from 16
    ```

**2. Low Detection Rate in Cycle 0**
* **Solution:** Lower the confidence threshold if the domain shift is severe.
    ```python
    conf_threshold = 0.15  # Lower from 0.25
    ```

**3. VLM Not Filtering Correctly**
* **Solution:** Check your `target_keywords`. Ensure they cover synonyms (e.g., use `["cab", "taxi", "car"]` instead of just `["taxi"]`).

---

## 📧 Contact

For questions about the paper or code implementation, please contact: gc.jo-isw@cbnu.ac.kr
