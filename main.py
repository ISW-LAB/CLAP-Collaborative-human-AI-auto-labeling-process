"""
main.py - Simplified Active Learning Experiment Runner with Captioning Classifier Support
"""

import os
import time
import traceback
from config import ExperimentConfig
from utils import check_dependencies, get_model_files, Timer
from active_learning import YOLOActiveLearning

class ExperimentRunner:
    """Experiment runner for active learning"""

    def __init__(self, check_deps=True):
        self.timer = Timer()

        # Check dependencies (optional)
        if check_deps:
            try:
                check_dependencies()
                print("✓ Dependencies check completed")
            except ImportError as e:
                print(f"⚠️ Dependency warning: {e}")
                print("⚠️ Some features may be limited. Continuing...")
        else:
            print("⚠️ Dependency check skipped")

    def run_experiment(self, config: ExperimentConfig, skip_cycle_0=False):
        """Run single experiment with captioning classifier support"""
        print(f"\n{'='*80}")
        print("Active Learning Experiment Started")
        print(f"{'='*80}")
        print(config.get_summary())
        print(f"Skip Cycle 0: {skip_cycle_0}")

        # Validate configuration
        try:
            config.validate()
            print("✓ Configuration validation completed")
        except ValueError as e:
            print(f"✗ Configuration error: {e}")
            return False

        # Check model files
        yolo_models = get_model_files(config.models_dir, ".pt")
        if not yolo_models:
            print(f"✗ YOLO models not found: {config.models_dir}")
            return False

        # Check classifier models based on configuration
        classifier_models = []

        if config.use_captioning_classifier:
            # Using captioning classifier - no separate model file needed
            print("✓ Using captioning classifier - pretrained model will be downloaded automatically")
            classifier_models = [None]  # Dummy entry

        elif config.use_classifier:
            # Using existing classifier
            classifier_models = get_model_files(config.classifiers_dir, ".pth")
            if not classifier_models:
                print(f"✗ Classification models not found: {config.classifiers_dir}")
                print(f"✗ Directory check: {config.classifiers_dir}")
                print("✗ Cannot proceed without classification models.")
                return False

            # Verify classification model files exist
            for clf_path in classifier_models:
                if not os.path.exists(clf_path):
                    print(f"✗ Classification model file does not exist: {clf_path}")
                    return False
                else:
                    print(f"✓ Classification model verified: {os.path.basename(clf_path)}")
        else:
            # Not using classifier
            classifier_models = [None]

        print(f"✓ Found {len(yolo_models)} YOLO model(s)")

        if config.use_captioning_classifier:
            print(f"✓ Using captioning classifier (model: {config.captioning_model_type}, keywords: {config.target_keywords})")
        elif config.use_classifier:
            print(f"✓ Found {len(classifier_models)} classification model(s)")

        # Run experiments
        self.timer.start()
        success_count = 0
        total_count = 0

        for classifier_path in classifier_models:
            # Determine classifier name
            if config.use_captioning_classifier:
                classifier_name = f"captioning_{config.captioning_model_type}"
            elif classifier_path is None:
                classifier_name = "no_classifier"
            else:
                classifier_name = os.path.splitext(os.path.basename(classifier_path))[0]

            for model_path in yolo_models:
                model_name = os.path.splitext(os.path.basename(model_path))[0]
                total_count += 1

                cycle_info = "Starting from Cycle 1" if skip_cycle_0 else "Starting from Cycle 0"
                print(f"\n--- Experiment {total_count}: {model_name} + {classifier_name} ({cycle_info}) ---")

                # Print classifier information
                if config.use_captioning_classifier:
                    print(f"Captioning classifier to use: {config.captioning_model_type}")
                    print(f"Target keywords: {config.target_keywords}")
                elif classifier_path:
                    print(f"Classification model to use: {classifier_path}")

                print(f"YOLO model to use: {model_path}")

                try:
                    # Configure individual experiment
                    experiment_config = ExperimentConfig(**config.__dict__)
                    experiment_config.output_dir = os.path.join(
                        config.output_dir,
                        f"{classifier_name}_{model_name}"
                    )

                    # Run Active Learning
                    al = YOLOActiveLearning(experiment_config, model_path, classifier_path)
                    al.run(skip_cycle_0=skip_cycle_0)

                    success_count += 1
                    print(f"✓ Experiment completed: {model_name}")

                except Exception as e:
                    print(f"✗ Experiment failed: {str(e)}")

                    # Save error log
                    error_dir = os.path.join(config.output_dir, f"{classifier_name}_{model_name}", "error_logs")
                    os.makedirs(error_dir, exist_ok=True)

                    with open(os.path.join(error_dir, "error.log"), "w") as f:
                        f.write(f"Error timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Classifier type: {'captioning' if config.use_captioning_classifier else 'traditional'}\n")
                        if config.use_captioning_classifier:
                            f.write(f"Captioning model: {config.captioning_model_type}\n")
                            f.write(f"Target keywords: {config.target_keywords}\n")
                        else:
                            f.write(f"Classification model used: {classifier_path}\n")
                        f.write(f"YOLO model used: {model_path}\n")
                        f.write(f"Skip Cycle 0: {skip_cycle_0}\n")
                        f.write(f"Error message: {str(e)}\n\n")
                        f.write(f"Detailed error:\n{traceback.format_exc()}")

        # Experiment completion
        total_time = self.timer.end()

        print(f"\n{'='*80}")
        print("Experiment Completed!")
        print(f"Success: {success_count}/{total_count}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Results saved to: {config.output_dir}")
        print(f"{'='*80}")

        return success_count > 0

def main():
    """Main experiment execution function"""

    # ==========================================
    # Experiment Parameters (Modify here)
    # ==========================================

    models_dir = "./models/yolo"
    classifiers_dir = "./models/classifiers"
    image_dir = "./data/images"
    label_dir = "./data/labels"  # Can be non-existent
    output_dir = "./results"


    # Basic training parameters
    conf_threshold = 0.25        # Object detection confidence threshold
    iou_threshold = 0.5          # IoU threshold
    class_conf_threshold = 0.5   # Classifier confidence threshold
    max_cycles = 10              # Maximum number of learning cycles
    gpu_num = 0                  # GPU device number to use

    # ===== Classifier Selection Settings =====
    # Option 1: Use traditional classification model
    use_classifier = False                    # Whether to use traditional classifier
    enable_classifier_retraining = False      # Whether to retrain classifier

    # Option 2: Use captioning classifier (VIT-GPT2 supported)
    use_captioning_classifier = True
    captioning_model_type = "vit-gpt2"
    target_keywords = ["car", "vehicle", "truck", "bus", "van"]

    # Note: Only one of use_classifier and use_captioning_classifier should be True
    # =========================================

    # ===== New Option Added =====
    skip_cycle_0 = True                     # Whether to skip Cycle 0
    # True: Start from Cycle 1 (skip baseline measurement, faster experiment)
    # False: Start from Cycle 0 (standard Active Learning, includes baseline)
    # ============================

    # Training detail parameters
    yolo_epochs = 50                # YOLO training epochs
    yolo_batch_size = 16            # YOLO batch size
    yolo_patience = 10              # YOLO early stopping patience

    classifier_epochs = 20          # Classifier training epochs (traditional classifier only)
    classifier_batch_size = 16      # Classifier batch size (traditional classifier only)
    max_samples_per_class = 500     # Max samples per class for classifier training (traditional only)

    # Other settings
    global_seed = 42                # Global random seed

    # ==========================================
    # Configuration Validation
    # ==========================================

    # Validate classifier settings
    if use_classifier and use_captioning_classifier:
        print("✗ Error: Cannot use traditional classifier and captioning classifier simultaneously.")
        print("   Set only one of use_classifier and use_captioning_classifier to True.")
        return

    if use_captioning_classifier:
        # Validate captioning classifier settings
        valid_models = ["blip", "blip2", "instructblip", "vit-gpt2"]
        if captioning_model_type not in valid_models:
            print(f"✗ Error: Captioning model type must be one of {valid_models}.")
            return

        if not target_keywords or len(target_keywords) == 0:
            print("✗ Error: target_keywords required when using captioning classifier.")
            return

        print(f"✓ Captioning classifier configuration:")
        print(f"   - Model: {captioning_model_type}")
        print(f"   - Target keywords: {target_keywords}")
        print(f"   - Retraining: Not supported (using pretrained model)")

        # Print additional model information
        if captioning_model_type == "vit-gpt2":
            print(f"   - VIT-GPT2 model: Vision Transformer + GPT-2 based captioning")
            print(f"   - Required library: transformers (VisionEncoderDecoderModel)")
            print(f"   - Model size: Medium (smaller than BLIP)")
            print(f"   - Caption quality: Good (specialized in natural language generation)")
        elif captioning_model_type == "blip":
            print(f"   - BLIP model: Balanced performance baseline captioning model")
            print(f"   - Required library: transformers")
        elif captioning_model_type in ["blip2", "instructblip"]:
            print(f"   - {captioning_model_type.upper()} model: High-performance captioning model")
            print(f"   - Required library: salesforce-lavis")
            print(f"   - Note: Currently unavailable if LAVIS is not installed")

    # ==========================================
    # Check Label Directory Existence
    # ==========================================

    labels_available = False
    if os.path.exists(label_dir):
        try:
            # Check for label files (.txt)
            label_files = [f for f in os.listdir(label_dir) if f.lower().endswith('.txt')]
            labels_available = len(label_files) > 0
        except:
            labels_available = False

    if not labels_available:
        print(f"\n⚠️ Label directory does not exist or is empty: {label_dir}")
        print("⚠️ Proceeding without performance evaluation (mAP, Precision, Recall).")
        print("⚠️ Detection and training will proceed normally.")
        print("⚠️ Performance metrics will be shown as -1 in result files.")

        response = input("\nDo you want to continue? (y/n): ").lower().strip()
        if response != 'y':
            print("Experiment canceled.")
            return
    else:
        label_count = len([f for f in os.listdir(label_dir) if f.lower().endswith('.txt')])
        print(f"✓ Label directory verified: {label_count} label files")

    # ==========================================
    # Create Experiment Configuration
    # ==========================================

    config = ExperimentConfig(
        # Path settings
        models_dir=models_dir,
        classifiers_dir=classifiers_dir,
        image_dir=image_dir,
        label_dir=label_dir,
        output_dir=output_dir,

        # Label availability
        labels_available=labels_available,

        # Hardware settings
        gpu_num=gpu_num,

        # Training parameters
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        class_conf_threshold=class_conf_threshold,
        max_cycles=max_cycles,
        max_samples_per_class=max_samples_per_class,

        # Feature activation settings
        use_classifier=use_classifier,
        enable_classifier_retraining=enable_classifier_retraining,
        use_captioning_classifier=use_captioning_classifier,
        captioning_model_type=captioning_model_type,
        target_keywords=target_keywords,

        # Training settings
        yolo_epochs=yolo_epochs,
        yolo_batch_size=yolo_batch_size,
        yolo_patience=yolo_patience,
        classifier_epochs=classifier_epochs,
        classifier_batch_size=classifier_batch_size,

        # Seed settings
        global_seed=global_seed
    )

    # ==========================================
    # Run Experiment
    # ==========================================

    print("YOLO Active Learning with Classification System")
    print("="*60)
    print(f"Skip Cycle 0: {skip_cycle_0}")

    # Print classifier information
    if use_captioning_classifier:
        print(f"Captioning classifier mode: Using {captioning_model_type} model")
        print(f"Target keywords: {target_keywords}")
        print("No retraining: Using pretrained model as-is")
    elif use_classifier:
        print(f"Traditional classifier mode: Retraining {'enabled' if enable_classifier_retraining else 'disabled'}")
    else:
        print("No classifier: Using YOLO detection results only")

    if skip_cycle_0:
        print("Fast experiment mode: Starting training immediately from Cycle 1")
        print("⚠️ Baseline performance measurement will be skipped.")
    else:
        print("Standard experiment mode: Including Cycle 0 baseline measurement")
        print("✓ Consistent performance comparison baseline will be provided.")

    try:
        runner = ExperimentRunner(check_deps=True)  # Enable dependency check
        success = runner.run_experiment(config, skip_cycle_0=skip_cycle_0)

        if success:
            print("\nExperiment completed successfully!")

            if use_captioning_classifier:
                print(f"Experiment using captioning classifier ({captioning_model_type}) completed.")
            elif use_classifier:
                print(f"Experiment using traditional classifier completed.")

            if skip_cycle_0:
                print("Time saved with fast experiment mode.")
            else:
                print("Complete experiment with baseline included finished.")
        else:
            print("\nExperiment failed.")

    except KeyboardInterrupt:
        print("\n\n⚠️ Experiment interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error occurred: {str(e)}")
        print("\nDetailed error information:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
