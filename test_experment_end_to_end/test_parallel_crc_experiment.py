"""
Test script for parallel GP experiment on CRC dataset.

Combines the CRC data loading from test_crc_experiment.py with the
parallel execution capability of test_parallel_experiment.py.

Usage:
    python test_parallel_crc_experiment.py

On SLURM:
    #SBATCH --cpus-per-task=8
    srun python test_parallel_crc_experiment.py
"""

import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import urllib.request
import zipfile
import shutil
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import gp_setup
from src.gp_experiment_parallel import run_parallel_experiment
from src.gp_parallel import get_available_workers
from src.gp_logging import setup_experiment_logging, get_logger, log_memory

# --- Configuration ---
IMG_SIZE = 32

def load_crc_data(samples_per_class=None, test_size=0.2):
    """
    Downloads and loads the Colorectal Cancer Histology Dataset.
    Uses the NCT-CRC-HE-100K-NONORM validation set (CRC-VAL-HE-7K) 
    as a representative sample if the full dataset is not available.
    """
    print("Loading CRC Dataset...")
    
    data_dir = os.path.join(os.getcwd(), 'data', 'crc_data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # We will use the Validation set (~800MB) as our "Sample" of the dataset
    dataset_name = "CRC-VAL-HE-7K"
    zip_filename = f"{dataset_name}.zip"
    zip_filepath = os.path.join(data_dir, zip_filename)
    extract_path = os.path.join(data_dir, dataset_name)
    
    # 1. Download if not exists
    if not os.path.exists(extract_path):
        if not os.path.exists(zip_filepath):
            url = f"https://zenodo.org/record/1214456/files/{dataset_name}.zip?download=1"
            print(f"Downloading {dataset_name} from {url}...")
            print("Note: Using the 7K validation set as a stratified sample of the full 100K dataset.")
            try:
                urllib.request.urlretrieve(url, zip_filepath)
                print("Download complete.")
            except Exception as e:
                print(f"Failed to download: {e}")
                return None, None, None, None, None, None

        print("Extracting...")
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            
    # 2. Load Images
    print("Loading images from disk...")
    # Check if extracted correctly (sometimes zip contains the folder, sometimes not)
    # Based on test_crc_experiment, it seems it extracts to data_dir/CRC-VAL-HE-7K/
    
    if not os.path.exists(extract_path):
         # Try finding it just in case structure differs
         print(f"Expected path {extract_path} not found.")
         return None, None, None, None, None, None

    classes = [d for d in os.listdir(extract_path) if os.path.isdir(os.path.join(extract_path, d))]
    classes.sort()
    
    X_list = []
    y_list = []
    
    label_map = {cls_name: i for i, cls_name in enumerate(classes)}
    print(f"Classes: {label_map}")

    for cls_name in classes:
        cls_dir = os.path.join(extract_path, cls_name)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))]
        
        # Stratified sampling if requested
        if samples_per_class and samples_per_class < len(files):
             files = np.random.choice(files, samples_per_class, replace=False)
        
        print(f"Loading {len(files)} images for class {cls_name}...")
        
        for f in files:
            img_path = os.path.join(cls_dir, f)
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Resize
            #img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Transpose to (Channels, Height, Width) -> (3, 32, 32)
            img = np.transpose(img, (2, 0, 1))
            
            X_list.append(img)
            y_list.append(label_map[cls_name])

    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Total dataset shape: {X.shape}")
    
    # Split
    # Split 1: Hold out Test set
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    
    # Split 2: Split remaining into Train and Val
    # If we want consistent size validation set
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
    
    print(f"Data Loaded. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    print("=" * 60)
    print("Parallel CRC GP Experiment")
    print("=" * 60)
    
    # --- Early logging setup so data-loading issues are captured too ---
    setup_experiment_logging("test_parallel_crc_extended", output_dir="experiments")
    logger = get_logger("main")
    logger.info("Parallel CRC GP Experiment starting")
    log_memory(label="startup", logger=logger)
    
    # Print worker detection info
    n_workers = get_available_workers()
    logger.info("Detected %d available workers", n_workers)
    
    # Check for SLURM environment
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        print(f"Running in SLURM job: {slurm_job_id}")
        print(f"  SLURM_CPUS_PER_TASK: {os.environ.get('SLURM_CPUS_PER_TASK', 'not set')}")
        print(f"  SLURM_NODELIST: {os.environ.get('SLURM_NODELIST', 'not set')}")
    else:
        print("Not running in SLURM environment")
    
    print()
    
    # 1. Load Data
    # Using defaults from original script (load all ~7k samples)
    X_train, y_train, X_val, y_val, X_test, y_test = load_crc_data()
    
    if X_train is None:
        print("Failed to load data. Exiting.")
        return

    # 2. Create primitive set with extended operations and parameters
    print("\n--- Creating Extended Primitive Set ---")
    print("  Extended Operations: ENABLED")
    print("  Extended Parameters: ENABLED")
    pset = gp_setup.create_primitive_set(extended_ops=True, extended_params=True)
    
    # 3. Running Parallel Experiment
    print("\n--- Running Parallel Experiment ---")
    print("  Max Depth: 20")
    
    runner = run_parallel_experiment(
        experiment_name="test_parallel_crc_extended",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        pset=pset,
        pop_size=3,       # Default from test_crc_experiment was 10, increased slightly for better parallel utilization example
        generations=2,    # Default from test_crc_experiment
        crossover_prob=0.5,
        mutation_prob=0.2,
        batch_size=32,
        n_workers=n_workers,  # Or None for auto-detect
        output_dir="experiments",
        max_depth=20      # Extended max depth
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)
    print(f"Best Individual:")
    if runner.best_individual_overall:
        print(f"  Generation: {runner.best_individual_overall['generation']}")
        print(f"  Validation Accuracy: {runner.best_individual_overall['val_acc']:.4f}")
        print(f"  Test Accuracy: {runner.best_individual_overall['test_acc']:.4f}")
        print(f"  Tree Depth: {runner.best_individual_overall['depth']}")
    
    print(f"\nResults saved to: {runner.base_dir}")


if __name__ == "__main__":
    # Important for multiprocessing on some platforms
    import multiprocessing
    # Use 'spawn' on Windows/macOS if needed, 'fork' on Linux usually fine but 'fork' is forced in the example
    if sys.platform != 'win32':
        multiprocessing.set_start_method('fork', force=True)
    
    main()
