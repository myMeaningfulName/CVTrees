"""
Test script for interactive GP experiment on CRC dataset.

This script runs the interactive experiment runner where the user manually
controls the population via JSON files instead of using automatic GP evolution.

Usage:
    python test_interactive_crc_experiment.py

On SLURM:
    #SBATCH --cpus-per-task=8
    srun python test_interactive_crc_experiment.py

Workflow:
1. Script loads CRC dataset
2. Creates experiments/interactive_crc/gen_0/population.json
3. User edits the JSON file to add S-expressions
4. User signals 'c' to continue, script evaluates population
5. Results saved, gen_1/population.json created
6. Repeat until user chooses 'q' to quit
"""

import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import urllib.request
import zipfile
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import gp_setup
from src.gp_experiment_interactive import run_interactive_experiment
from src.gp_parallel import get_available_workers


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
    
    if not os.path.exists(extract_path):
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
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Transpose to (Channels, Height, Width) -> (3, H, W)
            img = np.transpose(img, (2, 0, 1))
            
            X_list.append(img)
            y_list.append(label_map[cls_name])

    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Total dataset shape: {X.shape}")
    
    # Split: Hold out Test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    # Split remaining into Train and Val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )
    
    print(f"Data Loaded. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    print("=" * 70)
    print("Interactive CRC GP Experiment")
    print("=" * 70)
    
    # Print worker detection info
    n_workers = get_available_workers()
    print(f"\nDetected {n_workers} available workers")
    
    # Check for SLURM environment
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        print(f"Running in SLURM job: {slurm_job_id}")
        print(f"  SLURM_CPUS_PER_TASK: {os.environ.get('SLURM_CPUS_PER_TASK', 'not set')}")
        print(f"  SLURM_NODELIST: {os.environ.get('SLURM_NODELIST', 'not set')}")
    else:
        print("Not running in SLURM environment")
    
    print()
    
    # 1. Load Data (using a smaller sample for interactive experimentation)
    # You can adjust samples_per_class to control dataset size
    X_train, y_train, X_val, y_val, X_test, y_test = load_crc_data(
        samples_per_class=100  # Smaller sample for faster iteration
    )
    
    if X_train is None:
        print("Failed to load data. Exiting.")
        return

    # 2. Create primitive set
    pset = gp_setup.create_primitive_set()
    
    # 3. Run Interactive Experiment
    print("\n" + "=" * 70)
    print("Starting Interactive Experiment")
    print("=" * 70)
    print("\nIn this mode, YOU control the population.")
    print("After each generation, you'll provide S-expressions via a JSON file.")
    print("The script will wait for your input before proceeding.")
    print("\nPress Ctrl+C at any time to save progress and exit.")
    
    runner = run_interactive_experiment(
        experiment_name="interactive_crc_long",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        pset=pset,
        batch_size=32,
        n_workers=n_workers,
        max_generations=None,  # Unlimited - user controls when to stop
        output_dir="experiments"
    )
    
    print("\n" + "=" * 70)
    print("Interactive Experiment Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {runner.base_dir}")


if __name__ == "__main__":
    # Important for multiprocessing on some platforms
    import multiprocessing
    if sys.platform != 'win32':
        multiprocessing.set_start_method('fork', force=True)
    
    main()
