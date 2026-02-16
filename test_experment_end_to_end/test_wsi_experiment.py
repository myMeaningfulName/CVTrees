import os
import sys
import glob
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src import gp_experiment, gp_setup

# --- Configuration ---
EXP_NAME = "wsi_patch_experiment"
PATCH_DIR = "/Users/ronfranco/extracted_patchs"
CSV_PATH = "TCGA.pam50_calls.csv"
POPULATION_SIZE = 10
GENERATIONS = 10
BATCH_SIZE = 32
SAMPLE_SIZE_PER_CLASS = 50 # Default, can be changed

def load_wsi_data(patch_dir, csv_path, sample_size_per_class=None, resize=None):
    print(f"Loading WSI patches from {patch_dir}...")
    print(f"Loading labels from {csv_path}...")
    
    # 1. Load Labels
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return [], [], None

    df = pd.read_csv(csv_path)
    # Create a map: Pt.ID -> Call
    df = df.dropna(subset=['Pt.ID', 'Call'])
    
    # Normalize Pt.ID in CSV just in case (though they look dot-separated)
    id_to_label = dict(zip(df['Pt.ID'], df['Call']))
    
    # 2. Scan Images
    if not os.path.exists(patch_dir):
        print(f"Error: Patch directory not found at {patch_dir}")
        return [], [], None
        
    image_paths = glob.glob(os.path.join(patch_dir, "*.png"))
    print(f"Found {len(image_paths)} images.")
    
    data = []
    labels = []
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        # Example: TCGA-AC-A5EH-01A-01-TS1...
        # Extract first 3 parts: TCGA-AC-A5EH
        parts = filename.split('-')
        if len(parts) < 3:
            continue
            
        patient_id_hyphen = "-".join(parts[:3]) # TCGA-AC-A5EH
        patient_id_dot = patient_id_hyphen.replace('-', '.') # TCGA.AC.A5EH
        
        if patient_id_dot in id_to_label:
            label = id_to_label[patient_id_dot]
            data.append(img_path)
            labels.append(label)
            
    print(f"Matched {len(data)} images to labels.")
    
    if len(data) == 0:
        return [], [], None

    # 3. Balance / Subsample
    if sample_size_per_class:
        print(f"Subsampling to {sample_size_per_class} per class...")
        df_matched = pd.DataFrame({'path': data, 'label': labels})
        
        sampled_paths = []
        sampled_labels = []
        
        for label, group in df_matched.groupby('label'):
            n = min(len(group), sample_size_per_class)
            sampled = group.sample(n, random_state=42)
            sampled_paths.extend(sampled['path'].tolist())
            sampled_labels.extend(sampled['label'].tolist())
            
        data = sampled_paths
        labels = sampled_labels
        print(f"Selected {len(data)} images after subsampling.")

    # 4. Load Images into Memory
    print("Loading images into memory...")
    X = []
    valid_labels = []
    
    for i, img_path in enumerate(data):
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if resize:
                img = cv2.resize(img, resize)
                
            # Transpose to (3, H, W) for GP
            img = np.transpose(img, (2, 0, 1))
            
            X.append(img)
            valid_labels.append(labels[i])
            
            if (i+1) % 100 == 0:
                print(f"Loaded {i+1}/{len(data)}...")
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            
    X = np.array(X)
    y_raw = np.array(valid_labels)
    
    # Encode Labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    print(f"Classes: {le.classes_}")
    print(f"Data Shape: {X.shape}")
    
    return X, y, le

def main():
    # 1. Load Data
    # Note: resize=None means keep original size
    X, y, le = load_wsi_data(PATCH_DIR, CSV_PATH, sample_size_per_class=SAMPLE_SIZE_PER_CLASS, resize=None)
    
    if len(X) == 0:
        print("No data loaded. Exiting.")
        return

    # 2. Split Data (Train/Val/Test)
    # Split 1: Hold out Test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Split 2: Split remaining into Train and Val (25% of 80% = 20% total)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # 3. Setup Experiment
    runner = gp_experiment.ExperimentRunner(experiment_name=EXP_NAME)
    
    # 4. Setup Data in Runner
    runner.setup_data(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=BATCH_SIZE)
    
    # 5. Configure GP
    pset = gp_setup.create_primitive_set()
    
    runner.setup_gp(pset, pop_size=POPULATION_SIZE, generations=GENERATIONS)
    
    # 6. Run
    runner.run()

if __name__ == "__main__":
    main()
