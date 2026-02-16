import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import tarfile
import urllib.request
import pickle

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src import gp_setup, gp_experiment

# --- Configuration ---
EXP_NAME = "test_cifar_experiment3"
POPULATION_SIZE = 10
GENERATIONS = 10
SAMPLES_PER_CLASS = 64 
BATCH_SIZE = 10

def load_cifar10_subset(samples_per_class=50, test_size=0.2):
    print("Loading CIFAR-10...")
    
    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = url.split("/")[-1]
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, filepath)
        print("Download complete.")
        
    # Extract
    extract_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    if not os.path.exists(extract_dir):
        print("Extracting...")
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(path=data_dir)
            
    # Load batches
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    X_list = []
    y_list = []
    
    # Load all 5 training batches
    for i in range(1, 6):
        batch_file = os.path.join(extract_dir, f'data_batch_{i}')
        d = unpickle(batch_file)
        X_list.append(d[b'data'])
        y_list.extend(d[b'labels'])
        
    # Load test batch
    test_batch = os.path.join(extract_dir, 'test_batch')
    d_test = unpickle(test_batch)
    X_list.append(d_test[b'data'])
    y_list.extend(d_test[b'labels'])
    
    X = np.vstack(X_list)
    y = np.array(y_list)
    
    # Reshape: (N, 3072) -> (N, 3, 32, 32)
    X = X.reshape(-1, 3, 32, 32)
    
    # Stratified Subsampling
    classes = np.unique(y)
    indices = []
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        if len(cls_indices) > samples_per_class:
            selected = np.random.choice(cls_indices, samples_per_class, replace=False)
        else:
            selected = cls_indices
        indices.extend(selected)
    
    np.random.shuffle(indices)
    X_sub = X[indices]
    y_sub = y[indices]
    
    # Split 1: Hold out Test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(X_sub, y_sub, test_size=0.2, stratify=y_sub, random_state=42)
    
    # Split 2: Split remaining into Train and Val (25% of 80% = 20% total)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
    
    print(f"Data Loaded. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    # 1. Load Data
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10_subset(samples_per_class=SAMPLES_PER_CLASS)
    
    # 2. Setup Experiment
    runner = gp_experiment.ExperimentRunner(experiment_name=EXP_NAME)
    
    # 3. Setup Data in Runner
    runner.setup_data(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=BATCH_SIZE)
    
    # 4. Setup GP
    pset = gp_setup.create_primitive_set()
    runner.setup_gp(pset, pop_size=POPULATION_SIZE, generations=GENERATIONS)
    
    # 5. Run
    runner.run()

if __name__ == "__main__":
    main()
