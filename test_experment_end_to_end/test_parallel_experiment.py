"""
Test script for parallel GP experiment runner.

This demonstrates how to use the parallelized experiment runner which
distributes tree evaluations across multiple CPU cores.

Usage:
    python test_parallel_experiment.py

On SLURM:
    #SBATCH --cpus-per-task=8
    srun python test_parallel_experiment.py
"""

import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import gp_setup
from src.gp_experiment_parallel import ParallelExperimentRunner, run_parallel_experiment
from src.gp_parallel import get_available_workers


def load_cifar10_subset(n_train=500, n_val=100, n_test=100):
    """
    Load a subset of CIFAR-10 for testing.
    Returns data in (N, C, H, W) format.
    """
    import pickle
    
    data_dir = "data/cifar-10-batches-py"
    
    # Load training batch
    with open(os.path.join(data_dir, "data_batch_1"), "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    
    X = batch[b"data"]
    y = np.array(batch[b"labels"])
    
    # Reshape to (N, 3, 32, 32)
    X = X.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    
    # Split into train/val/test
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    X_test = X[n_train + n_val:n_train + n_val + n_test]
    y_test = y[n_train + n_val:n_train + n_val + n_test]
    
    print(f"Loaded CIFAR-10 subset: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    print("=" * 60)
    print("Parallel GP Experiment Test")
    print("=" * 60)
    
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
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10_subset(
        n_train=200,  # Small for testing
        n_val=50,
        n_test=50
    )
    
    # Create primitive set
    pset = gp_setup.create_primitive_set()
    
    # Option 1: Using the convenience function
    print("\n--- Running Parallel Experiment ---")
    
    runner = run_parallel_experiment(
        experiment_name="test_parallel_cifar",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        pset=pset,
        pop_size=10,      # Small population for testing
        generations=3,     # Few generations for testing
        crossover_prob=0.5,
        mutation_prob=0.2,
        batch_size=32,
        n_workers=n_workers,  # Or None for auto-detect
        output_dir="experiments"
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
    multiprocessing.set_start_method('fork', force=True)  # Use 'spawn' on Windows
    
    main()
