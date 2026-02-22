"""
Test script for parallel GP experiment on CIFAR-10 dataset.

Mirrors the structure of test_parallel_crc_experiment.py but targets the
CIFAR-10 dataset (already stored locally in data/cifar-10-batches-py/).

All paths are resolved relative to the script's own location so the
experiment can be launched from any working directory – including a
SLURM cluster with a matching repo layout.

Usage (local):
    cd test_experment_end_to_end
    python test_parallel_cifar_experiment.py

    # … or from the repo root:
    python test_experment_end_to_end/test_parallel_cifar_experiment.py

On SLURM:
    #SBATCH --cpus-per-task=8
    srun python test_experment_end_to_end/test_parallel_cifar_experiment.py
"""

import os
import sys
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Path bootstrapping – everything is relative to *this* script so the code
# works on any machine / working-directory.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # .../test_experment_end_to_end
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                       # .../CVTrees  (one level up)

# Ensure the project root is importable
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import gp_setup
from src.gp_experiment_parallel import run_parallel_experiment
from src.gp_parallel import get_available_workers
from src.gp_logging import setup_experiment_logging, get_logger, log_memory

# ---------------------------------------------------------------------------
# Configuration – tweak these knobs as needed
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "test_parallel_cifar"

# Controls which primitive set variant is built (mirrors the CRC experiment)
EXTENDED_OPS    = True      # Include extended GP operations
EXTENDED_PARAMS = True      # Include extended GP parameters

POP_SIZE        = 100
GENERATIONS     = 20
CROSSOVER_PROB  = 0.5
MUTATION_PROB   = 0.2
BATCH_SIZE      = 32
MAX_DEPTH       = 5

# Data subset sizes – set to None to use the *full* CIFAR-10 dataset
SAMPLES_PER_SPLIT = 100    # e.g. 2000 for a quick smoke-test
VAL_RATIO         = 0.2     # fraction of training data held out for validation
TEST_RATIO         = 0.2    # we already have a dedicated test batch, this is unused
                             # when loading the official test set


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_cifar_batch(fpath: str):
    """Unpickle a single CIFAR-10 batch file."""
    with open(fpath, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    X = batch[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y = np.array(batch[b"labels"], dtype=np.int64)
    return X, y


def load_cifar10(samples_per_split=None, val_ratio=0.1):
    """
    Load the full CIFAR-10 dataset from the local data directory.

    Uses *all five* training batches (50 000 images) and the official
    test batch (10 000 images).  Optionally sub-samples each split for
    faster iteration.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
        Arrays in (N, C, H, W) format, pixel values in [0, 1].
    """
    data_dir = os.path.join(PROJECT_ROOT, "data", "cifar-10-batches-py")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"CIFAR-10 data not found at {data_dir}.\n"
            "Download it with:\n"
            "  curl -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz\n"
            "  tar xzf cifar-10-python.tar.gz -C data/"
        )

    # ------------------------------------------------------------------
    # 1. Load all training batches
    # ------------------------------------------------------------------
    print("Loading CIFAR-10 training batches...")
    X_parts, y_parts = [], []
    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f"data_batch_{i}")
        X_b, y_b = _load_cifar_batch(batch_path)
        X_parts.append(X_b)
        y_parts.append(y_b)
        print(f"  Loaded data_batch_{i}: {X_b.shape[0]} images")

    X_all_train = np.concatenate(X_parts)
    y_all_train = np.concatenate(y_parts)

    # ------------------------------------------------------------------
    # 2. Load the official test batch
    # ------------------------------------------------------------------
    print("Loading CIFAR-10 test batch...")
    X_test, y_test = _load_cifar_batch(os.path.join(data_dir, "test_batch"))
    print(f"  Loaded test_batch: {X_test.shape[0]} images")

    # ------------------------------------------------------------------
    # 3. (Optional) Sub-sample for faster experiments
    # ------------------------------------------------------------------
    if samples_per_split is not None:
        n_train_cap = min(samples_per_split, len(X_all_train))
        n_test_cap  = min(samples_per_split, len(X_test))

        idx_train = np.random.choice(len(X_all_train), n_train_cap, replace=False)
        X_all_train = X_all_train[idx_train]
        y_all_train = y_all_train[idx_train]

        idx_test = np.random.choice(len(X_test), n_test_cap, replace=False)
        X_test = X_test[idx_test]
        y_test = y_test[idx_test]

        print(f"  Sub-sampled to {n_train_cap} train / {n_test_cap} test")

    # ------------------------------------------------------------------
    # 4. Split training data into train + validation
    # ------------------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_all_train, y_all_train,
        test_size=val_ratio,
        stratify=y_all_train,
        random_state=42,
    )

    print(
        f"Data Loaded.  Train: {X_train.shape}, "
        f"Val: {X_val.shape}, Test: {X_test.shape}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# Load CIFAR-10 class names (for reference / logging)
# ---------------------------------------------------------------------------

def load_cifar10_meta():
    """Return the list of human-readable class names."""
    meta_path = os.path.join(PROJECT_ROOT, "data", "cifar-10-batches-py", "batches.meta")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, "rb") as f:
        meta = pickle.load(f, encoding="bytes")
    return [name.decode("utf-8") for name in meta[b"label_names"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Parallel CIFAR-10 GP Experiment")
    print("=" * 60)

    # --- Logging ---
    experiment_output_dir = os.path.join(PROJECT_ROOT, "experiments")
    setup_experiment_logging(EXPERIMENT_NAME, output_dir=experiment_output_dir)
    logger = get_logger("main")
    logger.info("Parallel CIFAR-10 GP Experiment starting")
    log_memory(label="startup", logger=logger)

    # --- Worker detection ---
    n_workers = get_available_workers()
    logger.info("Detected %d available workers", n_workers)

    # --- SLURM info ---
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        print(f"Running in SLURM job: {slurm_job_id}")
        print(f"  SLURM_CPUS_PER_TASK: {os.environ.get('SLURM_CPUS_PER_TASK', 'not set')}")
        print(f"  SLURM_NODELIST:      {os.environ.get('SLURM_NODELIST', 'not set')}")
    else:
        print("Not running in SLURM environment")

    # --- Class names (informational) ---
    class_names = load_cifar10_meta()
    if class_names:
        print(f"\nCIFAR-10 classes: {class_names}")

    print()

    # ------------------------------------------------------------------
    # 1. Load Data
    # ------------------------------------------------------------------
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10(
        samples_per_split=SAMPLES_PER_SPLIT,
        val_ratio=VAL_RATIO,
    )

    # ------------------------------------------------------------------
    # 2. Create primitive set (with configurable extensions)
    # ------------------------------------------------------------------
    print("\n--- Creating Primitive Set ---")
    print(f"  Extended Operations:  {'ENABLED' if EXTENDED_OPS else 'DISABLED'}")
    print(f"  Extended Parameters:  {'ENABLED' if EXTENDED_PARAMS else 'DISABLED'}")
    pset = gp_setup.create_primitive_set(
        extended_ops=EXTENDED_OPS,
        extended_params=EXTENDED_PARAMS,
    )

    # ------------------------------------------------------------------
    # 3. Run parallel experiment
    # ------------------------------------------------------------------
    print("\n--- Running Parallel Experiment ---")
    print(f"  Population : {POP_SIZE}")
    print(f"  Generations: {GENERATIONS}")
    print(f"  Max Depth  : {MAX_DEPTH}")
    print(f"  Workers    : {n_workers}")

    runner = run_parallel_experiment(
        experiment_name=EXPERIMENT_NAME,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        pset=pset,
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        crossover_prob=CROSSOVER_PROB,
        mutation_prob=MUTATION_PROB,
        batch_size=BATCH_SIZE,
        n_workers=n_workers,
        output_dir=experiment_output_dir,
        max_depth=MAX_DEPTH,
    )

    # ------------------------------------------------------------------
    # 4. Report results
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)
    print("Best Individual:")
    if runner.best_individual_overall:
        print(f"  Generation:          {runner.best_individual_overall['generation']}")
        print(f"  Validation Accuracy: {runner.best_individual_overall['val_acc']:.4f}")
        print(f"  Test Accuracy:       {runner.best_individual_overall['test_acc']:.4f}")
        print(f"  Tree Depth:          {runner.best_individual_overall['depth']}")
    else:
        print("  (no individual recorded)")

    print(f"\nResults saved to: {runner.base_dir}")


if __name__ == "__main__":
    # Important for multiprocessing on some platforms
    import multiprocessing
    if sys.platform != "win32":
        multiprocessing.set_start_method("fork", force=True)

    main()
