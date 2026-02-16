"""
Parallel evaluation module for GP experiments.

This module provides worker functions and utilities for evaluating GP individuals
in parallel using multiprocessing. Each worker has its own isolated context to
prevent any memory sharing or information leakage between evaluations.

Key design principles:
1. Workers are completely isolated - no shared memory between processes
2. Each worker has its own GPContext instance
3. All global state (_CURRENT_IMAGE, context) is managed per-worker
4. Results are serialized and returned to the main process
5. File system documentation is done by workers to avoid bottlenecks
"""

import os
import json
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from sklearn.metrics import accuracy_score
from deap import gp, base, creator, tools

# Import types - these don't have global state
from .gp_types import Image, Channel, FeatureVector, Prediction, Batch
from .gp_context import GPContext, ExecutionMode


def get_available_workers() -> int:
    """
    Detect available CPUs for parallel evaluation.
    
    Priority:
    1. SLURM_CPUS_PER_TASK environment variable (if on SLURM cluster)
    2. os.cpu_count() as fallback
    
    Returns k-1 workers (leaving 1 CPU for main process).
    """
    # Try SLURM environment variable first
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    
    if slurm_cpus is not None:
        try:
            n_cpus = int(slurm_cpus)
            n_workers = max(1, n_cpus - 1)
            print(f"[PARALLEL] SLURM detected: {n_cpus} CPUs allocated, using {n_workers} workers")
            return n_workers
        except ValueError:
            print(f"[PARALLEL] Warning: Could not parse SLURM_CPUS_PER_TASK='{slurm_cpus}'")
    
    # Fallback to system CPU count
    n_cpus = os.cpu_count() or 2
    n_workers = max(1, n_cpus - 1)
    print(f"[PARALLEL] System detected: {n_cpus} CPUs available, using {n_workers} workers")
    
    return n_workers


@dataclass
class EvaluationTask:
    """
    A task to be evaluated by a worker.
    Contains all information needed to evaluate an individual in complete isolation.
    """
    individual_idx: int  # Index in population
    individual_str: str  # String representation of the tree (for compilation)
    gen_num: int
    tree_output_dir: str  # Where to save results


@dataclass
class EvaluationResult:
    """
    Result of evaluating an individual.
    Returned from worker to main process.
    """
    individual_idx: int
    train_acc: float
    val_acc: float
    test_acc: float
    success: bool
    error_msg: Optional[str] = None


class WorkerContext:
    """
    Encapsulates all per-worker state including context and current image.
    This replaces the global context and _CURRENT_IMAGE for parallel execution.
    """
    def __init__(self):
        self.context = GPContext()
        self.current_image: Optional[Image] = None
    
    def set_image(self, image: Image):
        self.current_image = image
    
    def get_image(self) -> Image:
        if self.current_image is None:
            raise ValueError("Image not set in worker context")
        return self.current_image


# Global worker context (process-local due to multiprocessing fork)
_worker_context: Optional[WorkerContext] = None
_worker_data: Optional[Dict] = None
_worker_pset: Optional[gp.PrimitiveSetTyped] = None


def _init_worker(X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray,
                 batch_size: int, pset_context: Dict):
    """
    Initialize worker process with its own isolated state.
    Called once per worker when the pool is created.
    """
    global _worker_context, _worker_data, _worker_pset
    
    # Create fresh context for this worker
    _worker_context = WorkerContext()
    
    # Store data reference (copy-on-write in most cases due to fork)
    _worker_data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'batch_size': batch_size
    }
    
    # Store pset context for compilation
    _worker_pset = pset_context
    
    # Setup DEAP creator in this process if not already done
    if not hasattr(creator, 'FitnessMax'):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, 'Individual'):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    
    pid = os.getpid()
    print(f"[WORKER {pid}] Initialized with batch_size={batch_size}")


def _make_image_iterator_worker(X: np.ndarray, batch_size: int):
    """
    Creates a factory for an iterator that yields Batches of images.
    Worker-local version that doesn't depend on any instance state.
    """
    def iterator():
        n_samples = len(X)
        for i in range(0, n_samples, batch_size):
            batch_data = X[i:i+batch_size]
            r = batch_data[:, 0, :, :]
            g = batch_data[:, 1, :, :]
            b = batch_data[:, 2, :, :]
            yield Batch([r, g, b])
    return iterator


def _get_channel_iter(worker_ctx: WorkerContext, channel_idx: int):
    """Generator for a specific color channel."""
    image = worker_ctx.get_image()
    for batch in image:
        if isinstance(batch.data, list):
            yield Batch(batch.data[channel_idx])
        else:
            yield Batch(batch.data[:, channel_idx])


def _get_gray_iter_worker(worker_ctx: WorkerContext):
    """Generator for grayscale channel."""
    image = worker_ctx.get_image()
    for batch in image:
        if isinstance(batch.data, list):
            r, g, b = batch.data[0], batch.data[1], batch.data[2]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            yield Batch(gray.astype(np.float32))
        else:
            gray = 0.299 * batch.data[:, 0] + 0.587 * batch.data[:, 1] + 0.114 * batch.data[:, 2]
            yield Batch(gray.astype(np.float32))


def _setup_worker_ops_context(worker_ctx: WorkerContext):
    """
    Setup the gp_ops module to use worker-local context.
    This patches the module's globals for this worker process.
    """
    from . import gp_ops
    
    # Patch the context reference in gp_ops
    gp_ops.context = worker_ctx.context
    
    # Patch set_image to use worker context
    def worker_set_image(image: Image):
        worker_ctx.set_image(image)
        gp_ops._CURRENT_IMAGE = image
    
    gp_ops.set_image = worker_set_image
    
    # Create terminals that use worker context
    def make_red_iter():
        return _get_channel_iter(worker_ctx, 0)
    
    def make_green_iter():
        return _get_channel_iter(worker_ctx, 1)
    
    def make_blue_iter():
        return _get_channel_iter(worker_ctx, 2)
    
    def make_gray_iter():
        return _get_gray_iter_worker(worker_ctx)
    
    # Update the terminal objects
    gp_ops.RedTerminal = Channel(make_red_iter)
    gp_ops.GreenTerminal = Channel(make_green_iter)
    gp_ops.BlueTerminal = Channel(make_blue_iter)
    gp_ops.GrayTerminal = Channel(make_gray_iter)


def _evaluate_pipeline_worker(expr_str: str, X: np.ndarray, y: np.ndarray,
                              mode: ExecutionMode, batch_size: int) -> Tuple[float, Optional[str]]:
    """
    Evaluate an individual on a dataset in worker process.
    Uses worker-local context.
    
    Returns:
        Tuple of (accuracy, error_message). error_message is None on success.
    """
    global _worker_context, _worker_pset
    
    pid = os.getpid()
    
    try:
        # Compile the expression
        code = f"lambda: {expr_str}"
        func = eval(code, _worker_pset, {})
        
        # Setup context for this evaluation
        _setup_worker_ops_context(_worker_context)
        _worker_context.context.reset(mode)
        
        if mode == ExecutionMode.TRAIN:
            _worker_context.context.set_train_labels(y)
        
        # Setup data iterator
        iterator_factory = _make_image_iterator_worker(X, batch_size)
        input_image = Image(iterator_factory)
        
        from . import gp_ops
        gp_ops.set_image(input_image)
        
        # Execute pipeline
        pipeline = func()
        
        preds = []
        for batch in pipeline:
            preds.append(batch.data)
        
        if not preds:
            return 0.0, f"No predictions produced (empty pipeline output)"
        
        preds = np.vstack(preds)
        y_pred = np.argmax(preds, axis=1)
        acc = accuracy_score(y, y_pred)
        return acc, None
        
    except Exception as e:
        import traceback
        error_msg = f"[WORKER {pid}] Evaluation failed: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        return 0.0, error_msg


def evaluate_individual_worker(task: EvaluationTask) -> EvaluationResult:
    """
    Worker function to evaluate a single individual.
    This is the main entry point called by the multiprocessing pool.
    
    Each call is completely isolated - no state persists between evaluations.
    """
    global _worker_context, _worker_data
    
    pid = os.getpid()
    
    try:
        # Reset worker context for this evaluation (fresh context per tree)
        _worker_context = WorkerContext()
        
        # Get data
        X_train = _worker_data['X_train']
        y_train = _worker_data['y_train']
        X_val = _worker_data['X_val']
        y_val = _worker_data['y_val']
        X_test = _worker_data['X_test']
        y_test = _worker_data['y_test']
        batch_size = _worker_data['batch_size']
        
        # Collect errors from all phases
        errors = []
        
        # 1. Train (fit models in the tree)
        train_acc, train_err = _evaluate_pipeline_worker(
            task.individual_str, X_train, y_train,
            ExecutionMode.TRAIN, batch_size
        )
        if train_err:
            errors.append(f"TRAIN: {train_err}")
        
        # 2. Validation (for fitness)
        val_acc, val_err = _evaluate_pipeline_worker(
            task.individual_str, X_val, y_val,
            ExecutionMode.EVAL, batch_size
        )
        if val_err:
            errors.append(f"VAL: {val_err}")
        
        # 3. Test (for reporting)
        test_acc, test_err = _evaluate_pipeline_worker(
            task.individual_str, X_test, y_test,
            ExecutionMode.EVAL, batch_size
        )
        if test_err:
            errors.append(f"TEST: {test_err}")
        
        # Determine success and error message
        success = len(errors) == 0
        error_msg = "\n".join(errors) if errors else None
        
        # Save individual results to filesystem (worker responsibility)
        if task.tree_output_dir:
            os.makedirs(task.tree_output_dir, exist_ok=True)
            
            stats = {
                "generation": task.gen_num,
                "individual_id": task.individual_idx,
                "train_accuracy": float(train_acc),
                "validation_accuracy": float(val_acc),
                "test_accuracy": float(test_acc),
                "expression": task.individual_str,
                "worker_pid": pid,
                "error": error_msg  # Now we capture errors in the results
            }
            
            results_path = os.path.join(task.tree_output_dir, "results.json")
            with open(results_path, "w") as f:
                json.dump(stats, f, indent=4)
        
        return EvaluationResult(
            individual_idx=task.individual_idx,
            train_acc=train_acc,
            val_acc=val_acc,
            test_acc=test_acc,
            success=success,
            error_msg=error_msg
        )
        
    except Exception as e:
        import traceback
        error_msg = f"[WORKER {pid}] Top-level error: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        return EvaluationResult(
            individual_idx=task.individual_idx,
            train_acc=0.0,
            val_acc=0.0,
            test_acc=0.0,
            success=False,
            error_msg=error_msg
        )


class ParallelEvaluator:
    """
    Manages parallel evaluation of GP individuals using a process pool.
    
    Usage:
        evaluator = ParallelEvaluator(X_train, y_train, X_val, y_val, X_test, y_test,
                                       batch_size, pset.context, n_workers)
        results = evaluator.evaluate_population(population, gen_num, output_dir)
        evaluator.shutdown()
    """
    
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray,
                 batch_size: int, pset_context: Dict,
                 n_workers: Optional[int] = None):
        """
        Initialize the parallel evaluator with a process pool.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data  
            X_test, y_test: Test data
            batch_size: Batch size for evaluation
            pset_context: The primitive set context dictionary for compilation
            n_workers: Number of worker processes (default: auto-detect)
        """
        self.n_workers = n_workers if n_workers else get_available_workers()
        
        print(f"[PARALLEL] Creating process pool with {self.n_workers} workers")
        
        # Create pool with initializer
        self.pool = Pool(
            processes=self.n_workers,
            initializer=_init_worker,
            initargs=(X_train, y_train, X_val, y_val, X_test, y_test,
                      batch_size, pset_context)
        )
        
        self.batch_size = batch_size
    
    def evaluate_population(self, population: List, gen_num: int,
                           base_output_dir: str) -> List[EvaluationResult]:
        """
        Evaluate an entire population in parallel.
        
        Args:
            population: List of DEAP individuals
            gen_num: Current generation number
            base_output_dir: Base directory for output
            
        Returns:
            List of EvaluationResult objects (in same order as population)
        """
        gen_dir = os.path.join(base_output_dir, f"gen_{gen_num}")
        os.makedirs(gen_dir, exist_ok=True)
        
        # Create tasks for each individual
        tasks = []
        for idx, ind in enumerate(population):
            tree_dir = os.path.join(gen_dir, f"tree_{idx}")
            task = EvaluationTask(
                individual_idx=idx,
                individual_str=str(ind),
                gen_num=gen_num,
                tree_output_dir=tree_dir
            )
            tasks.append(task)
        
        # Submit tasks and collect results
        print(f"[PARALLEL] Submitting {len(tasks)} evaluation tasks...")
        
        # Use imap_unordered for better load balancing
        results_dict = {}
        for result in self.pool.imap_unordered(evaluate_individual_worker, tasks):
            results_dict[result.individual_idx] = result
            if result.success:
                print(f"[PARALLEL] Tree {result.individual_idx}: "
                      f"train={result.train_acc:.4f}, val={result.val_acc:.4f}, "
                      f"test={result.test_acc:.4f}")
            else:
                print(f"[PARALLEL] Tree {result.individual_idx}: FAILED - {result.error_msg}")
        
        # Return results in original order
        results = [results_dict[i] for i in range(len(population))]
        return results
    
    def evaluate_individuals(self, individuals: List, indices: List[int],
                            gen_num: int, base_output_dir: str) -> List[EvaluationResult]:
        """
        Evaluate a subset of individuals (e.g., only invalid ones after mutation).
        
        Args:
            individuals: List of individuals to evaluate
            indices: Their indices in the original population
            gen_num: Current generation number
            base_output_dir: Base directory for output
            
        Returns:
            List of EvaluationResult objects
        """
        gen_dir = os.path.join(base_output_dir, f"gen_{gen_num}")
        os.makedirs(gen_dir, exist_ok=True)
        
        # Create tasks
        tasks = []
        for ind, idx in zip(individuals, indices):
            tree_dir = os.path.join(gen_dir, f"tree_{idx}")
            task = EvaluationTask(
                individual_idx=idx,
                individual_str=str(ind),
                gen_num=gen_num,
                tree_output_dir=tree_dir
            )
            tasks.append(task)
        
        print(f"[PARALLEL] Submitting {len(tasks)} evaluation tasks...")
        
        # Collect results
        results = []
        for result in self.pool.imap_unordered(evaluate_individual_worker, tasks):
            results.append(result)
            if result.success:
                print(f"[PARALLEL] Tree {result.individual_idx}: "
                      f"train={result.train_acc:.4f}, val={result.val_acc:.4f}, "
                      f"test={result.test_acc:.4f}")
        
        return results
    
    def shutdown(self):
        """Clean up the process pool."""
        print("[PARALLEL] Shutting down worker pool...")
        self.pool.close()
        self.pool.join()
        print("[PARALLEL] Pool shutdown complete")
