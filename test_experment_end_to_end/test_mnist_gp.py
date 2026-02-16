import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, gp
from sklearn.model_selection import train_test_split
import sys
import os
import random
import warnings
warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src import gp_setup, gp_ops, gp_types, gp_context

def load_mnist(n_samples=1000, test_size=0.2):
    print("Loading MNIST...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    X = X[:n_samples].reshape(-1, 28, 28)
    y = y[:n_samples].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def make_image_iterator(X, batch_size=32):
    def iterator():
        n_samples = len(X)
        for i in range(0, n_samples, batch_size):
            batch_data = X[i:i+batch_size]
            # Fake RGB: (N, 28, 28) -> (N, 3, 28, 28)
            # Or List of (N, 28, 28)
            # gp_ops expects List[np.ndarray] for channels or (N, C, H, W)
            # Let's use List[np.ndarray] for R, G, B
            r = batch_data
            g = batch_data
            b = batch_data
            yield gp_types.Batch([r, g, b])
    return iterator

def main():
    # 1. Load Data
    n_samples = 200
    batch_size = 50
    X_train, X_test, y_train, y_test = load_mnist(n_samples=n_samples)
    print(f"Loaded {n_samples} samples: {len(X_train)} train, {len(X_test)} test.")
    
    # 2. Prepare Training Data for GP
    train_iterator_factory = make_image_iterator(X_train, batch_size)
    train_image = gp_types.Image(train_iterator_factory)

    # 3. Prepare Test Data for GP
    test_iterator_factory = make_image_iterator(X_test, batch_size)
    test_image = gp_types.Image(test_iterator_factory)
    
    # 4. Setup DEAP
    pset = gp_setup.create_primitive_set()
    
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genGrow, pset=pset, min_=2, max_=5)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    
    def compile_lazy(expr, pset):
        code = str(expr)
        code = f"lambda: {code}"
        return eval(code, pset.context, {})
        
    toolbox.register("compile", compile_lazy, pset=pset)
    
    # 5. Create a Tree
    print("Generating Tree...")
    # Force a specific tree structure if random is too chaotic, or just try random
    # Let's try random first.
    individual = toolbox.individual()
    print("Tree generated:")
    print(str(individual))
    
    # 6. Evaluate in TRAIN mode
    print("\n--- TRAIN MODE ---")
    gp_context.context.reset(gp_context.ExecutionMode.TRAIN)
    gp_context.context.set_train_labels(y_train)
    
    gp_ops.set_image(train_image)
    
    func = toolbox.compile(expr=individual)
    
    # Execute
    # We call func() ONCE to build the computation graph and assign persistent Node IDs.
    # The result 'pipeline' is a DataWrapper that can be iterated multiple times.
    pipeline = func()
    
    # Consume the iterator to trigger training
    print("Executing Training...")
    train_probs = []
    for batch in pipeline:
        train_probs.append(batch.data)
        
    if train_probs:
        train_probs = np.vstack(train_probs)
        print(f"Train Output shape: {train_probs.shape}")
        y_pred = np.argmax(train_probs, axis=1)
        acc = accuracy_score(y_train, y_pred)
        print(f"Train Accuracy (CV-like): {acc * 100:.2f}%")
    else:
        print("No output from training execution.")

    # 7. Evaluate in EVAL mode
    print("\n--- EVAL MODE ---")
    gp_context.context.reset(gp_context.ExecutionMode.EVAL)

    # Use the test data for evaluation
    gp_ops.set_image(test_image)

    # Reuse the SAME pipeline object to preserve Node IDs
    print("Executing Evaluation...")
    eval_probs = []
    for batch in pipeline:
        eval_probs.append(batch.data)

    if eval_probs:
        eval_probs = np.vstack(eval_probs)
        print(f"Eval Output shape: {eval_probs.shape}")
        y_pred_eval = np.argmax(eval_probs, axis=1)
        acc_eval = accuracy_score(y_test, y_pred_eval)
        print(f"Eval Accuracy: {acc_eval * 100:.2f}%")
        
        # Verify consistency
    else:
        print("No output from eval execution.")

if __name__ == "__main__":
    main()

