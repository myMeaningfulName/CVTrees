import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from deap import base, creator, tools, gp, algorithms
import sys
import os
import random
import warnings
import time

# Suppress warnings
warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src import gp_setup, gp_ops, gp_types, gp_context, gp_draw, gp_utils

# --- Configuration ---
POPULATION_SIZE = 20
GENERATIONS = 50
SAMPLES_PER_CLASS = 5 # Small number for speed in testing
BATCH_SIZE = 50
CROSSOVER_PROB = 0.5
MUTATION_PROB = 0.2

import tarfile
import urllib.request
import pickle

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
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=test_size, stratify=y_sub, random_state=42)
    
    print(f"Data Loaded. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test

def make_image_iterator(X, batch_size=32):
    """
    Creates a factory for an iterator that yields Batches of images.
    X is expected to be (N, 3, H, W).
    """
    def iterator():
        n_samples = len(X)
        for i in range(0, n_samples, batch_size):
            batch_data = X[i:i+batch_size]
            # gp_ops expects List[np.ndarray] for channels [R, G, B]
            # batch_data is (B, 3, H, W)
            r = batch_data[:, 0, :, :]
            g = batch_data[:, 1, :, :]
            b = batch_data[:, 2, :, :]
            yield gp_types.Batch([r, g, b])
    return iterator

def eval_individual(individual, pset, X_train, y_train, batch_size):
    # 1. Compile
    # Use the lazy compilation logic manually or via toolbox if passed
    # We'll just replicate the lazy logic here to be sure
    try:
        code = str(individual)
        code = f"lambda: {code}"
        func = eval(code, pset.context, {})
    except Exception as e:
        return (0.0,) # Invalid tree

    # 2. Setup Context for Training
    # We must reset context to clear old models and node counters
    gp_context.context.reset(gp_context.ExecutionMode.TRAIN)
    gp_context.context.set_train_labels(y_train)
    
    # 3. Setup Data
    iterator_factory = make_image_iterator(X_train, batch_size)
    input_image = gp_types.Image(iterator_factory)
    gp_ops.set_image(input_image)
    
    # 4. Execute
    try:
        # func() builds the pipeline
        pipeline = func()
        
        # Consume pipeline to train
        preds = []
        for batch in pipeline:
            preds.append(batch.data)
            
        if not preds:
            return (0.0,)
            
        preds = np.vstack(preds)
        y_pred = np.argmax(preds, axis=1)
        acc = accuracy_score(y_train, y_pred)
        
        # Optional: Penalize depth/size?
        # For now, just accuracy.
        return (acc,)
        
    except Exception as e:
        # print(f"Eval failed: {e}")
        return (0.0,)

def main():
    # 1. Load Data
    X_train, y_train, X_test, y_test = load_cifar10_subset(samples_per_class=SAMPLES_PER_CLASS)
    
    # 2. Setup DEAP
    pset = gp_setup.create_primitive_set()
    
    if hasattr(creator, "FitnessMax"): del creator.FitnessMax
    if hasattr(creator, "Individual"): del creator.Individual
        
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    # Use gen_safe to handle types without terminals (Prediction, FeatureVector)
    # We must pass type_=pset.ret (Prediction) explicitly
    toolbox.register("expr", gp_utils.gen_safe, pset=pset, min_=2, max_=5, type_=gp_types.Prediction)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Custom compile to return a callable (lazy evaluation)
    def compile_lazy(expr, pset):
        code = str(expr)
        # Wrap in lambda to prevent immediate execution
        code = f"lambda: {code}"
        return eval(code, pset.context, {})
        
    toolbox.register("compile", compile_lazy, pset=pset)
    
    toolbox.register("evaluate", eval_individual, pset=pset, X_train=X_train, y_train=y_train, batch_size=BATCH_SIZE)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    # Use gen_safe for mutation too
    toolbox.register("expr_mut", gp_utils.gen_safe, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    
    # Decorate mate/mutate to limit depth
    toolbox.decorate("mate", gp.staticLimit(key=lambda ind: ind.height, max_value=10))
    toolbox.decorate("mutate", gp.staticLimit(key=lambda ind: ind.height, max_value=10))

    # 3. Evolution
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    
    print(f"Starting Evolution: Pop={POPULATION_SIZE}, Gens={GENERATIONS}")
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CROSSOVER_PROB, mutpb=MUTATION_PROB, 
                                   ngen=GENERATIONS, stats=mstats, 
                                   halloffame=hof, verbose=True)
    
    # 4. Test Best Individual
    best_ind = hof[0]
    print("\nBest Individual Found:")
    print(str(best_ind))
    print(f"Best Train Accuracy: {best_ind.fitness.values[0]*100:.2f}%")
    
    # Draw the best individual
    try:
        print("Drawing best individual...")
        gp_draw.draw_tree(best_ind, filename="best_tree_cifar10.png")
    except Exception as e:
        print(f"Failed to draw tree: {e}")
    
    # 5. Retrain Best on Full Train Set (to populate context models)
    # We need to run it one last time in TRAIN mode to ensure the context has the models for THIS tree.
    # (The context might have models from the last evaluated individual in the population, which might not be the best one)
    print("\nRetraining Best Individual to restore state...")
    gp_context.context.reset(gp_context.ExecutionMode.TRAIN)
    gp_context.context.set_train_labels(y_train)
    
    iterator_factory_train = make_image_iterator(X_train, BATCH_SIZE)
    gp_ops.set_image(gp_types.Image(iterator_factory_train))
    
    func = toolbox.compile(expr=best_ind)
    pipeline = func()
    
    # Consume to train
    for _ in pipeline: pass
    
    # 6. Evaluate on Test Set
    print("Evaluating on Test Set...")
    gp_context.context.reset(gp_context.ExecutionMode.EVAL)
    
    iterator_factory_test = make_image_iterator(X_test, BATCH_SIZE)
    gp_ops.set_image(gp_types.Image(iterator_factory_test))
    
    # Reuse pipeline? No, func() creates a new pipeline structure but reuses Node IDs if we are careful?
    # Wait, `func` is the compiled lambda. Calling it again generates a NEW iterator chain.
    # But `rf_classification` calls `context.get_next_node_id()`.
    # If we call `func()` again, it will generate NEW Node IDs!
    # This is a problem.
    # `rf_classification` is a function called at pipeline construction time.
    # If we re-call `func()`, we re-execute `rf_classification` setup logic, which increments node counters.
    
    # FIX: We need to ensure that when we switch to EVAL, we use the SAME Node IDs.
    # But `func` is a lambda that calls `rf_classification(...)`.
    # Every time we call `func()`, it builds a new graph.
    
    # How did `test_mnist_gp.py` work?
    # It called `func()` ONCE.
    # Then iterated the result (pipeline) twice?
    # No, `pipeline` is an iterator. Once consumed, it's done.
    # Wait, `DataWrapper` has `__iter__` which calls `iterator_factory`.
    # So `pipeline` IS reusable if `rf_classification` returns a `Prediction` (which is a `DataWrapper`).
    
    # Let's check `rf_classification` in `gp_ops.py`:
    # return Prediction(iterator_factory)
    # Yes! It returns a DataWrapper.
    # So we can iterate `pipeline` multiple times.
    
    # So, we do NOT call `func()` again. We use the `pipeline` object returned during the "Retraining" step.
    # But wait, `pipeline` is bound to the `iterator_factory` of the inputs.
    # The inputs (Image) were created with `iterator_factory_train`.
    # If we iterate `pipeline` again, it will iterate `iterator_factory_train` again!
    # We want it to iterate `iterator_factory_test`.
    
    # This is the tricky part of the "Global Context" + "Functional" approach.
    # The `pipeline` object is hardwired to the input `Image` object.
    # The `Image` object is hardwired to `iterator_factory_train`.
    
    # We need a way to SWAP the data source of the `Image` object without rebuilding the tree.
    # `gp_types.Image` has `_iterator_factory`. We can swap it.
    
    print("Swapping input data source to Test Set...")
    # We need access to the `input_image` object that was passed to `gp_ops.set_image`.
    # But we created it inside the `Retraining` block and didn't keep a reference easily accessible 
    # unless we grab it from `gp_ops._CURRENT_IMAGE`.
    
    current_image_wrapper = gp_ops._CURRENT_IMAGE
    # Update its factory
    current_image_wrapper._iterator_factory = iterator_factory_test
    
    # Now iterate the SAME pipeline object
    test_probs = []
    for batch in pipeline:
        test_probs.append(batch.data)
        
    if test_probs:
        test_probs = np.vstack(test_probs)
        y_pred_test = np.argmax(test_probs, axis=1)
        acc_test = accuracy_score(y_test, y_pred_test)
        print(f"Test Accuracy: {acc_test * 100:.2f}%")
    else:
        print("No output on test set.")

if __name__ == "__main__":
    main()
