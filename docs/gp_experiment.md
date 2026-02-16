# GP Experiment Module (`gp_experiment.py`) - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [The ExperimentRunner Class](#the-experimentrunner-class)
3. [DEAP Framework Integration](#deap-framework-integration)
4. [Data Setup](#data-setup)
5. [GP System Setup](#gp-system-setup)
6. [Tree Compilation](#tree-compilation)
7. [Evaluation Pipeline](#evaluation-pipeline)
8. [The Evolutionary Loop](#the-evolutionary-loop)
9. [Saving and Logging](#saving-and-logging)
10. [Complete Workflow Example](#complete-workflow-example)
11. [Complete Code Walkthrough](#complete-code-walkthrough)

---

## Overview

The `gp_experiment.py` module contains the `ExperimentRunner` class, which orchestrates the entire Genetic Programming experiment. It ties together all other modules:

- Uses `gp_setup` to create the primitive set
- Uses `gp_ops` to execute tree operations
- Uses `gp_context` to manage training/evaluation state
- Uses `gp_utils` for safe tree generation
- Uses `gp_draw` for visualization
- Uses DEAP for evolutionary operations

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT RUNNER FLOW                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. INITIALIZATION                                              │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │ ExperimentRunner("my_experiment")                         │  │
│   │  └─► Creates experiment directory                         │  │
│   └──────────────────────────────────────────────────────────┘  │
│                          ▼                                       │
│   2. DATA SETUP                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │ setup_data(X_train, y_train, X_val, y_val, X_test, y_test)│  │
│   │  └─► Stores training, validation, and test data          │  │
│   └──────────────────────────────────────────────────────────┘  │
│                          ▼                                       │
│   3. GP SETUP                                                    │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │ setup_gp(pset, pop_size=50, generations=20, ...)          │  │
│   │  └─► Configures DEAP toolbox with operators               │  │
│   └──────────────────────────────────────────────────────────┘  │
│                          ▼                                       │
│   4. RUN EVOLUTION                                               │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │ run()                                                     │  │
│   │  ├─► Initialize population                                │  │
│   │  ├─► Evaluate all individuals                             │  │
│   │  ├─► For each generation:                                 │  │
│   │  │    ├─► Select parents                                  │  │
│   │  │    ├─► Apply crossover                                 │  │
│   │  │    ├─► Apply mutation                                  │  │
│   │  │    ├─► Evaluate offspring                              │  │
│   │  │    └─► Save generation results                         │  │
│   │  └─► Save experiment summary                              │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## The ExperimentRunner Class

### Constructor

```python
class ExperimentRunner:
    def __init__(self, experiment_name, output_dir="experiments"):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.base_dir = os.path.join(output_dir, experiment_name)
        
        if os.path.exists(self.base_dir):
            print(f"Warning: Experiment directory {self.base_dir} exists.")
        else:
            os.makedirs(self.base_dir)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `experiment_name` | `str` | Required | Name for this experiment |
| `output_dir` | `str` | `"experiments"` | Parent directory for all experiments |

### Instance Attributes

| Attribute | Type | Purpose |
|-----------|------|---------|
| `experiment_name` | `str` | Identifier for this run |
| `output_dir` | `str` | Parent directory |
| `base_dir` | `str` | Full path: `output_dir/experiment_name` |
| `X_train`, `y_train` | `np.ndarray` | Training data and labels |
| `X_val`, `y_val` | `np.ndarray` | Validation data and labels |
| `X_test`, `y_test` | `np.ndarray` | Test data and labels |
| `batch_size` | `int` | Batch size for processing |
| `pset` | `PrimitiveSetTyped` | The GP primitive set |
| `toolbox` | `base.Toolbox` | DEAP's operator registry |
| `pop_size` | `int` | Population size |
| `generations` | `int` | Number of generations |
| `cx_prob` | `float` | Crossover probability |
| `mut_prob` | `float` | Mutation probability |
| `all_generations_stats` | `list` | Statistics for all generations |
| `best_individual_overall` | `dict` | Best tree found |
| `best_fitness_overall` | `float` | Best validation accuracy |

### Directory Structure Created

```
experiments/
└── my_experiment/
    ├── gen_0/
    │   ├── generation_summary.json
    │   ├── tree_0/
    │   │   ├── results.json
    │   │   └── tree_viz.png
    │   ├── tree_1/
    │   │   ├── results.json
    │   │   └── tree_viz.png
    │   └── ...
    ├── gen_1/
    │   └── ...
    ├── gen_N/
    │   └── ...
    ├── experiment_summary.json
    └── progress_plot.png
```

---

## DEAP Framework Integration

### What Is DEAP?

DEAP (Distributed Evolutionary Algorithms in Python) is a library for evolutionary computation. It provides:

- **Genetic operators**: Selection, crossover, mutation
- **Data structures**: Individuals, populations
- **GP tools**: Tree representation, primitives, typed GP

### Key DEAP Components Used

#### `creator` Module

Creates custom classes for individuals and fitness:

```python
from deap import creator, base

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
```

- **FitnessMax**: Fitness class where higher is better (weight = 1.0)
- **Individual**: A GP tree with an attached fitness

#### `base.Toolbox`

A registry for evolutionary operators:

```python
toolbox = base.Toolbox()
toolbox.register("expr", ...)        # Tree expression generator
toolbox.register("individual", ...)  # Individual creator
toolbox.register("population", ...)  # Population creator
toolbox.register("select", ...)      # Selection operator
toolbox.register("mate", ...)        # Crossover operator
toolbox.register("mutate", ...)      # Mutation operator
```

#### `gp.PrimitiveTree`

DEAP's tree representation:

```python
# A tree is stored as a list in prefix notation
tree = [rf_classification, hog_features, GetRed, 100, 50]
# Represents: rf_classification(hog_features(GetRed), 100, 50)

# Access properties
tree.height  # Depth of the tree
len(tree)    # Number of nodes
str(tree)    # String representation
```

---

## Data Setup

### The `setup_data` Method

```python
def setup_data(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    self.X_train = X_train
    self.y_train = y_train
    self.X_val = X_val
    self.y_val = y_val
    self.X_test = X_test
    self.y_test = y_test
    self.batch_size = batch_size
```

### Expected Data Format

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `X_train` | `(N, 3, H, W)` | Training images (N samples, 3 channels, H×W pixels) |
| `y_train` | `(N,)` | Training labels (integers 0 to K-1) |
| `X_val` | `(M, 3, H, W)` | Validation images |
| `y_val` | `(M,)` | Validation labels |
| `X_test` | `(P, 3, H, W)` | Test images |
| `y_test` | `(P,)` | Test labels |

### The Three-Split Strategy

| Split | Purpose | Used For |
|-------|---------|----------|
| Training | Fit models inside trees | Classifier training |
| Validation | Compute fitness | Selection, evolution |
| Test | Final evaluation | Reporting (not used in evolution) |

### Why Validation for Fitness?

Using **validation** accuracy for fitness instead of training:
- Prevents overfitting (trees that memorize training data)
- Gives more realistic performance estimates
- Better generalization to unseen data

### The `_make_image_iterator` Method

```python
def _make_image_iterator(self, X):
    """
    Creates a factory for an iterator that yields Batches of images.
    """
    batch_size = self.batch_size
    def iterator():
        n_samples = len(X)
        for i in range(0, n_samples, batch_size):
            batch_data = X[i:i+batch_size]
            r = batch_data[:, 0, :, :]
            g = batch_data[:, 1, :, :]
            b = batch_data[:, 2, :, :]
            yield gp_types.Batch([r, g, b])
    return iterator
```

This creates a **factory function** (not an iterator!) that:
1. Takes data X in shape (N, 3, H, W)
2. Splits into batches of `batch_size`
3. Separates into R, G, B channels
4. Yields each batch as `Batch([r, g, b])`

---

## GP System Setup

### The `setup_gp` Method

```python
def setup_gp(self, pset, pop_size=20, generations=10, crossover_prob=0.5, mutation_prob=0.2):
    self.pop_size = pop_size
    self.generations = generations
    self.cx_prob = crossover_prob
    self.mut_prob = mutation_prob
    self.pset = pset
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pset` | Required | The primitive set from `gp_setup.create_primitive_set()` |
| `pop_size` | 20 | Number of individuals per generation |
| `generations` | 10 | Number of generations to evolve |
| `crossover_prob` | 0.5 | Probability of crossover per pair |
| `mutation_prob` | 0.2 | Probability of mutation per individual |

### DEAP Creator Setup

```python
if hasattr(creator, "FitnessMax"): del creator.FitnessMax
if hasattr(creator, "Individual"): del creator.Individual
    
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
```

**Why delete first?**

DEAP's creator adds classes to a global namespace. If you run the experiment multiple times, the classes already exist and can cause issues. Deleting first ensures a clean slate.

**FitnessMax explained**:
- Inherits from `base.Fitness`
- `weights=(1.0,)` means "maximize" (negative weight would minimize)
- The tuple allows multi-objective optimization (we only use one objective)

**Individual explained**:
- Inherits from `gp.PrimitiveTree` (a list-like tree structure)
- Has a `fitness` attribute of type `FitnessMax`

### Toolbox Registration

```python
self.toolbox = base.Toolbox()

# Expression generator
self.toolbox.register("expr", gp_utils.gen_safe, pset=pset, min_=2, max_=5, type_=gp_types.Prediction)

# Individual creator
self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)

# Population creator
self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
```

#### Understanding `tools.initIterate`

```python
tools.initIterate(container, generator)
```
- `container`: The class to create (e.g., `creator.Individual`)
- `generator`: A function that returns content

This calls `generator()` and passes the result to `container()`.

#### Understanding `tools.initRepeat`

```python
tools.initRepeat(container, func, n)
```
- `container`: What to put results in (e.g., `list`)
- `func`: Function to call repeatedly
- `n`: Number of times

This calls `func()` n times and collects results in `container`.

### Genetic Operators

```python
# Compiler
self.toolbox.register("compile", self._compile_lazy)

# Selection: Tournament with size 3
self.toolbox.register("select", tools.selTournament, tournsize=3)

# Crossover: One-point crossover
self.toolbox.register("mate", gp.cxOnePoint)

# Mutation expression generator
self.toolbox.register("expr_mut", gp_utils.gen_safe, min_=0, max_=2)

# Mutation: Uniform mutation with new subtrees
self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=pset)
```

### Decorators (Constraints)

```python
self.toolbox.decorate("mate", gp.staticLimit(key=lambda ind: ind.height, max_value=10))
self.toolbox.decorate("mutate", gp.staticLimit(key=lambda ind: ind.height, max_value=10))
```

**What decorators do**:
- Wrap the operator with additional logic
- `staticLimit` rejects offspring that exceed the limit
- If rejected, returns the original parents unchanged

**Why limit height?**
- Very deep trees are slow to evaluate
- They tend to overfit
- Height 10 is usually sufficient for complex pipelines

---

## Tree Compilation

### The `_compile_lazy` Method

```python
def _compile_lazy(self, expr):
    code = str(expr)
    code = f"lambda: {code}"
    return eval(code, self.pset.context, {})
```

### What This Does

1. **Convert tree to string**:
   ```python
   str(expr)  # "rf_classification(hog_features(GetRed), 100, 50)"
   ```

2. **Wrap in lambda**:
   ```python
   f"lambda: {code}"  # "lambda: rf_classification(hog_features(GetRed), 100, 50)"
   ```

3. **Evaluate to get function**:
   ```python
   eval(code, self.pset.context, {})  # Returns a callable
   ```

### Why Lambda?

The lambda makes the pipeline **lazy**. When we call the function:
```python
func = toolbox.compile(expr=tree)
pipeline = func()  # NOW the tree executes
```

The tree doesn't execute until we call `func()`.

### The `pset.context`

DEAP's primitive set has a `context` dictionary containing all the functions:
```python
pset.context = {
    "rf_classification": gp_ops.rf_classification,
    "hog_features": gp_ops.hog_features,
    "GetRed": gp_ops.RedTerminal,
    # ... etc
}
```

This allows `eval` to resolve the function names.

---

## Evaluation Pipeline

### The `_evaluate_pipeline` Method

```python
def _evaluate_pipeline(self, individual, X, y, mode=gp_context.ExecutionMode.TRAIN):
    """
    Evaluates an individual on a dataset X, y.
    """
    try:
        # 1. Compile (Lazy)
        func = self.toolbox.compile(expr=individual)
        
        # 2. Setup Context
        gp_context.context.reset(mode)
        if mode == gp_context.ExecutionMode.TRAIN:
            gp_context.context.set_train_labels(y)
        
        # 3. Setup Data
        iterator_factory = self._make_image_iterator(X)
        input_image = gp_types.Image(iterator_factory)
        gp_ops.set_image(input_image)
        
        # 4. Execute
        pipeline = func()
        
        preds = []
        for batch in pipeline:
            preds.append(batch.data)
            
        if not preds:
            return 0.0
            
        preds = np.vstack(preds)
        y_pred = np.argmax(preds, axis=1)
        acc = accuracy_score(y, y_pred)
        return acc
        
    except Exception as e:
        return 0.0
```

### Step-by-Step Breakdown

#### Step 1: Compile

```python
func = self.toolbox.compile(expr=individual)
```

Converts the tree into a callable function.

#### Step 2: Setup Context

```python
gp_context.context.reset(mode)
if mode == gp_context.ExecutionMode.TRAIN:
    gp_context.context.set_train_labels(y)
```

- Reset context for this evaluation
- If training, provide labels (classifiers need them)

#### Step 3: Setup Data

```python
iterator_factory = self._make_image_iterator(X)
input_image = gp_types.Image(iterator_factory)
gp_ops.set_image(input_image)
```

- Create iterator factory for images
- Wrap in Image type
- Set as global input for terminals

#### Step 4: Execute

```python
pipeline = func()  # Create the lazy pipeline

preds = []
for batch in pipeline:  # Iterate to trigger execution
    preds.append(batch.data)
```

The `for` loop pulls data through the entire pipeline:
- Terminals yield image batches
- Filters transform them
- Features are extracted
- Classifiers make predictions

#### Step 5: Compute Accuracy

```python
preds = np.vstack(preds)  # Combine all batches
y_pred = np.argmax(preds, axis=1)  # Get predicted classes
acc = accuracy_score(y, y_pred)  # Compare to true labels
return acc
```

### The `_evaluate_and_store_stats` Method

```python
def _evaluate_and_store_stats(self, ind):
    """
    Runs the full evaluation cycle: Train -> Val -> Test.
    """
    # 1. Train (Fit models)
    train_acc = self._evaluate_pipeline(ind, self.X_train, self.y_train, 
                                        mode=gp_context.ExecutionMode.TRAIN)
    
    # 2. Validation (Fitness)
    val_acc = self._evaluate_pipeline(ind, self.X_val, self.y_val, 
                                      mode=gp_context.ExecutionMode.EVAL)
    
    # 3. Test (Reporting)
    test_acc = self._evaluate_pipeline(ind, self.X_test, self.y_test, 
                                       mode=gp_context.ExecutionMode.EVAL)
    
    # Store stats on the individual
    ind.train_acc = train_acc
    ind.val_acc = val_acc
    ind.test_acc = test_acc
    
    # Set Fitness (Validation Accuracy)
    ind.fitness.values = (val_acc,)
    
    return val_acc
```

### The Three Evaluations

| Phase | Mode | Purpose | Used For |
|-------|------|---------|----------|
| Train | TRAIN | Fit classifiers | Model training |
| Validation | EVAL | Measure generalization | Fitness (selection) |
| Test | EVAL | Final performance | Reporting only |

### Why This Order?

1. **Train first**: Classifiers need to be fitted before predictions
2. **Then eval on val/test**: Use the fitted models

---

## The Evolutionary Loop

### The `run` Method

```python
def run(self):
    print(f"Starting Experiment: {self.experiment_name}")
    
    # 1. Initialize Population
    pop = self.toolbox.population(n=self.pop_size)
    
    # 2. Evaluate Initial Population
    for ind in pop:
        self._evaluate_and_store_stats(ind)
    self.save_generation(pop, 0)
    
    # 3. Evolution Loop
    for g in range(1, self.generations + 1):
        # Select
        offspring = self.toolbox.select(pop, len(pop))
        offspring = list(map(self.toolbox.clone, offspring))
        
        # Mate & Mutate
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < self.cx_prob:
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if np.random.random() < self.mut_prob:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate Invalid
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            self._evaluate_and_store_stats(ind)
        
        pop[:] = offspring
        self.save_generation(pop, g)
    
    self.save_experiment_summary()
```

### Step 1: Initialize Population

```python
pop = self.toolbox.population(n=self.pop_size)
```

Creates `pop_size` random individuals using `gen_safe`.

### Step 2: Evaluate Generation 0

```python
for ind in pop:
    self._evaluate_and_store_stats(ind)
self.save_generation(pop, 0)
```

Every individual gets trained, validated, and tested.

### Step 3: Evolution Loop

#### Selection

```python
offspring = self.toolbox.select(pop, len(pop))
offspring = list(map(self.toolbox.clone, offspring))
```

- **Tournament selection**: Pick 3 random individuals, keep the best
- Repeat until we have `len(pop)` selected individuals
- **Clone**: Copy so we don't modify originals

#### Crossover

```python
for child1, child2 in zip(offspring[::2], offspring[1::2]):
    if np.random.random() < self.cx_prob:
        self.toolbox.mate(child1, child2)
        del child1.fitness.values  # Mark as needing re-evaluation
```

- Pairs adjacent individuals
- With probability `cx_prob`, swap subtrees
- `gp.cxOnePoint`: Pick random points in both trees, swap subtrees

**Visual example**:
```
Before:                      After:
Tree1: A(B, C)              Tree1: A(B, X)
Tree2: D(E, X)              Tree2: D(E, C)
            ^  ^
         Swap these
```

#### Mutation

```python
for mutant in offspring:
    if np.random.random() < self.mut_prob:
        self.toolbox.mutate(mutant)
        del mutant.fitness.values
```

- `gp.mutUniform`: Replace random subtree with new random tree
- Uses `expr_mut` (max depth 2) for new subtrees

#### Evaluate Only Changed Individuals

```python
invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
for ind in invalid_ind:
    self._evaluate_and_store_stats(ind)
```

- `fitness.valid` is `False` when `fitness.values` was deleted
- Only evaluate individuals that changed
- Saves computation!

#### Replace Population

```python
pop[:] = offspring
```

The entire population is replaced by offspring (generational model).

---

## Saving and Logging

### The `save_generation` Method

```python
def save_generation(self, population, gen_num):
    gen_dir = os.path.join(self.base_dir, f"gen_{gen_num}")
    os.makedirs(gen_dir, exist_ok=True)
    
    # For each individual...
    for i, ind in enumerate(population):
        tree_dir = os.path.join(gen_dir, f"tree_{i}")
        os.makedirs(tree_dir, exist_ok=True)
        
        stats = {
            "generation": gen_num,
            "individual_id": i,
            "train_accuracy": getattr(ind, "train_acc", 0.0),
            "validation_accuracy": getattr(ind, "val_acc", 0.0),
            "test_accuracy": getattr(ind, "test_acc", 0.0),
            "depth": ind.height,
            "size": len(ind),
            "expression": str(ind)
        }
        
        # Save JSON
        with open(os.path.join(tree_dir, "results.json"), "w") as f:
            json.dump(stats, f, indent=4)
        
        # Save visualization
        gp_draw.draw_tree(ind, filename=os.path.join(tree_dir, "tree_viz.png"))
```

### What Gets Saved

For each individual:
```
tree_0/
├── results.json      # Performance metrics and tree expression
└── tree_viz.png      # Visual representation of the tree
```

`results.json` example:
```json
{
    "generation": 5,
    "individual_id": 3,
    "train_accuracy": 0.85,
    "validation_accuracy": 0.72,
    "test_accuracy": 0.70,
    "depth": 4,
    "size": 12,
    "expression": "rf_classification(concat_features_2(hog_features(sobel_filter(GetRed)), lbp_features(GetGray)), 200, 30)"
}
```

### Generation Summary

```python
gen_summary = {
    "generation": gen_num,
    "avg_train_accuracy": np.mean(train_accuracies),
    "max_train_accuracy": np.max(train_accuracies),
    # ... similar for val and test ...
    "avg_depth": np.mean(depths),
    "avg_size": np.mean(sizes)
}
```

### The `save_experiment_summary` Method

```python
def save_experiment_summary(self):
    summary = {
        "experiment_name": self.experiment_name,
        "total_generations": self.generations,
        "population_size": self.pop_size,
        "best_individual": self.best_individual_overall,
        "generations_stats": self.all_generations_stats
    }
    
    # Save JSON
    with open(os.path.join(self.base_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    
    # Create progress plot
    plt.figure(figsize=(12, 6))
    # ... plot code ...
    plt.savefig(os.path.join(self.base_dir, "progress_plot.png"))
```

### Progress Plot

Shows accuracy over generations:
- Average and max training accuracy
- Average and max validation accuracy
- Average test accuracy

---

## Complete Workflow Example

### Setting Up and Running an Experiment

```python
from src import gp_setup, gp_experiment
import numpy as np

# 1. Load your data
X_train = np.random.rand(1000, 3, 32, 32)  # 1000 training images
y_train = np.random.randint(0, 10, 1000)    # 10 classes
X_val = np.random.rand(200, 3, 32, 32)
y_val = np.random.randint(0, 10, 200)
X_test = np.random.rand(200, 3, 32, 32)
y_test = np.random.randint(0, 10, 200)

# 2. Create experiment runner
runner = gp_experiment.ExperimentRunner(
    experiment_name="cifar10_test",
    output_dir="experiments"
)

# 3. Setup data
runner.setup_data(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    batch_size=32
)

# 4. Setup GP
pset = gp_setup.create_primitive_set()
runner.setup_gp(
    pset,
    pop_size=20,
    generations=10,
    crossover_prob=0.7,
    mutation_prob=0.3
)

# 5. Run!
runner.run()

# 6. Check results
print(f"Best validation accuracy: {runner.best_fitness_overall:.2%}")
print(f"Best tree: {runner.best_individual_overall['expression']}")
```

### What Happens During `run()`

```
Starting Experiment: cifar10_test
Population: 20, Generations: 10

Evaluating Generation 0...
  [20 trees evaluated]
Saving Generation 0 to experiments/cifar10_test/gen_0...

--- Generation 1 ---
Evaluating 18 individuals...  # 2 unchanged from selection
Saving Generation 1...

--- Generation 2 ---
Evaluating 15 individuals...
Saving Generation 2...

... (continues) ...

--- Generation 10 ---
Evaluating 16 individuals...
Saving Generation 10...

Saving Experiment Summary...
Experiment Completed.
```

### Output Files

```
experiments/
└── cifar10_test/
    ├── gen_0/
    │   ├── generation_summary.json
    │   ├── tree_0/
    │   │   ├── results.json
    │   │   └── tree_viz.png
    │   ├── tree_1/
    │   │   └── ...
    │   └── ... (18 more)
    ├── gen_1/
    │   └── ...
    ├── ... (8 more generations)
    ├── gen_10/
    │   └── ...
    ├── experiment_summary.json
    └── progress_plot.png
```

---

## Complete Code Walkthrough

```python
import os
import json
import time
import numpy as np
from deap import base, creator, tools, gp, algorithms
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import shutil

from src import gp_setup, gp_ops, gp_types, gp_context, gp_draw, gp_utils
```

**Lines 1-10**: Import standard library, DEAP, sklearn, matplotlib, and our modules.

---

```python
class ExperimentRunner:
    def __init__(self, experiment_name, output_dir="experiments"):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.base_dir = os.path.join(output_dir, experiment_name)
        
        if os.path.exists(self.base_dir):
            print(f"Warning: Experiment directory {self.base_dir} exists.")
        else:
            os.makedirs(self.base_dir)
```

**Lines 12-22**: Constructor creates experiment directory.

---

```python
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.batch_size = 32
        self.pset = None
        self.toolbox = None
        
        self.all_generations_stats = []
        self.best_individual_overall = None
        self.best_fitness_overall = -1.0
```

**Lines 24-36**: Initialize all instance attributes to sensible defaults.

---

```python
    def setup_data(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size
```

**Lines 38-46**: Simple setters for data.

---

```python
    def _make_image_iterator(self, X):
        batch_size = self.batch_size
        def iterator():
            n_samples = len(X)
            for i in range(0, n_samples, batch_size):
                batch_data = X[i:i+batch_size]
                r = batch_data[:, 0, :, :]
                g = batch_data[:, 1, :, :]
                b = batch_data[:, 2, :, :]
                yield gp_types.Batch([r, g, b])
        return iterator
```

**Lines 48-59**: Creates a factory function for image batches. Note it returns `iterator` (the function), not `iterator()` (the generator).

---

```python
    def setup_gp(self, pset, pop_size=20, generations=10, crossover_prob=0.5, mutation_prob=0.2):
        # ... store parameters ...
        
        if hasattr(creator, "FitnessMax"): del creator.FitnessMax
        if hasattr(creator, "Individual"): del creator.Individual
            
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp_utils.gen_safe, pset=pset, min_=2, max_=5, type_=gp_types.Prediction)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("compile", self._compile_lazy)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp_utils.gen_safe, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=pset)
        
        self.toolbox.decorate("mate", gp.staticLimit(key=lambda ind: ind.height, max_value=10))
        self.toolbox.decorate("mutate", gp.staticLimit(key=lambda ind: ind.height, max_value=10))
```

**Lines 61-92**: Configure DEAP's creator and toolbox.

---

```python
    def _compile_lazy(self, expr):
        code = str(expr)
        code = f"lambda: {code}"
        return eval(code, self.pset.context, {})
```

**Lines 94-97**: Convert tree to callable function.

---

```python
    def _evaluate_pipeline(self, individual, X, y, mode=gp_context.ExecutionMode.TRAIN):
        try:
            func = self.toolbox.compile(expr=individual)
            
            gp_context.context.reset(mode)
            if mode == gp_context.ExecutionMode.TRAIN:
                gp_context.context.set_train_labels(y)
            
            iterator_factory = self._make_image_iterator(X)
            input_image = gp_types.Image(iterator_factory)
            gp_ops.set_image(input_image)
            
            pipeline = func()
            
            preds = []
            for batch in pipeline:
                preds.append(batch.data)
                
            if not preds:
                return 0.0
                
            preds = np.vstack(preds)
            y_pred = np.argmax(preds, axis=1)
            acc = accuracy_score(y, y_pred)
            return acc
            
        except Exception as e:
            return 0.0
```

**Lines 99-127**: The core evaluation logic.

---

```python
    def _evaluate_and_store_stats(self, ind):
        train_acc = self._evaluate_pipeline(ind, self.X_train, self.y_train, mode=gp_context.ExecutionMode.TRAIN)
        val_acc = self._evaluate_pipeline(ind, self.X_val, self.y_val, mode=gp_context.ExecutionMode.EVAL)
        test_acc = self._evaluate_pipeline(ind, self.X_test, self.y_test, mode=gp_context.ExecutionMode.EVAL)
        
        ind.train_acc = train_acc
        ind.val_acc = val_acc
        ind.test_acc = test_acc
        
        ind.fitness.values = (val_acc,)
        
        return val_acc
```

**Lines 129-157**: Evaluate on all three sets, store results, set fitness.

---

```python
    def run(self):
        print(f"Starting Experiment: {self.experiment_name}")
        
        pop = self.toolbox.population(n=self.pop_size)
        
        for ind in pop:
            self._evaluate_and_store_stats(ind)
        self.save_generation(pop, 0)
        
        for g in range(1, self.generations + 1):
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Crossover and mutation...
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid_ind:
                self._evaluate_and_store_stats(ind)
            
            pop[:] = offspring
            self.save_generation(pop, g)
        
        self.save_experiment_summary()
```

**Lines 306-360**: The main evolutionary loop.

---

## Summary

The `gp_experiment.py` module is the **orchestrator** of the GP system:

| Component | Purpose |
|-----------|---------|
| `ExperimentRunner` | Main class managing the experiment |
| `setup_data` | Configure training/validation/test data |
| `setup_gp` | Configure DEAP evolutionary operators |
| `_evaluate_pipeline` | Run one tree on data, get accuracy |
| `run` | Execute the evolutionary loop |
| `save_generation` | Save results for each generation |
| `save_experiment_summary` | Create final report and plots |

The module brings together all components to evolve image classification pipelines, track their performance, and save detailed results for analysis.
