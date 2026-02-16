# GP Context Module (`gp_context.py`) - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [Why Do We Need a Context?](#why-do-we-need-a-context)
3. [The ExecutionMode Enum](#the-executionmode-enum)
4. [The GPContext Class](#the-gpcontext-class)
5. [Method-by-Method Breakdown](#method-by-method-breakdown)
6. [The Global Context Instance](#the-global-context-instance)
7. [Complete Workflow Examples](#complete-workflow-examples)
8. [State Management Deep Dive](#state-management-deep-dive)
9. [Common Pitfalls and How to Avoid Them](#common-pitfalls-and-how-to-avoid-them)

---

## Overview

The `gp_context.py` module provides a **global execution context** for the Genetic Programming system. It manages:

1. **Execution Mode**: Whether we're training (fitting models) or evaluating (using fitted models)
2. **Model Storage**: A registry of trained classifiers keyed by node ID
3. **Node Counter**: Unique identifiers for classifier nodes in the GP tree
4. **Training Labels**: Ground truth labels needed during training

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTEXT STATE DIAGRAM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   context.reset(TRAIN)                                           │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │ mode = TRAIN                                              │  │
│   │ models = {}  (cleared!)                                   │  │
│   │ node_counter = 0                                          │  │
│   │ train_labels = None                                       │  │
│   └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│   context.set_train_labels(y)                                    │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │ train_labels = [0, 1, 0, 3, 2, ...]                       │  │
│   └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼ (During tree execution)           │
│   context.get_next_node_id()  →  "node_0"                        │
│   context.save_model("node_0", RandomForest(...))                │
│   context.get_next_node_id()  →  "node_1"                        │
│   context.save_model("node_1", SVM(...))                         │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │ models = {"node_0": RF, "node_1": SVM}                    │  │
│   │ node_counter = 2                                          │  │
│   └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│   context.reset(EVAL)                                            │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │ mode = EVAL                                               │  │
│   │ models = {"node_0": RF, "node_1": SVM}  (PRESERVED!)      │  │
│   │ node_counter = 0  (reset for consistent lookup)           │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why Do We Need a Context?

### The Problem: Stateful Classification Nodes

Consider a GP tree that uses Random Forest classification:

```
rf_classification(hog_features(GetGray()), trees=100, depth=50)
```

This tree contains a **Random Forest model** that needs to be:
1. **Trained** on the training data (fit the model)
2. **Saved** so we can use it later
3. **Retrieved** when evaluating on validation/test data

Without a context, where would we store this trained model?

### The Problem: Multiple Classifier Nodes

A tree might have multiple classifiers:

```
sum_prediction_2(
    rf_classification(hog_features(GetRed()), 100, 50),
    lr_classification(lbp_features(GetGray()))
)
```

Each classifier needs its own storage. We need a way to:
1. Give each classifier a unique ID
2. Store/retrieve each model by that ID

### The Solution: A Global Context

The `GPContext` class provides:
- A dictionary (`models`) to store classifiers by ID
- A counter (`node_counter`) to generate unique IDs
- A mode flag to switch between TRAIN and EVAL behavior
- Storage for training labels (needed by classifiers during training)

---

## The ExecutionMode Enum

```python
from enum import Enum

class ExecutionMode(Enum):
    TRAIN = "train"
    EVAL = "eval"
```

### What Is an Enum?

An `Enum` (enumeration) is a way to define a set of named constants. Instead of using strings like `"train"` or `"eval"` directly (which are error-prone), we use `ExecutionMode.TRAIN` and `ExecutionMode.EVAL`.

### Why Use an Enum?

```python
# BAD: String literals are error-prone
if mode == "trian":  # Typo! Silent bug.
    train_model()

# GOOD: Enum catches typos at definition time
if mode == ExecutionMode.TRIAN:  # Error! "TRIAN" doesn't exist.
    train_model()
```

### The Two Modes

| Mode | Description | When Used |
|------|-------------|-----------|
| `TRAIN` | Models are fitted, cross-validation is run, models are saved | First evaluation pass on training data |
| `EVAL` | Models are loaded from storage, only predictions are made | Validation and test evaluation |

### How Modes Affect Behavior

```python
# In rf_classification (gp_ops.py):
if context.mode == ExecutionMode.TRAIN:
    # 1. Collect all features from the batch iterator
    # 2. Run cross-validation (for more realistic probability estimates)
    # 3. Fit the model on all training data
    # 4. Save the model to context.models
    # 5. Yield predictions
else:  # EVAL mode
    # 1. Load the model from context.models
    # 2. Predict on each batch
    # 3. Yield predictions
```

---

## The GPContext Class

```python
class GPContext:
    def __init__(self):
        self.mode = ExecutionMode.TRAIN
        self.models: Dict[str, Any] = {}
        self.node_counter = 0
        self.train_labels: Optional[np.ndarray] = None
        self.batch_size = 32
```

### Instance Attributes

| Attribute | Type | Default | Purpose |
|-----------|------|---------|---------|
| `mode` | `ExecutionMode` | `TRAIN` | Current execution mode |
| `models` | `Dict[str, Any]` | `{}` | Trained models keyed by node ID |
| `node_counter` | `int` | `0` | Counter for generating unique node IDs |
| `train_labels` | `Optional[np.ndarray]` | `None` | Ground truth labels for training |
| `batch_size` | `int` | `32` | Batch size (currently unused) |

### The Models Dictionary

The `models` dictionary stores trained scikit-learn classifiers:

```python
{
    "node_0": <RandomForestClassifier object>,
    "node_1": <LogisticRegression object>,
    "node_2": <SVC object>,
    ...
}
```

### Why String Keys?

We use string keys like `"node_0"` instead of integers for:
1. **Debugging**: Easier to read in logs and error messages
2. **Flexibility**: Could include additional info in the future (e.g., `"rf_node_0"`)
3. **JSON compatibility**: Strings serialize cleanly to JSON

---

## Method-by-Method Breakdown

### `reset(mode)`

```python
def reset(self, mode: ExecutionMode = ExecutionMode.TRAIN):
    self.mode = mode
    self.node_counter = 0
    # We do NOT clear models if switching to EVAL
    if mode == ExecutionMode.TRAIN:
        self.models = {}
```

#### Purpose

Prepares the context for a new evaluation pass.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `ExecutionMode` | `TRAIN` | The mode to switch to |

#### Behavior

**When `mode == TRAIN`:**
- Sets mode to TRAIN
- Resets `node_counter` to 0
- **Clears the models dictionary** (fresh start!)

**When `mode == EVAL`:**
- Sets mode to EVAL
- Resets `node_counter` to 0
- **Keeps the models dictionary intact** (we need those trained models!)

#### Why Reset the Counter?

The node counter must reset to 0 for both modes because classifier nodes generate their IDs in order of execution. If we don't reset:

```python
# TRAIN: nodes get IDs 0, 1, 2
# EVAL without reset: nodes get IDs 3, 4, 5
# Models saved under 0, 1, 2 would never be found!
```

#### Example

```python
# Training phase
context.reset(ExecutionMode.TRAIN)
context.set_train_labels(y_train)
# ... run tree on training data ...
# models = {"node_0": RF, "node_1": LR}

# Validation phase
context.reset(ExecutionMode.EVAL)
# mode = EVAL, node_counter = 0
# models = {"node_0": RF, "node_1": LR}  # Still here!
# ... run same tree on validation data ...
# node_0 correctly retrieves RF, node_1 correctly retrieves LR
```

---

### `get_next_node_id()`

```python
def get_next_node_id(self) -> str:
    nid = f"node_{self.node_counter}"
    self.node_counter += 1
    return nid
```

#### Purpose

Generates a unique identifier for a classifier node.

#### Returns

A string like `"node_0"`, `"node_1"`, etc.

#### Behavior

1. Creates string from current counter value
2. Increments counter
3. Returns the string

#### Critical Insight: Order Matters!

The node ID system relies on **deterministic execution order**. Every time the tree executes:
1. The first classifier to run gets `"node_0"`
2. The second gets `"node_1"`
3. And so on...

If execution order changed between TRAIN and EVAL, models would be mismatched. Fortunately, our lazy evaluation ensures consistent order.

#### Example

```python
context.reset(ExecutionMode.TRAIN)

id1 = context.get_next_node_id()  # "node_0"
id2 = context.get_next_node_id()  # "node_1"
id3 = context.get_next_node_id()  # "node_2"

context.reset(ExecutionMode.EVAL)

id1 = context.get_next_node_id()  # "node_0" again!
id2 = context.get_next_node_id()  # "node_1" again!
id3 = context.get_next_node_id()  # "node_2" again!
```

---

### `set_train_labels(labels)`

```python
def set_train_labels(self, labels: np.ndarray):
    self.train_labels = labels
```

#### Purpose

Stores the ground truth labels needed for training classifiers.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `labels` | `np.ndarray` | 1D array of class labels (e.g., `[0, 1, 0, 2, 1, ...]`) |

#### When Is This Called?

Only before a TRAIN pass:

```python
context.reset(ExecutionMode.TRAIN)
context.set_train_labels(y_train)  # Set labels for this training run
```

#### Why Store Labels Globally?

Classifiers like Random Forest need access to labels during fitting:

```python
# In rf_classification:
clf = RandomForestClassifier(...)
clf.fit(X, context.train_labels)  # Access via context
```

Alternative designs (like passing labels through the tree) would require changing the type system.

---

### `get_model(node_id)`

```python
def get_model(self, node_id: str) -> Any:
    return self.models.get(node_id)
```

#### Purpose

Retrieves a previously saved model by its node ID.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `node_id` | `str` | The unique identifier (e.g., `"node_0"`) |

#### Returns

- The saved model if found
- `None` if not found (using `dict.get()` instead of `dict[]`)

#### Example

```python
# During EVAL mode:
clf = context.get_model("node_0")
if clf:
    predictions = clf.predict_proba(X)
else:
    print("Model not found!")
```

---

### `save_model(node_id, model)`

```python
def save_model(self, node_id: str, model: Any):
    self.models[node_id] = model
```

#### Purpose

Stores a trained model for later retrieval.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `node_id` | `str` | The unique identifier |
| `model` | `Any` | The trained scikit-learn classifier |

#### Example

```python
# During TRAIN mode:
clf = RandomForestClassifier(n_estimators=100, max_depth=50)
clf.fit(X_train, y_train)
context.save_model("node_0", clf)
```

---

## The Global Context Instance

```python
# Global instance
context = GPContext()
```

### What Is a Global Instance?

The module creates a single `GPContext` object that all other modules import and share:

```python
# In gp_ops.py:
from .gp_context import context, ExecutionMode

def rf_classification(features, t, d):
    node_id = context.get_next_node_id()  # Uses global context
    # ...
```

### Why Use a Global Instance?

1. **Simplicity**: All operations automatically share state
2. **Implicit context**: No need to pass context through every function
3. **Singleton pattern**: Guarantees exactly one context exists

### Alternative: Explicit Context Passing

We could have designed the system to pass context explicitly:

```python
# Alternative (not used):
def rf_classification(features, t, d, context):
    node_id = context.get_next_node_id()
    # ...
```

But this would:
- Complicate the GP type system (context would need a type)
- Make tree expressions longer and harder to read
- Require changes throughout the codebase

---

## Complete Workflow Examples

### Example 1: Single Classifier Tree

Let's trace through a simple tree:

```python
# Tree: rf_classification(hog_features(GetRed()), 100, 50)

# === TRAINING PHASE ===

# 1. Reset context for training
context.reset(ExecutionMode.TRAIN)
# State: mode=TRAIN, models={}, counter=0

# 2. Set labels
context.set_train_labels(np.array([0, 1, 0, 2, 1, 0, ...]))
# State: mode=TRAIN, models={}, counter=0, labels=[0, 1, ...]

# 3. Execute tree (internally):
#    - GetRed yields batches
#    - hog_features processes batches
#    - rf_classification:
#      a. Calls context.get_next_node_id() → "node_0"
#      b. Collects all feature batches
#      c. Trains RandomForest on features + context.train_labels
#      d. Calls context.save_model("node_0", rf_model)
#      e. Yields predictions

# State: mode=TRAIN, models={"node_0": RF}, counter=1

# === VALIDATION PHASE ===

# 4. Reset context for eval (models preserved!)
context.reset(ExecutionMode.EVAL)
# State: mode=EVAL, models={"node_0": RF}, counter=0

# 5. Execute same tree on validation data:
#    - GetRed yields validation batches
#    - hog_features processes batches
#    - rf_classification:
#      a. Calls context.get_next_node_id() → "node_0" (counter reset!)
#      b. Calls context.get_model("node_0") → RF (found!)
#      c. RF.predict_proba(features)
#      d. Yields predictions

# State: mode=EVAL, models={"node_0": RF}, counter=1
```

### Example 2: Ensemble Tree (Multiple Classifiers)

```python
# Tree: sum_prediction_2(
#     rf_classification(hog_features(GetRed()), 100, 50),
#     lr_classification(lbp_features(GetGray()))
# )

# === TRAINING PHASE ===

context.reset(ExecutionMode.TRAIN)
context.set_train_labels(y_train)

# Execution order:
# 1. rf_classification runs first (left branch)
#    - Gets ID "node_0"
#    - Trains and saves RF
# 2. lr_classification runs second (right branch)
#    - Gets ID "node_1"
#    - Trains and saves LR
# 3. sum_prediction_2 adds the predictions

# State: models={"node_0": RF, "node_1": LR}, counter=2

# === EVAL PHASE ===

context.reset(ExecutionMode.EVAL)

# Execution order (SAME as training):
# 1. rf_classification runs first
#    - Gets ID "node_0" (counter was reset!)
#    - Retrieves RF from models["node_0"]
# 2. lr_classification runs second
#    - Gets ID "node_1"
#    - Retrieves LR from models["node_1"]
# 3. sum_prediction_2 adds the predictions
```

### Example 3: Cascade Classifier (Nested Classifiers)

```python
# Tree: rf_classification(
#     cascade_lr(hog_features(GetGray())),
#     100, 50
# )

# This tree has TWO classifiers:
# 1. LR in cascade_lr (generates features)
# 2. RF in rf_classification (final prediction)

# === TRAINING ===

# Execution order:
# 1. cascade_lr processes first
#    a. Extracts HOG features
#    b. lr_classification internally gets ID "node_0"
#    c. Trains LR, saves to models["node_0"]
#    d. Concatenates features with LR predictions
# 2. rf_classification processes second
#    a. Gets ID "node_1"
#    b. Trains RF on (features + LR_probs)
#    c. Saves to models["node_1"]

# State: models={"node_0": LR, "node_1": RF}

# === EVAL ===

# Same order, IDs match, models retrieved correctly.
```

---

## State Management Deep Dive

### Why Global State Can Be Dangerous

Global state is generally considered an anti-pattern because:
1. **Hidden dependencies**: Functions depend on state set elsewhere
2. **Testing difficulty**: Hard to isolate tests
3. **Concurrency issues**: Multiple threads could corrupt shared state

### Why It Works Here

Our use case has specific properties that make global state acceptable:

1. **Single-threaded execution**: We process one tree at a time
2. **Clear lifecycle**: `reset()` → `set_labels()` → execute → repeat
3. **Deterministic order**: Lazy evaluation guarantees consistent node ordering
4. **Limited scope**: Only classifier nodes access the context

### The Reset/Execute/Reset Pattern

Every evaluation follows this pattern:

```python
# Pattern for each evaluation:
context.reset(mode)
if mode == TRAIN:
    context.set_train_labels(labels)
# Set up image data (separate from context)
# Execute tree
# Collect predictions
```

This clear pattern prevents state leakage between evaluations.

---

## Common Pitfalls and How to Avoid Them

### Pitfall 1: Forgetting to Reset Before Training

```python
# WRONG:
context.set_train_labels(y_train)
# Old models still in context! Tree might use stale models.
run_tree(...)

# RIGHT:
context.reset(ExecutionMode.TRAIN)  # Clears old models
context.set_train_labels(y_train)
run_tree(...)
```

### Pitfall 2: Resetting to TRAIN Before Validation

```python
# WRONG:
# After training...
context.reset(ExecutionMode.TRAIN)  # Oops! Cleared the models!
run_tree_on_validation(...)  # Models not found!

# RIGHT:
context.reset(ExecutionMode.EVAL)  # Preserves models
run_tree_on_validation(...)  # Works!
```

### Pitfall 3: Non-Deterministic Tree Execution

```python
# WRONG: Using randomness in tree execution
def my_random_feature(channel):
    if random.random() > 0.5:
        return hog_features(channel)
    else:
        return lbp_features(channel)

# Different path each time → different node order → ID mismatch!

# RIGHT: All randomness should be fixed at tree generation time
# (handled by DEAP's ephemeral constants)
```

### Pitfall 4: Parallel Execution

```python
# WRONG: Running multiple trees in parallel
from multiprocessing import Pool

pool = Pool(4)
pool.map(evaluate_tree, trees)  # Each process shares context???

# All processes would overwrite each other's models!

# RIGHT: Single-threaded execution (current design)
# OR: Create separate context per process
```

---

## Complete Code with Annotations

```python
from enum import Enum
from typing import Any, Dict, Optional, Iterator
import numpy as np

class ExecutionMode(Enum):
    """
    Enum defining the two execution modes for the GP system.
    
    TRAIN: Models are being fitted. Classifiers run cross-validation,
           then fit on all data, and save themselves to context.
    
    EVAL: Models are already fitted. Classifiers load themselves
          from context and only run predictions.
    """
    TRAIN = "train"  # String value for debugging/logging
    EVAL = "eval"


class GPContext:
    """
    Global execution context for the Genetic Programming system.
    
    This class manages state that needs to be shared across all nodes
    in a GP tree during execution. It handles:
    
    1. Execution mode (training vs evaluation)
    2. Model storage and retrieval
    3. Unique node ID generation
    4. Training label access
    
    USAGE:
        # Training
        context.reset(ExecutionMode.TRAIN)
        context.set_train_labels(y_train)
        # ... execute tree ...
        
        # Evaluation
        context.reset(ExecutionMode.EVAL)
        # ... execute same tree ...
    """
    
    def __init__(self):
        """Initialize a fresh context with default values."""
        self.mode = ExecutionMode.TRAIN      # Start in training mode
        self.models: Dict[str, Any] = {}     # Empty model registry
        self.node_counter = 0                # First node will be "node_0"
        self.train_labels: Optional[np.ndarray] = None  # No labels yet
        self.batch_size = 32                 # Default batch size (informational)
        
    def reset(self, mode: ExecutionMode = ExecutionMode.TRAIN):
        """
        Reset the context for a new evaluation pass.
        
        CRITICAL: This MUST be called before each tree evaluation to ensure
        node IDs are assigned consistently.
        
        If mode is TRAIN:
            - Clears all saved models (fresh start)
            - Resets node counter to 0
            
        If mode is EVAL:
            - PRESERVES saved models (we need them!)
            - Resets node counter to 0 (for consistent ID lookup)
        
        Parameters:
            mode: The execution mode for the upcoming evaluation
        """
        self.mode = mode
        self.node_counter = 0  # Always reset for consistent ID generation
        
        # Only clear models when starting fresh training
        if mode == ExecutionMode.TRAIN:
            self.models = {}
            
    def get_next_node_id(self) -> str:
        """
        Generate the next unique node identifier.
        
        Called by classifier nodes (rf_classification, lr_classification, etc.)
        to obtain a unique key for storing/retrieving their trained models.
        
        Returns:
            A string like "node_0", "node_1", etc.
            
        IMPORTANT: Node IDs are assigned in execution order. As long as
        tree execution is deterministic (which it is with lazy evaluation),
        the same classifier node will get the same ID across TRAIN and EVAL.
        """
        nid = f"node_{self.node_counter}"
        self.node_counter += 1
        return nid
        
    def set_train_labels(self, labels: np.ndarray):
        """
        Store training labels for use by classifier nodes.
        
        This should be called after reset(TRAIN) and before tree execution.
        Classifier nodes access these labels during fitting:
            clf.fit(X, context.train_labels)
        
        Parameters:
            labels: 1D numpy array of shape (n_samples,) containing
                   integer class labels (0, 1, 2, ...)
        """
        self.train_labels = labels

    def get_model(self, node_id: str) -> Any:
        """
        Retrieve a trained model by its node ID.
        
        Called by classifier nodes during EVAL mode to get their
        previously trained model.
        
        Parameters:
            node_id: The unique identifier (e.g., "node_0")
            
        Returns:
            The trained model if found, None otherwise.
            
        NOTE: Returns None rather than raising KeyError for graceful
        handling of edge cases.
        """
        return self.models.get(node_id)

    def save_model(self, node_id: str, model: Any):
        """
        Store a trained model for later retrieval.
        
        Called by classifier nodes during TRAIN mode after fitting.
        
        Parameters:
            node_id: The unique identifier for this node
            model: A fitted scikit-learn classifier (RandomForest,
                   LogisticRegression, SVC, etc.)
        """
        self.models[node_id] = model


# Global instance - imported by all other modules
# This is the single source of truth for execution state
context = GPContext()
```

---

## Summary

The `gp_context.py` module provides essential infrastructure for the GP system:

| Component | Purpose |
|-----------|---------|
| `ExecutionMode` | Distinguishes training from evaluation |
| `GPContext` | Manages shared state across tree nodes |
| `context` (global) | Single instance used by all modules |

Key methods:
- `reset()`: Prepare for new evaluation, optionally clearing models
- `get_next_node_id()`: Generate unique identifiers
- `set_train_labels()`: Store labels for training
- `save_model()` / `get_model()`: Store and retrieve trained classifiers

The context is the "glue" that allows classifier nodes to maintain state across the training and evaluation phases of GP tree execution.
