# GP Types Module (`gp_types.py`) - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [Why Custom Types?](#why-custom-types)
3. [The Batch Class](#the-batch-class)
4. [The DataWrapper Base Class](#the-datawrapper-base-class)
5. [Data Flow Types](#data-flow-types)
   - [Image](#image)
   - [Channel](#channel)
   - [FeatureVector](#featurevector)
   - [Prediction](#prediction)
6. [Terminal (Parameter) Types](#terminal-parameter-types)
7. [How DEAP Uses These Types](#how-deap-uses-these-types)
8. [The Lazy Evaluation Pattern](#the-lazy-evaluation-pattern)
9. [Complete Code Walkthrough](#complete-code-walkthrough)

---

## Overview

The `gp_types.py` module defines the **type system** for the entire Genetic Programming (GP) framework. In Genetic Programming, we evolve tree-structured programs where each node has a specific input/output type signature. This module defines all those types.

Think of this module as the "grammar" of the programming language we're evolving. Just like Python has types like `int`, `str`, `list`, etc., our GP system has types like `Image`, `Channel`, `FeatureVector`, and `Prediction`.

```
┌─────────────────────────────────────────────────────────────────┐
│                    TYPE HIERARCHY OVERVIEW                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Batch (Simple data container)                                  │
│     └─ Holds: numpy array                                        │
│                                                                  │
│   DataWrapper (Abstract base for lazy data streams)              │
│     ├─ Image      → Stream of (N, C, H, W) batches              │
│     ├─ Channel    → Stream of (N, H, W) batches                  │
│     ├─ FeatureVector → Stream of (N, Features) batches           │
│     └─ Prediction → Stream of (N, Classes) batches               │
│                                                                  │
│   Terminal Types (Simple marker classes for parameters)          │
│     ├─ Weight     → Float weight (0.0 - 1.0)                    │
│     ├─ Trees      → Number of trees in RF (50-1000)              │
│     ├─ Depth      → Max tree depth (10-100)                      │
│     ├─ Frequency  → Gabor filter frequency                       │
│     ├─ Theta      → Angle parameter                              │
│     ├─ Sigma      → Gaussian sigma (1-3)                         │
│     └─ Order      → Derivative order (0-2)                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why Custom Types?

### The Problem with Raw Python Types

In DEAP (Distributed Evolutionary Algorithms in Python), when creating **Strongly-Typed Genetic Programming (STGP)**, we need to tell the system:

1. What types each function accepts as input
2. What type each function returns

If we just used raw numpy arrays everywhere, DEAP couldn't know that a `sobel_filter` should only connect to a `Channel`, not to a `Prediction`. We'd get nonsensical trees like:

```
# BAD: This makes no sense!
random_forest(sobel_filter(logistic_regression_output))
```

### The Solution: Custom Type Classes

By creating distinct type classes, we enable DEAP's type checker to enforce valid connections:

```python
# GOOD: Types enforce valid data flow
# sobel_filter: Channel -> Channel
# hog_features: Channel -> FeatureVector  
# rf_classification: FeatureVector -> Prediction

rf_classification(
    hog_features(
        sobel_filter(
            GetRed()  # Returns Channel
        )           # Returns Channel
    )               # Returns FeatureVector
)                   # Returns Prediction
```

---

## The Batch Class

```python
class Batch:
    def __init__(self, data: np.ndarray):
        self.data = data
```

### What Is It?

`Batch` is the simplest class in the module. It's just a **wrapper around a numpy array**. Think of it as a labeled box that says "this is a batch of data."

### Why Not Just Use numpy Arrays?

We wrap arrays in `Batch` objects for several reasons:

1. **Explicit semantics**: When you see `Batch`, you know it's a chunk of data flowing through the pipeline
2. **Future extensibility**: We could add metadata (like batch index, timestamp, etc.)
3. **Consistent interface**: All data flows through the same container type

### The `data` Attribute

The `data` attribute holds the actual numpy array. Its shape depends on where in the pipeline we are:

| Context | Shape | Example |
|---------|-------|---------|
| Image input | `(N, C, H, W)` | `(32, 3, 32, 32)` for 32 RGB images of 32x32 pixels |
| Single channel | `(N, H, W)` | `(32, 32, 32)` for 32 grayscale images |
| Feature vectors | `(N, F)` | `(32, 1024)` for 32 samples with 1024 features each |
| Predictions | `(N, K)` | `(32, 10)` for 32 samples with 10 class probabilities |

### Example Usage

```python
import numpy as np
from gp_types import Batch

# Create a batch of 16 grayscale images (32x32 pixels)
image_data = np.random.rand(16, 32, 32)
batch = Batch(image_data)

# Access the data
print(batch.data.shape)  # (16, 32, 32)
print(batch.data.dtype)  # float64
```

---

## The DataWrapper Base Class

```python
class DataWrapper:
    """
    Base class for all data types flowing through the GP tree.
    Supports lazy iteration over batches.
    """
    def __init__(self, iterator_factory: Callable[[], Iterator[Batch]]):
        self._iterator_factory = iterator_factory

    def __iter__(self) -> Iterator[Batch]:
        return self._iterator_factory()
```

### What Is It?

`DataWrapper` is the **abstract base class** for all data-flow types in the system. It implements the **lazy evaluation pattern** (explained in detail below).

### The Constructor

```python
def __init__(self, iterator_factory: Callable[[], Iterator[Batch]]):
    self._iterator_factory = iterator_factory
```

The constructor takes a single argument: `iterator_factory`. This is **not** an iterator itself, but a **function that creates an iterator**. This distinction is crucial!

#### Why a Factory Function?

Consider this scenario:

```python
# If we stored an iterator directly...
my_data = SomeDataWrapper(iter([batch1, batch2, batch3]))

# First iteration works fine
for batch in my_data:
    process(batch)

# Second iteration gives NOTHING! Iterator is exhausted!
for batch in my_data:
    process(batch)  # Never runs!
```

With a factory function:

```python
# Store a FUNCTION that creates a fresh iterator
def make_iterator():
    return iter([batch1, batch2, batch3])

my_data = SomeDataWrapper(make_iterator)

# First iteration
for batch in my_data:  # Calls make_iterator() internally
    process(batch)

# Second iteration works! Fresh iterator created
for batch in my_data:  # Calls make_iterator() again
    process(batch)
```

### The `__iter__` Method

```python
def __iter__(self) -> Iterator[Batch]:
    return self._iterator_factory()
```

This implements Python's iterator protocol. When you write:

```python
for batch in data_wrapper:
    ...
```

Python calls `data_wrapper.__iter__()`, which calls the factory function to create a fresh iterator.

---

## Data Flow Types

These are the four types that represent data flowing through the GP tree. They all inherit from `DataWrapper` without adding any new functionality—they exist purely for **type distinction**.

### Image

```python
class Image(DataWrapper):
    """
    Represents a stream of image batches.
    Each batch.data is List[np.ndarray] (channels) or np.ndarray (N, C, H, W).
    """
    pass
```

#### Purpose

`Image` represents the **input to the entire pipeline**. It's the raw image data before any processing.

#### Data Format

Each batch can be in two formats:

1. **List format**: `[red_channel, green_channel, blue_channel]` where each is `(N, H, W)`
2. **Tensor format**: `(N, C, H, W)` where C=3 for RGB

#### Where It's Used

- Created by `ExperimentRunner._make_image_iterator()`
- Consumed by `gp_ops.set_image()` to provide input to the pipeline
- Never appears as an intermediate type—only at the very beginning

#### Example

```python
def make_image_iterator():
    for i in range(0, len(X), batch_size):
        batch_data = X[i:i+batch_size]  # Shape: (B, 3, H, W)
        r = batch_data[:, 0, :, :]  # Red channel
        g = batch_data[:, 1, :, :]  # Green channel  
        b = batch_data[:, 2, :, :]  # Blue channel
        yield Batch([r, g, b])

image = Image(make_image_iterator)
```

---

### Channel

```python
class Channel(DataWrapper):
    """
    Represents a stream of single channel batches.
    Each batch.data is (N, H, W).
    """
    pass
```

#### Purpose

`Channel` represents a **single-channel (grayscale) image stream**. This is the main type used for image filtering operations.

#### Data Format

Each batch has shape `(N, H, W)`:
- `N`: Number of images in the batch
- `H`: Height in pixels
- `W`: Width in pixels

#### Operations That Produce Channels

| Operation | Input | Output | Description |
|-----------|-------|--------|-------------|
| `GetRed` | (terminal) | Channel | Extracts red channel from Image |
| `GetGreen` | (terminal) | Channel | Extracts green channel |
| `GetBlue` | (terminal) | Channel | Extracts blue channel |
| `GetGray` | (terminal) | Channel | Computes grayscale |
| `sobel_filter` | Channel | Channel | Edge detection |
| `gaussian_filter` | Channel, Sigma | Channel | Blur/smooth |
| `linear_combination` | 3×Channel, 3×Weight | Channel | Weighted sum |

#### Example

```python
# A channel that applies Sobel edge detection
def make_sobel_channel():
    for batch in input_channel:
        sobel_result = apply_sobel(batch.data)
        yield Batch(sobel_result)

sobel_channel = Channel(make_sobel_channel)
```

---

### FeatureVector

```python
class FeatureVector(DataWrapper):
    """
    Represents a stream of feature vector batches.
    Each batch.data is (N, Features).
    """
    pass
```

#### Purpose

`FeatureVector` represents **extracted features** from images. These are 1D vectors suitable for machine learning classifiers.

#### Data Format

Each batch has shape `(N, F)`:
- `N`: Number of samples in the batch
- `F`: Number of features (varies by extraction method)

#### Common Feature Dimensions

| Extraction Method | Typical Dimensions | Notes |
|-------------------|-------------------|-------|
| `histogram_features` | 256 | 256 bins for pixel intensity histogram |
| `hog_features` | Varies | Depends on image size and cell/block params |
| `lbp_features` | ~12 | Histogram of LBP patterns |
| `sift_features` | 128 | Mean of dense SIFT descriptors |
| Flattened image | H×W | For a 32×32 image: 1024 features |

#### Operations That Produce FeatureVectors

| Operation | Input | Output |
|-----------|-------|--------|
| `histogram_features` | Channel | FeatureVector |
| `hog_features` | Channel | FeatureVector |
| `lbp_features` | Channel | FeatureVector |
| `sobel_features` | Channel | FeatureVector |
| `concat_features_2` | 2×FeatureVector | FeatureVector |
| `cascade_rf` | FeatureVector, Trees, Depth | FeatureVector |

---

### Prediction

```python
class Prediction(DataWrapper):
    """
    Represents a stream of prediction batches.
    Each batch.data is (N, Classes).
    """
    pass
```

#### Purpose

`Prediction` represents **classification probabilities**. This is always the **output type of the entire GP tree**.

#### Data Format

Each batch has shape `(N, K)`:
- `N`: Number of samples in the batch
- `K`: Number of classes (e.g., 10 for CIFAR-10)

Each row sums to 1.0 (probability distribution over classes).

#### Example Prediction

```python
# For a 3-class problem with 2 samples:
predictions = np.array([
    [0.7, 0.2, 0.1],  # Sample 1: 70% class 0, 20% class 1, 10% class 2
    [0.1, 0.1, 0.8],  # Sample 2: 80% class 2
])
```

#### Operations That Produce Predictions

| Operation | Input | Output |
|-----------|-------|--------|
| `rf_classification` | FeatureVector, Trees, Depth | Prediction |
| `erf_classification` | FeatureVector, Trees, Depth | Prediction |
| `lr_classification` | FeatureVector | Prediction |
| `svm_classification` | FeatureVector | Prediction |
| `sum_prediction_2` | 2×Prediction | Prediction |

---

## Terminal (Parameter) Types

These are "marker classes" with no implementation. They exist solely to give DEAP's type system something to work with.

```python
class Weight: pass
class Trees: pass
class Depth: pass
class Frequency: pass
class Theta: pass
class Sigma: pass
class Order: pass
```

### What Are Terminal Types?

In GP terminology, **terminals** are the leaves of the tree—they don't have children. Parameter types are terminals that represent **hyperparameter values**.

### Why Empty Classes?

These classes don't need any methods or attributes because:

1. **They're just type markers**: DEAP only needs distinct types to exist
2. **Values are primitives**: The actual values (like `100` trees or `0.5` sigma) are plain Python numbers
3. **DEAP's ephemeral constants**: DEAP handles value generation and storage

### How They're Used in DEAP

```python
# In gp_setup.py
pset.addEphemeralConstant("t", lambda: random.randrange(50, 1001, 50), Trees)
#                          ^                                            ^
#                     name prefix                               TYPE MARKER

# This tells DEAP:
# 1. Create terminals of type "Trees"
# 2. When you need a Trees value, call the lambda to generate one
# 3. Values will be like: 50, 100, 150, ..., 1000
```

### Parameter Value Ranges

| Type | Range | Step | Purpose |
|------|-------|------|---------|
| `Trees` | 50-1000 | 50 | Number of trees in Random Forest |
| `Depth` | 10-100 | 10 | Maximum tree depth |
| `Frequency` | π/8 to π/2 | π/8 | Gabor filter frequency |
| `Theta` | 0 to 7π/8 | π/8 | Gabor filter angle |
| `Sigma` | 1-3 | 1 | Gaussian blur sigma |
| `Weight` | 0.0-1.0 | continuous | Linear combination weight |
| `Order` | 0-2 | 1 | Gaussian derivative order |

---

## How DEAP Uses These Types

### Strongly-Typed Genetic Programming (STGP)

DEAP's `gp.PrimitiveSetTyped` uses types to ensure valid trees:

```python
# When we register a primitive like:
pset.addPrimitive(sobel_filter, [Channel], Channel)
#                     ^            ^          ^
#               function name   input types  output type

# DEAP knows:
# - sobel_filter needs ONE input of type Channel
# - sobel_filter produces output of type Channel
```

### Type Checking During Tree Generation

When DEAP generates or mutates a tree:

```
1. Start with root needing type Prediction (our goal)
2. Find primitives that OUTPUT Prediction
   → rf_classification(FeatureVector, Trees, Depth) works!
3. Now need FeatureVector, Trees, and Depth
4. For Trees and Depth: use ephemeral constants
5. For FeatureVector: find primitives that OUTPUT FeatureVector
   → hog_features(Channel) works!
6. Now need Channel
7. Find primitives that OUTPUT Channel
   → Use terminal GetRed, or a filter, etc.
```

### Example Type-Valid Tree

```
rf_classification                  # Output: Prediction
├── hog_features                   # Output: FeatureVector
│   └── sobel_filter              # Output: Channel
│       └── GetRed                # Output: Channel (terminal)
├── 200                           # Output: Trees (ephemeral constant)
└── 50                            # Output: Depth (ephemeral constant)
```

Every connection is type-valid:
- `GetRed` → `Channel` → `sobel_filter` ✓
- `sobel_filter` → `Channel` → `hog_features` ✓
- `hog_features` → `FeatureVector` → `rf_classification` ✓

---

## The Lazy Evaluation Pattern

### What Is Lazy Evaluation?

**Lazy evaluation** means we don't compute anything until we actually need the result. Instead of processing all data upfront, we set up a pipeline of operations that execute on-demand.

### Why Use Lazy Evaluation?

1. **Memory efficiency**: We never load the entire dataset into memory at once
2. **Composability**: We can chain operations without intermediate storage
3. **Flexibility**: The same pipeline works for any dataset size

### How It Works in This System

```python
# EAGER (bad for large datasets):
channel = get_red_channel(all_images)  # Process ALL images now
filtered = sobel_filter(channel)        # Process ALL now
features = extract_features(filtered)   # Process ALL now
# Memory usage: 3x dataset size!

# LAZY (memory efficient):
channel = Channel(_get_red_iter)        # Just store the recipe
filtered = sobel_filter(channel)        # Chain the recipe
features = hog_features(filtered)       # Chain the recipe
# Nothing computed yet! Just recipes stored.

# Now actually compute, batch by batch:
for batch in features:                  # Triggers computation
    # batch 1: get_red → sobel → hog → yield
    # batch 2: get_red → sobel → hog → yield
    # ...
    process(batch)
# Memory usage: 1 batch at a time!
```

### The Iterator Chain

When you iterate over a `FeatureVector`, here's what happens:

```
You call: for batch in features
         │
         ▼
features.__iter__()  →  calls features._iterator_factory()
         │
         ▼
features iterator starts, needs data from filtered channel
         │
         ▼
filtered.__iter__()  →  calls filtered._iterator_factory()
         │
         ▼
filtered iterator starts, needs data from channel
         │
         ▼
channel.__iter__()  →  calls channel._iterator_factory()
         │
         ▼
channel iterator starts, needs data from _CURRENT_IMAGE
         │
         ▼
_CURRENT_IMAGE.__iter__() yields Batch([r, g, b])
         │
         ▼
channel processes → yields Batch(r)  # just red channel
         │
         ▼
filtered processes → yields Batch(sobel(r))
         │
         ▼
features processes → yields Batch(hog(sobel(r)))
         │
         ▼
You receive the batch!
```

---

## Complete Code Walkthrough

Let's go through the entire file line by line:

```python
import numpy as np
from typing import List, Union, Iterator, Callable
```

**Lines 1-2**: Import numpy for array operations and typing hints for documentation.

---

```python
class Batch:
    def __init__(self, data: np.ndarray):
        self.data = data
```

**Lines 4-6**: The `Batch` class is a simple container. The `__init__` method stores the numpy array in `self.data`. No validation, no transformation—just storage.

---

```python
class DataWrapper:
    """
    Base class for all data types flowing through the GP tree.
    Supports lazy iteration over batches.
    """
    def __init__(self, iterator_factory: Callable[[], Iterator[Batch]]):
        self._iterator_factory = iterator_factory
```

**Lines 8-14**: The `DataWrapper` class stores an `iterator_factory`. The type hint `Callable[[], Iterator[Batch]]` means:
- `Callable`: This is a function
- `[]`: It takes no arguments
- `Iterator[Batch]`: It returns an iterator that yields `Batch` objects

The leading underscore in `_iterator_factory` indicates it's "private" (internal use).

---

```python
    def __iter__(self) -> Iterator[Batch]:
        return self._iterator_factory()
```

**Lines 16-17**: The `__iter__` method makes `DataWrapper` iterable. When you write `for x in wrapper`, Python calls this method. It invokes the factory function to get a fresh iterator.

---

```python
class Image(DataWrapper):
    """
    Represents a stream of image batches.
    Each batch.data is List[np.ndarray] (channels) or np.ndarray (N, C, H, W).
    """
    pass
```

**Lines 19-24**: `Image` inherits everything from `DataWrapper` and adds nothing. The `pass` statement is Python's way of saying "this class body is intentionally empty."

---

```python
class Channel(DataWrapper):
    """
    Represents a stream of single channel batches.
    Each batch.data is (N, H, W).
    """
    pass
```

**Lines 26-31**: Same pattern as `Image`. The docstring documents the expected shape.

---

```python
class FeatureVector(DataWrapper):
    """
    Represents a stream of feature vector batches.
    Each batch.data is (N, Features).
    """
    pass
```

**Lines 33-38**: Features are 2D arrays where rows are samples and columns are features.

---

```python
class Prediction(DataWrapper):
    """
    Represents a stream of prediction batches.
    Each batch.data is (N, Classes).
    """
    pass
```

**Lines 40-45**: Predictions are probability distributions over classes.

---

```python
# Terminal Types
class Weight: pass
class Trees: pass
class Depth: pass
class Frequency: pass
class Theta: pass
class Sigma: pass
class Order: pass
```

**Lines 47-54**: Seven empty classes serving as type markers. They have no `__init__`, no methods, no attributes. They exist purely for DEAP's type system to distinguish between different kinds of parameters.

---

## Summary

The `gp_types.py` module establishes the **type foundation** for the entire GP system:

1. **`Batch`**: Simple container for numpy arrays
2. **`DataWrapper`**: Base class implementing lazy evaluation via iterator factories
3. **`Image`, `Channel`, `FeatureVector`, `Prediction`**: Data-flow types with distinct semantics
4. **`Weight`, `Trees`, etc.**: Parameter types for DEAP's type system

This type system ensures that evolved trees always represent **valid image classification pipelines** with proper data flow from input images to output predictions.
