# GP Setup Module (`gp_setup.py`) - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [What Is a Primitive Set?](#what-is-a-primitive-set)
3. [DEAP Fundamentals](#deap-fundamentals)
4. [The `create_primitive_set()` Function](#the-create_primitive_set-function)
5. [Terminals In Depth](#terminals-in-depth)
6. [Ephemeral Constants Explained](#ephemeral-constants-explained)
7. [Primitives In Depth](#primitives-in-depth)
8. [The Complete Type Flow](#the-complete-type-flow)
9. [Design Decisions and Rationale](#design-decisions-and-rationale)
10. [Complete Code Walkthrough](#complete-code-walkthrough)

---

## Overview

The `gp_setup.py` module creates the **primitive set** that defines the "language" of our Genetic Programming system. It tells DEAP:

1. What **terminals** (leaf nodes) exist and their types
2. What **primitives** (function nodes) exist, their input types, and output types
3. What **ephemeral constants** (random parameter values) can be generated

Think of this module as defining the grammar and vocabulary of a programming language that DEAP will use to generate and evolve programs (trees).

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRIMITIVE SET STRUCTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  EPHEMERAL CONSTANTS (Parameters)                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ t (Trees)     : 50, 100, 150, ..., 1000                  │   │
│  │ d (Depth)     : 10, 20, 30, ..., 100                     │   │
│  │ f (Frequency) : π/8, π/4, 3π/8, π/2                      │   │
│  │ theta         : 0, π/8, π/4, ..., 7π/8                   │   │
│  │ sigma         : 1, 2, 3                                   │   │
│  │ weight        : 0.0 to 1.0 (continuous)                  │   │
│  │ order         : 0, 1, 2                                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  TERMINALS (Data Inputs)                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ GetRed   : → Channel                                      │   │
│  │ GetGreen : → Channel                                      │   │
│  │ GetBlue  : → Channel                                      │   │
│  │ GetGray  : → Channel                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  PRIMITIVES (Functions)                                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Filters:      Channel → Channel                           │   │
│  │ Features:     Channel → FeatureVector                     │   │
│  │ Concat:       FeatureVector × N → FeatureVector           │   │
│  │ Classifiers:  FeatureVector → Prediction                  │   │
│  │ Ensemble:     Prediction × N → Prediction                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ROOT TYPE                                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Prediction (all trees must output this type)              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## What Is a Primitive Set?

### Definition

A **primitive set** in DEAP is a collection of:
- **Terminals**: Nodes with no children (leaves of the tree)
- **Primitives**: Nodes with children (internal nodes)

Together, they define all possible trees that can be generated.

### Analogy: Building Blocks

Think of LEGO blocks:
- **Terminals** are base plates (no connection below)
- **Primitives** are blocks with studs (can connect to things below)
- **Types** are the different stud patterns (ensure only compatible blocks connect)

### Strongly-Typed GP (STGP)

In **Strongly-Typed GP**, every terminal and primitive has:
- **Input types**: What types of children it expects
- **Output type**: What type it produces

This prevents nonsensical trees like:
```
sobel_filter(rf_classification(...))  # ERROR: RF outputs Prediction, not Channel!
```

---

## DEAP Fundamentals

### The `gp.PrimitiveSetTyped` Class

DEAP provides `gp.PrimitiveSetTyped` for creating typed primitive sets:

```python
from deap import gp

# Create a primitive set named "MAIN"
# No explicit inputs (we use terminals instead)
# Output type is Prediction (root type)
pset = gp.PrimitiveSetTyped("MAIN", [], Prediction)
```

### Parameters Explained

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `"MAIN"` | String | Name of the primitive set (for display/debugging) |
| `[]` | Empty list | Input types (none - we use terminals) |
| `Prediction` | Type class | The return type of the entire tree |

### Adding Components

DEAP provides three methods for adding components:

```python
# Add a terminal (leaf node)
pset.addTerminal(value, output_type, name="OptionalName")

# Add an ephemeral constant (random terminal)
pset.addEphemeralConstant(name, generator_function, output_type)

# Add a primitive (function node)
pset.addPrimitive(function, [input_types], output_type)
```

---

## The `create_primitive_set()` Function

This is the main function that builds the complete primitive set:

```python
def create_primitive_set():
    """
    Creates the primitive set defining the GP language for image classification trees.
    """
    pset = gp.PrimitiveSetTyped("MAIN", [], Prediction)
    
    # ... add all terminals, ephemeral constants, and primitives ...
    
    return pset
```

### The Legal Tree Structure

The function's docstring defines the valid data flow:

```
Legal Tree Structure (data flow):
1. Input: Channel terminals (GetRed, GetGreen, GetBlue, GetGray)
2. Image Filtering: Channel -> Channel (optional, can be stacked)
3. Feature Extraction: Channel -> FeatureVector
4. Feature Concatenation: FeatureVector x N -> FeatureVector (optional)
5. Cascade Classification: FeatureVector -> FeatureVector (optional)
6. Final Classification: FeatureVector -> Prediction
7. Ensemble Summation: Prediction x N -> Prediction (optional)
```

Every valid tree follows this pipeline, though steps 2, 4, 5, and 7 are optional.

---

## Terminals In Depth

### Data Terminals (Channel Inputs)

```python
pset.addTerminal(gp_ops.RedTerminal, Channel, name="GetRed")
pset.addTerminal(gp_ops.GreenTerminal, Channel, name="GetGreen")
pset.addTerminal(gp_ops.BlueTerminal, Channel, name="GetBlue")
pset.addTerminal(gp_ops.GrayTerminal, Channel, name="GetGray")
```

#### What Each Parameter Means

| Parameter | Example | Purpose |
|-----------|---------|---------|
| `value` | `gp_ops.RedTerminal` | The actual object that will appear in the tree |
| `output_type` | `Channel` | The type this terminal produces |
| `name` | `"GetRed"` | Display name (used in tree visualization) |

#### Why Pre-Created Objects?

The terminals (`RedTerminal`, etc.) are pre-created `Channel` objects in `gp_ops.py`:

```python
RedTerminal = Channel(_get_red_iter)
```

They're **lazy** - they don't hold data, they hold a function that produces data when iterated.

### What Terminals Look Like in Trees

```
rf_classification(
    hog_features(
        sobel_filter(
            GetRed           ← TERMINAL (no children)
        )
    ),
    500,                     ← TERMINAL (ephemeral constant)
    30                       ← TERMINAL (ephemeral constant)
)
```

---

## Ephemeral Constants Explained

### What Are Ephemeral Constants?

Ephemeral constants are **randomly generated terminals**. Unlike regular terminals (which are fixed), ephemeral constants:

1. Generate a random value when created
2. Keep that value for the life of the node
3. Can produce different values in different nodes

### Syntax

```python
pset.addEphemeralConstant("t", lambda: random.randrange(50, 1001, 50), Trees)
#                          ^          ^                                 ^
#                      name prefix   generator function             output type
```

### All Ephemeral Constants

#### Trees Parameter

```python
pset.addEphemeralConstant("t", lambda: random.randrange(50, 1001, 50), Trees)
```

- **Values**: 50, 100, 150, 200, ..., 1000
- **Purpose**: Number of trees in Random Forest / Extra Trees
- **Why this range?**: 
  - <50: Often too few trees for good ensemble
  - >1000: Diminishing returns, slower training

#### Depth Parameter

```python
pset.addEphemeralConstant("d", lambda: random.randrange(10, 101, 10), Depth)
```

- **Values**: 10, 20, 30, ..., 100
- **Purpose**: Maximum tree depth for RF/ERF
- **Why this range?**:
  - <10: May underfit
  - >100: Risk of overfitting, slower training

#### Frequency Parameter

```python
pset.addEphemeralConstant("f", lambda: random.choice([math.pi/8 * i for i in range(1, 5)]), Frequency)
```

- **Values**: π/8, π/4, 3π/8, π/2 (≈ 0.39, 0.79, 1.18, 1.57)
- **Purpose**: Spatial frequency for Gabor filters
- **Why these values?**: Covers low to high frequencies for texture detection

#### Theta (Angle) Parameter

```python
pset.addEphemeralConstant("theta", lambda: random.choice([math.pi/8 * i for i in range(0, 8)]), Theta)
```

- **Values**: 0, π/8, π/4, 3π/8, π/2, 5π/8, 3π/4, 7π/8 (0° to 157.5° in 22.5° steps)
- **Purpose**: Orientation angle for Gabor filters
- **Why these values?**: Covers all major orientations

#### Sigma Parameter

```python
pset.addEphemeralConstant("sigma", lambda: random.randint(1, 3), Sigma)
```

- **Values**: 1, 2, 3
- **Purpose**: Standard deviation for Gaussian filters
- **Why this range?**:
  - 1: Slight blur
  - 2: Moderate blur
  - 3: Strong blur

#### Weight Parameter

```python
pset.addEphemeralConstant("weight", lambda: random.random(), Weight)
```

- **Values**: 0.0 to 1.0 (continuous uniform)
- **Purpose**: Weights for linear combination of channels
- **Why continuous?**: Allows fine-grained control over channel mixing

#### Order Parameter

```python
pset.addEphemeralConstant("order", lambda: random.randint(0, 2), Order)
```

- **Values**: 0, 1, 2
- **Purpose**: Derivative order for Gaussian derivative filters
- **Meaning**:
  - 0: No derivative (just Gaussian blur)
  - 1: First derivative (edge detection)
  - 2: Second derivative (edge enhancement)

### How Ephemeral Constants Work Internally

When DEAP needs a `Trees` terminal:

1. It calls the generator function: `random.randrange(50, 1001, 50)` → e.g., `350`
2. It creates an ephemeral constant object with value `350`
3. This object becomes a node in the tree

The name prefix (`"t"`, `"d"`, etc.) is used internally by DEAP to distinguish ephemeral constants.

---

## Primitives In Depth

Primitives are the **function nodes** of the tree. They have children (inputs) and produce an output.

### Syntax

```python
pset.addPrimitive(function, [input_types], output_type)
```

### Category: Channel → Channel (Filters)

```python
pset.addPrimitive(gp_ops.sobel_filter, [Channel], Channel)
pset.addPrimitive(gp_ops.gaussian_filter, [Channel, Sigma], Channel)
```

**Example tree node**:
```
sobel_filter
    └── GetGray (Channel)
    
gaussian_filter
    ├── GetRed (Channel)
    └── 2 (Sigma)
```

### All Filter Primitives

| Function | Inputs | Output | Description |
|----------|--------|--------|-------------|
| `mean_filter` | Channel | Channel | 3×3 averaging |
| `median_filter` | Channel | Channel | 3×3 median |
| `min_filter` | Channel | Channel | 3×3 minimum |
| `max_filter` | Channel | Channel | 3×3 maximum |
| `gaussian_filter` | Channel, Sigma | Channel | Gaussian blur |
| `laplacian_filter` | Channel | Channel | Laplacian |
| `sobel_filter` | Channel | Channel | Sobel edges |
| `relu` | Channel | Channel | ReLU activation |
| `sqrt_op` | Channel | Channel | Square root |
| `gaud_filter` | Channel, Sigma, Order, Order | Channel | Gaussian derivatives |
| `log1_filter` | Channel | Channel | LoG σ=1 |
| `log2_filter` | Channel | Channel | LoG σ=2 |
| `gabor_filter` | Channel, Theta, Frequency | Channel | Gabor texture |
| `hog_filter` | Channel | Channel | HOG visualization |
| `lbp_filter` | Channel | Channel | LBP texture |
| `add_max_pool` | Channel, Channel | Channel | Add pooled |
| `sub_max_pool` | Channel, Channel | Channel | Subtract pooled |

### Special: `linear_combination`

```python
pset.addPrimitive(gp_ops.linear_combination, 
                  [Channel, Channel, Channel, Weight, Weight, Weight], Channel)
```

This is the most complex filter primitive:
- Takes 3 channels and 3 weights
- Outputs weighted combination: `w1*c1 + w2*c2 + w3*c3`

### Category: Channel → FeatureVector (Feature Extraction)

```python
pset.addPrimitive(gp_ops.hog_features, [Channel], FeatureVector)
pset.addPrimitive(gp_ops.gabor_features, [Channel, Theta, Frequency], FeatureVector)
```

| Function | Inputs | Output |
|----------|--------|--------|
| `histogram_features` | Channel | FeatureVector |
| `hog_features` | Channel | FeatureVector |
| `lbp_features` | Channel | FeatureVector |
| `concat_images` | Channel, Channel | FeatureVector |
| `sift_features` | Channel | FeatureVector |
| `hog_image_features` | Channel | FeatureVector |
| `lbp_image_features` | Channel | FeatureVector |
| `sobel_features` | Channel | FeatureVector |
| `gabor_features` | Channel, Theta, Frequency | FeatureVector |
| `gaussian_features` | Channel, Sigma | FeatureVector |
| `gaud_features` | Channel, Sigma, Order, Order | FeatureVector |

### Category: FeatureVector → FeatureVector (Concatenation & Cascade)

```python
pset.addPrimitive(gp_ops.concat_features_2, [FeatureVector, FeatureVector], FeatureVector)
pset.addPrimitive(gp_ops.cascade_rf, [FeatureVector, Trees, Depth], FeatureVector)
```

| Function | Inputs | Output | Description |
|----------|--------|--------|-------------|
| `concat_features_2` | FV, FV | FV | Concatenate 2 |
| `concat_features_3` | FV, FV, FV | FV | Concatenate 3 |
| `concat_features_4` | FV, FV, FV, FV | FV | Concatenate 4 |
| `cascade_rf` | FV, Trees, Depth | FV | RF predictions as features |
| `cascade_erf` | FV, Trees, Depth | FV | ERF predictions as features |
| `cascade_lr` | FV | FV | LR predictions as features |
| `cascade_svm` | FV | FV | SVM predictions as features |

### Category: FeatureVector → Prediction (Classification)

```python
pset.addPrimitive(gp_ops.rf_classification, [FeatureVector, Trees, Depth], Prediction)
pset.addPrimitive(gp_ops.lr_classification, [FeatureVector], Prediction)
```

| Function | Inputs | Output |
|----------|--------|--------|
| `rf_classification` | FeatureVector, Trees, Depth | Prediction |
| `erf_classification` | FeatureVector, Trees, Depth | Prediction |
| `lr_classification` | FeatureVector | Prediction |
| `svm_classification` | FeatureVector | Prediction |

### Category: Prediction → Prediction (Ensemble)

```python
pset.addPrimitive(gp_ops.sum_prediction_2, [Prediction, Prediction], Prediction)
```

| Function | Inputs | Output |
|----------|--------|--------|
| `sum_prediction_2` | Prediction, Prediction | Prediction |
| `sum_prediction_3` | Prediction, Prediction, Prediction | Prediction |

---

## The Complete Type Flow

### Valid Path Through Types

```
                 ┌─────────┐
                 │ Channel │ ◄── Terminals: GetRed, GetGreen, GetBlue, GetGray
                 └────┬────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    ┌───────────┐ ┌───────┐ ┌───────────────┐
    │ Filters   │ │ More  │ │ Feature       │
    │ (Channel  │ │Filters│ │ Extraction    │
    │ →Channel) │ │       │ │ (Channel →    │
    └─────┬─────┘ └───────┘ │ FeatureVector)│
          │                  └───────┬───────┘
          │                          │
          └───────────┬──────────────┘
                      ▼
              ┌───────────────┐
              │ FeatureVector │
              └───────┬───────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    ┌───────────┐ ┌───────┐ ┌───────────────┐
    │ Concat    │ │Cascade│ │Classification │
    │ Features  │ │ (FV→  │ │ (FV →         │
    │ (FV×N→FV) │ │  FV)  │ │ Prediction)   │
    └─────┬─────┘ └───────┘ └───────┬───────┘
          │                          │
          └───────────┬──────────────┘
                      ▼
              ┌────────────┐
              │ Prediction │
              └──────┬─────┘
                     │
          ┌──────────┼──────────┐
          ▼                     ▼
    ┌───────────┐         ┌─────────┐
    │ Ensemble  │         │ OUTPUT  │
    │ (Pred×N → │         │ (Done!) │
    │  Pred)    │         └─────────┘
    └─────┬─────┘
          │
          └────────► OUTPUT
```

### Example Valid Trees

**Minimal tree**:
```
lr_classification(histogram_features(GetGray()))
```

**With filtering**:
```
rf_classification(
    hog_features(sobel_filter(GetRed())),
    100, 20
)
```

**With concatenation**:
```
svm_classification(
    concat_features_2(
        hog_features(GetRed()),
        lbp_features(GetGray())
    )
)
```

**With cascade**:
```
rf_classification(
    cascade_lr(histogram_features(GetBlue())),
    200, 30
)
```

**With ensemble**:
```
sum_prediction_2(
    rf_classification(hog_features(GetRed()), 100, 20),
    lr_classification(lbp_features(GetGray()))
)
```

---

## Design Decisions and Rationale

### Why No Identity Primitives for Parameters?

The code explicitly states:

```python
# NOTE: We intentionally do NOT add identity primitives for parameter types.
# Parameter types (Trees, Depth, etc.) should only appear as leaf terminals,
# not as intermediate nodes that can be grown.
```

**The Problem**:
If we added `identity_trees(t: Trees) -> Trees`, DEAP could create:
```
rf_classification(
    features,
    identity_trees(identity_trees(identity_trees(100))),  # Useless!
    50
)
```

This wastes tree nodes without adding value.

**The Solution**:
Parameters only exist as leaf terminals. They can't be modified or wrapped.

### Why No DefaultHOG or DefaultRF Terminals?

The code mentions:

```python
# NOTE: We intentionally do NOT add DefaultHOG or DefaultRF terminals.
# These would allow degenerate trees that bypass the proper pipeline.
```

**The Problem**:
If we added a `DefaultRF` terminal that returns a `Prediction`, DEAP could create:
```
DefaultRF  # That's the entire tree!
```

This bypasses the evolution of feature extraction, which is the whole point.

**The Solution**:
Predictions can only be produced by classification primitives, which require features.

### Why These Specific Filters?

The selected filters cover different aspects of image analysis:

| Category | Filters | Purpose |
|----------|---------|---------|
| Smoothing | mean, median, gaussian | Noise reduction |
| Edge detection | sobel, laplacian, log | Finding boundaries |
| Texture | gabor, lbp, hog | Surface patterns |
| Multi-scale | log1, log2, gaud | Features at different scales |
| Combination | linear_combination, add/sub_max_pool | Custom representations |

### Why These Classifiers?

| Classifier | Strength | When It Wins |
|------------|----------|--------------|
| Random Forest | Robust, handles non-linear | Most general cases |
| Extra Trees | Faster, sometimes more accurate | Similar to RF |
| Logistic Regression | Fast, linear | Linearly separable classes |
| SVM | Strong margins | Clear class boundaries |

Offering multiple classifiers lets evolution find the best match for the data.

---

## Complete Code Walkthrough

```python
import random
import math
from deap import gp
from .gp_types import Image, Channel, FeatureVector, Prediction, Weight, Sigma, Trees, Depth, Frequency, Theta, Order
from . import gp_ops
```

**Lines 1-5**: Import standard library, DEAP, our types, and operations.

---

```python
def create_primitive_set():
    """
    Creates the primitive set defining the GP language for image classification trees.
    
    Legal Tree Structure (data flow):
    1. Input: Channel terminals (GetRed, GetGreen, GetBlue, GetGray)
    2. Image Filtering: Channel -> Channel (optional, can be stacked)
    3. Feature Extraction: Channel -> FeatureVector
    4. Feature Concatenation: FeatureVector x N -> FeatureVector (optional)
    5. Cascade Classification: FeatureVector -> FeatureVector (optional)
    6. Final Classification: FeatureVector -> Prediction
    7. Ensemble Summation: Prediction x N -> Prediction (optional)
    
    Parameter types (Trees, Depth, Sigma, etc.) are only used as arguments to primitives,
    never as intermediate data flow types. They should NOT have identity primitives.
    """
```

**Lines 7-21**: Docstring explaining the valid tree structure.

---

```python
    pset = gp.PrimitiveSetTyped("MAIN", [], Prediction)
```

**Line 23**: Create the primitive set. No input arguments; output type is `Prediction`.

---

```python
    # --- Terminals ---
    # Parameters (these are ephemeral constants - sampled once per tree node)
    pset.addEphemeralConstant("t", lambda: random.randrange(50, 1001, 50), Trees)
    pset.addEphemeralConstant("d", lambda: random.randrange(10, 101, 10), Depth)
    pset.addEphemeralConstant("f", lambda: random.choice([math.pi/8 * i for i in range(1, 5)]), Frequency)
    pset.addEphemeralConstant("theta", lambda: random.choice([math.pi/8 * i for i in range(0, 8)]), Theta)
    pset.addEphemeralConstant("sigma", lambda: random.randint(1, 3), Sigma)
    pset.addEphemeralConstant("weight", lambda: random.random(), Weight)
    pset.addEphemeralConstant("order", lambda: random.randint(0, 2), Order)
```

**Lines 26-32**: Add all ephemeral constants for parameters.

---

```python
    # NOTE: We intentionally do NOT add identity primitives for parameter types.
```

**Line 34-36**: Comment explaining why no identity functions.

---

```python
    # --- Input Nodes (Terminals) ---
    pset.addTerminal(gp_ops.RedTerminal, Channel, name="GetRed")
    pset.addTerminal(gp_ops.GreenTerminal, Channel, name="GetGreen")
    pset.addTerminal(gp_ops.BlueTerminal, Channel, name="GetBlue")
    pset.addTerminal(gp_ops.GrayTerminal, Channel, name="GetGray")
```

**Lines 38-44**: Add the four channel terminals.

---

```python
    # NOTE: We intentionally do NOT add DefaultHOG or DefaultRF terminals.
```

**Lines 46-48**: Comment explaining why no degenerate terminals.

---

```python
    # --- Channel Combination ---
    pset.addPrimitive(gp_ops.linear_combination, [Channel, Channel, Channel, Weight, Weight, Weight], Channel)
```

**Lines 50-51**: The complex 6-input channel combiner.

---

```python
    # --- Image Filtering (Channel -> Channel) ---
    pset.addPrimitive(gp_ops.mean_filter, [Channel], Channel)
    pset.addPrimitive(gp_ops.median_filter, [Channel], Channel)
    # ... (many more filters)
```

**Lines 53-69**: All Channel → Channel filter primitives.

---

```python
    # --- Feature Extraction (Channel -> FeatureVector) ---
    pset.addPrimitive(gp_ops.histogram_features, [Channel], FeatureVector)
    # ... (many more extractors)
```

**Lines 71-81**: All Channel → FeatureVector primitives.

---

```python
    # --- Feature Concatenation (FeatureVector x N -> FeatureVector) ---
    pset.addPrimitive(gp_ops.concat_features_2, [FeatureVector, FeatureVector], FeatureVector)
    pset.addPrimitive(gp_ops.concat_features_3, [FeatureVector, FeatureVector, FeatureVector], FeatureVector)
    pset.addPrimitive(gp_ops.concat_features_4, [FeatureVector, FeatureVector, FeatureVector, FeatureVector], FeatureVector)
```

**Lines 83-86**: Feature concatenation primitives.

---

```python
    # --- Cascade Classification (FeatureVector -> FeatureVector) ---
    pset.addPrimitive(gp_ops.cascade_rf, [FeatureVector, Trees, Depth], FeatureVector)
    pset.addPrimitive(gp_ops.cascade_erf, [FeatureVector, Trees, Depth], FeatureVector)
    pset.addPrimitive(gp_ops.cascade_lr, [FeatureVector], FeatureVector)
    pset.addPrimitive(gp_ops.cascade_svm, [FeatureVector], FeatureVector)
```

**Lines 88-92**: Cascade classifier primitives.

---

```python
    # --- Final Classification (FeatureVector -> Prediction) ---
    pset.addPrimitive(gp_ops.rf_classification, [FeatureVector, Trees, Depth], Prediction)
    pset.addPrimitive(gp_ops.erf_classification, [FeatureVector, Trees, Depth], Prediction)
    pset.addPrimitive(gp_ops.lr_classification, [FeatureVector], Prediction)
    pset.addPrimitive(gp_ops.svm_classification, [FeatureVector], Prediction)
```

**Lines 94-98**: Final classification primitives.

---

```python
    # --- Ensemble Summation (Prediction x N -> Prediction) ---
    pset.addPrimitive(gp_ops.sum_prediction_2, [Prediction, Prediction], Prediction)
    pset.addPrimitive(gp_ops.sum_prediction_3, [Prediction, Prediction, Prediction], Prediction)
```

**Lines 100-102**: Ensemble primitives.

---

```python
    return pset
```

**Line 104**: Return the fully configured primitive set.

---

## Summary

The `gp_setup.py` module defines the complete language of the GP system:

| Component | Count | Purpose |
|-----------|-------|---------|
| Ephemeral Constants | 7 | Random parameter values |
| Data Terminals | 4 | Channel inputs (R, G, B, Gray) |
| Filter Primitives | 17 | Image processing |
| Feature Extractors | 11 | Convert images to vectors |
| Feature Combiners | 7 | Concatenate and cascade |
| Classifiers | 4 | Make predictions |
| Ensemble | 2 | Combine predictions |

Total primitives: **45+** building blocks that can be combined into millions of possible trees.

The carefully designed type system ensures that every generated tree is a valid image classification pipeline.
