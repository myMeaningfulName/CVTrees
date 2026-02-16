# Extended Primitive Sets and Tree Structure

This document describes the extended primitive set system, including the new operations, parameterized variants, and the enforced tree structure that ensures valid GP individuals.

## Table of Contents

1. [Overview](#overview)
2. [Tree Structure Enforcement](#tree-structure-enforcement)
3. [Mode System](#mode-system)
4. [Extended Operations](#extended-operations)
5. [Extended Parameters](#extended-parameters)
6. [Primitive Set Statistics](#primitive-set-statistics)
7. [Usage Examples](#usage-examples)

---

## Overview

The CVTrees GP system uses DEAP's strongly-typed genetic programming to evolve image classification pipelines. The primitive set defines what operations can be combined to form valid trees.

The system now supports **two independent extension modes**:
- `extended_ops`: Adds more image filtering and feature extraction operations
- `extended_params`: Adds parameterized versions of operations for finer control

These modes can be enabled independently or together.

---

## Tree Structure Enforcement

### The Problem

Previously, trees could have illegal structures where:
- Summation operations (`sum_prediction_2`, `sum_prediction_3`) could appear in the middle of the tree
- Classification heads could appear after summation
- Nested summations like `sum(sum(...), classifier(...))` were possible

### The Solution: EnsembleOutput Type

We introduced a new type `EnsembleOutput` that enforces proper tree structure through the type system.

#### Type Hierarchy

```
Channel          → Image data (single color channel)
    ↓ (filters)
Channel          → Processed image data
    ↓ (feature extractors)
FeatureVector    → Extracted features
    ↓ (concatenation, optional)
FeatureVector    → Combined features
    ↓ (cascade classifiers, optional)
FeatureVector    → Augmented features with predictions
    ↓ (classifiers)
Prediction       → Class probability predictions
    ↓ (ensemble operations)
EnsembleOutput   → Final ensemble output (ROOT TYPE)
```

#### Why This Works

- **EnsembleOutput is the root type**: The primitive set is defined with `EnsembleOutput` as the return type
- **Nothing consumes EnsembleOutput**: No operation takes `EnsembleOutput` as input
- **Ensemble ops only produce EnsembleOutput**: `ensemble_single`, `ensemble_sum_2`, `ensemble_sum_3`

This makes it **impossible** for ensemble/summation to appear anywhere except at the root of the tree.

### Legal Tree Structure

A legal tree now follows this pattern:

```
ensemble_*(
    classifier_*(
        [cascade_*(
            [concat_features_*(
                feature_extractor_*(
                    [filter_*(
                        GetChannel
                    )]*
                )
            )]*
        )]*
    ),
    ...  # more classifiers for ensemble_sum_2/3
)
```

#### Example Legal Trees

```python
# Single classifier wrapped in ensemble
ensemble_single(
    rf_classification(
        hog_features(
            sobel_filter(GetRed)
        ), 100, 10
    )
)

# Two classifiers summed
ensemble_sum_2(
    rf_classification(histogram_features(GetRed), 100, 10),
    svm_classification(lbp_features(GetGreen))
)

# Three classifiers with cascading
ensemble_sum_3(
    rf_classification(
        cascade_rf(
            gabor_features(GetGray, 0.5, 1.0),
            50, 5
        ),
        100, 10
    ),
    erf_classification(hog_features(GetBlue), 200, 15),
    lr_classification(histogram_features(GetRed))
)
```

### Ensemble Operations

| Operation | Inputs | Output | Description |
|-----------|--------|--------|-------------|
| `ensemble_single` | 1 Prediction | EnsembleOutput | Wraps single classifier |
| `ensemble_sum_2` | 2 Predictions | EnsembleOutput | Sums two classifier outputs |
| `ensemble_sum_3` | 3 Predictions | EnsembleOutput | Sums three classifier outputs |

### Deprecated Operations

The following operations are deprecated and should not be used:
- `sum_prediction_2` - Use `ensemble_sum_2` instead
- `sum_prediction_3` - Use `ensemble_sum_3` instead

---

## Mode System

The primitive set can be created with different modes using `create_primitive_set()`:

```python
from src import gp_setup

# Basic mode (default)
pset = gp_setup.create_primitive_set()

# Extended operations only
pset = gp_setup.create_primitive_set(extended_ops=True)

# Extended parameters only  
pset = gp_setup.create_primitive_set(extended_params=True)

# Both extended operations and parameters
pset = gp_setup.create_primitive_set(extended_ops=True, extended_params=True)
```

---

## Extended Operations

When `extended_ops=True`, the following additional operations are registered:

### Extended Filters (Channel → Channel)

| Operation | Description |
|-----------|-------------|
| `bilateral_filter` | Edge-preserving smoothing filter |
| `erosion_filter` | Morphological erosion |
| `dilation_filter` | Morphological dilation |
| `opening_filter` | Morphological opening (erosion then dilation) |
| `closing_filter` | Morphological closing (dilation then erosion) |
| `tophat_filter` | Top-hat transform (difference between input and opening) |
| `blackhat_filter` | Black-hat transform (difference between closing and input) |
| `gradient_filter` | Morphological gradient (dilation minus erosion) |
| `prewitt_filter` | Prewitt edge detection |
| `scharr_filter` | Scharr edge detection (more accurate than Sobel) |
| `roberts_filter` | Roberts cross edge detection |
| `canny_filter` | Canny edge detection (binary output) |
| `frangi_filter` | Frangi vesselness filter (for tubular structures) |
| `hessian_filter` | Hessian-based ridge detection |
| `meijering_filter` | Meijering neuriteness filter |
| `sato_filter` | Sato tubeness filter |
| `unsharp_mask` | Unsharp masking for sharpening |
| `wiener_filter` | Wiener deconvolution filter |
| `nlm_filter` | Non-local means denoising |

### Extended Feature Extractors (Channel → FeatureVector)

| Operation | Description |
|-----------|-------------|
| `orb_features` | ORB (Oriented FAST and Rotated BRIEF) keypoint features |
| `brief_features` | BRIEF binary descriptor features |
| `daisy_features` | DAISY dense descriptor features |
| `glcm_features` | Gray-Level Co-occurrence Matrix texture features |
| `tamura_features` | Tamura texture features (coarseness, contrast, directionality) |
| `laws_features` | Laws' texture energy measures |
| `fourier_features` | Fourier transform magnitude spectrum features |
| `wavelet_features` | Wavelet decomposition features |
| `color_histogram_features` | Multi-channel color histogram |
| `moments_features` | Image moments (Hu moments) |

---

## Extended Parameters

When `extended_params=True`, additional parameter types and parameterized operations are registered.

### New Parameter Types

| Type | Description | Range |
|------|-------------|-------|
| `PixelsPerCell` | HOG pixels per cell | {4, 8, 16} |
| `CellsPerBlock` | HOG cells per block | {1, 2, 3} |
| `Orientations` | HOG orientations | {6, 9, 12} |
| `LBP_P` | LBP number of points | {8, 16, 24} |
| `LBP_R` | LBP radius | {1, 2, 3} |
| `KernelSize` | Filter kernel size | {3, 5, 7} |
| `MorphKernel` | Morphological kernel size | {3, 5, 7} |
| `CannyLow` | Canny low threshold | [10, 100] |
| `CannyHigh` | Canny high threshold | [100, 250] |
| `NumBins` | Histogram bins | {16, 32, 64, 128, 256} |
| `NumKeypoints` | Number of keypoints | {50, 100, 200, 500} |

### Parameterized Operations

#### Parameterized Filters

| Operation | Parameters | Description |
|-----------|------------|-------------|
| `hog_filter_param` | PixelsPerCell, CellsPerBlock, Orientations | Configurable HOG filter |
| `lbp_filter_param` | LBP_P, LBP_R | Configurable LBP filter |
| `gaussian_filter_param` | KernelSize | Configurable Gaussian blur |
| `median_filter_param` | KernelSize | Configurable median filter |
| `bilateral_filter_param` | KernelSize, Sigma, Sigma | Configurable bilateral filter |
| `erosion_filter_param` | MorphKernel | Configurable erosion |
| `dilation_filter_param` | MorphKernel | Configurable dilation |
| `opening_filter_param` | MorphKernel | Configurable opening |
| `closing_filter_param` | MorphKernel | Configurable closing |
| `canny_filter_param` | CannyLow, CannyHigh | Configurable Canny edge detection |

#### Parameterized Feature Extractors

| Operation | Parameters | Description |
|-----------|------------|-------------|
| `hog_features_param` | PixelsPerCell, CellsPerBlock, Orientations | Configurable HOG features |
| `lbp_features_param` | LBP_P, LBP_R | Configurable LBP features |
| `histogram_features_param` | NumBins | Configurable histogram |
| `orb_features_param` | NumKeypoints | Configurable ORB features |
| `brief_features_param` | NumKeypoints | Configurable BRIEF features |
| `glcm_features_param` | NumBins | Configurable GLCM features |

---

## Primitive Set Statistics

### Basic Mode
- **Primitives**: ~43
- **Terminals**: ~11
- Total: ~54

### Extended Operations Only
- **Primitives**: ~72 (+29 operations)
- **Terminals**: ~11
- Total: ~83

### Extended Parameters Only
- **Primitives**: ~59 (+16 parameterized ops)
- **Terminals**: ~22 (+11 parameter terminals)
- Total: ~81

### Both Extended Ops and Params
- **Primitives**: ~88 (+45 total)
- **Terminals**: ~22
- Total: ~110

---

## Usage Examples

### Command Line (test_tree_generation.py)

```bash
# Basic mode
python test_tree_generation.py --num-trees 10

# Extended operations
python test_tree_generation.py --extended-ops --num-trees 10

# Extended parameters
python test_tree_generation.py --extended-params --num-trees 10

# Both extended
python test_tree_generation.py --extended-ops --extended-params --num-trees 10

# With custom max depth
python test_tree_generation.py --extended-ops --extended-params --max-depth 20 --num-trees 10
```

### Python API

```python
from src import gp_setup, gp_types
from src.gp_experiment_parallel import run_parallel_experiment

# Create extended primitive set
pset = gp_setup.create_primitive_set(
    extended_ops=True,
    extended_params=True
)

# Run experiment with max_depth=20
runner = run_parallel_experiment(
    experiment_name="my_experiment",
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
    pset=pset,
    pop_size=50,
    generations=20,
    max_depth=20,  # Extended max depth
    n_workers=None  # Auto-detect
)
```

### Validating Trees

```python
from deap import gp

def validate_tree(individual, pset):
    """Check if a tree has valid structure."""
    expr_str = str(individual)
    
    # Must start with ensemble operation
    valid_roots = ['ensemble_single', 'ensemble_sum_2', 'ensemble_sum_3']
    has_valid_root = any(expr_str.startswith(root) for root in valid_roots)
    
    # Should not use deprecated operations
    deprecated = ['sum_prediction_2', 'sum_prediction_3']
    uses_deprecated = any(dep in expr_str for dep in deprecated)
    
    return has_valid_root and not uses_deprecated
```

---

## Implementation Files

| File | Purpose |
|------|---------|
| [gp_types.py](../src/gp_types.py) | Type definitions including `EnsembleOutput` |
| [gp_ops.py](../src/gp_ops.py) | All primitive operations |
| [gp_setup.py](../src/gp_setup.py) | Primitive set creation with mode support |
| [gp_experiment_parallel.py](../src/gp_experiment_parallel.py) | Parallel experiment runner |

---

## Changelog

### January 2026

- Added `EnsembleOutput` type to enforce proper tree structure
- Added `ensemble_single`, `ensemble_sum_2`, `ensemble_sum_3` operations
- Deprecated `sum_prediction_2`, `sum_prediction_3`
- Added 19 extended filter operations
- Added 10 extended feature extractor operations
- Added 11 new parameter types
- Added 16 parameterized operations
- Updated `create_primitive_set()` to accept `extended_ops` and `extended_params` flags
- Updated `run_parallel_experiment()` to accept `max_depth` parameter
- Changed default root type from `Prediction` to `EnsembleOutput`
