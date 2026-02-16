# GP Utils Module (`gp_utils.py`) - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [The Problem with Standard Tree Generation](#the-problem-with-standard-tree-generation)
3. [The `gen_safe` Function](#the-gen_safe-function)
4. [Algorithm Deep Dive](#algorithm-deep-dive)
5. [Helper Functions](#helper-functions)
6. [Step-by-Step Execution Examples](#step-by-step-execution-examples)
7. [Edge Cases and Error Handling](#edge-cases-and-error-handling)
8. [Complete Code Walkthrough](#complete-code-walkthrough)

---# GP Operations Module (`gp_ops.py`) - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [Module Architecture](#module-architecture)
3. [Batch Processing Helpers](#batch-processing-helpers)
4. [Input Nodes (Terminals)](#input-nodes-terminals)
5. [Image Filtering Operations](#image-filtering-operations)
6. [Feature Extraction Operations](#feature-extraction-operations)
7. [Feature Concatenation](#feature-concatenation)
8. [Classification Heads](#classification-heads)
9. [Ensemble Operations](#ensemble-operations)
10. [Cascade Classification](#cascade-classification)
11. [Complete Function Reference](#complete-function-reference)

---

## Overview

The `gp_ops.py` module is the **heart of the GP system**. It contains all the primitive operations (functions) that can appear as nodes in evolved GP trees. Think of this module as a library of image processing and machine learning building blocks that can be combined in countless ways.

```
┌─────────────────────────────────────────────────────────────────┐
│                    OPERATIONS OVERVIEW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT NODES (Terminals)           FILTERING OPERATIONS          │
│  ┌──────────────────────┐          ┌──────────────────────┐      │
│  │ GetRed               │          │ mean_filter          │      │
│  │ GetGreen             │──────────│ gaussian_filter      │      │
│  │ GetBlue              │          │ sobel_filter         │      │
│  │ GetGray              │          │ gabor_filter         │      │
│  └──────────────────────┘          │ ... (16 total)       │      │
│          │                         └──────────────────────┘      │
│          │                                   │                   │
│          │      FEATURE EXTRACTION           │                   │
│          │      ┌──────────────────────┐     │                   │
│          └──────│ histogram_features   │─────┘                   │
│                 │ hog_features         │                         │
│                 │ lbp_features         │                         │
│                 │ sift_features        │                         │
│                 │ ... (11 total)       │                         │
│                 └──────────────────────┘                         │
│                          │                                       │
│          FEATURE CONCATENATION                                   │
│          ┌──────────────────────┐                                │
│          │ concat_features_2    │                                │
│          │ concat_features_3    │                                │
│          │ concat_features_4    │                                │
│          └──────────────────────┘                                │
│                          │                                       │
│          CLASSIFICATION HEADS                                    │
│          ┌──────────────────────┐                                │
│          │ rf_classification    │                                │
│          │ erf_classification   │                                │
│          │ lr_classification    │                                │
│          │ svm_classification   │                                │
│          └──────────────────────┘                                │
│                          │                                       │
│          ENSEMBLE                                                │
│          ┌──────────────────────┐                                │
│          │ sum_prediction_2     │                                │
│          │ sum_prediction_3     │                                │
│          └──────────────────────┘                                │
│                          │                                       │
│                          ▼                                       │
│                    PREDICTION OUTPUT                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Architecture

### Imports

```python
import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from typing import Iterator, List
import skimage.measure
from .gp_types import Image, Channel, FeatureVector, Prediction, Weight, Sigma, Trees, Depth, Frequency, Theta, Batch, Order
from .gp_context import context, ExecutionMode
```

The module brings together:
- **NumPy**: Core array operations
- **OpenCV (cv2)**: SIFT features
- **scikit-image**: HOG, LBP, Gabor filters
- **SciPy ndimage**: Image filtering (Gaussian, Sobel, etc.)
- **scikit-learn**: Machine learning classifiers
- **Our types**: From `gp_types.py`
- **Context**: From `gp_context.py`

---

## Batch Processing Helpers

These helper functions enable the lazy evaluation pattern by creating new DataWrappers that transform input data.

### `map_batches(input_wrapper, func, output_type)`

```python
def map_batches(input_wrapper, func, output_type):
    """
    Helper to create a new DataWrapper that applies func to each batch.
    """
    def iterator_factory():
        for batch in input_wrapper:
            yield Batch(func(batch.data))
    return output_type(iterator_factory)
```

#### What It Does

1. Takes an input DataWrapper (like a Channel)
2. Takes a function to apply to each batch's data
3. Returns a new DataWrapper of the specified output type

#### The Lazy Magic

```python
# When you call:
result = map_batches(channel, sobel_function, Channel)

# NOTHING happens yet! Just returns a new Channel object
# containing a recipe (iterator_factory) for processing.

# Only when you iterate:
for batch in result:
    # NOW the processing happens, one batch at a time
```

#### Example Usage

```python
def apply_sobel(data):
    sx = ndimage.sobel(data, axis=1)
    sy = ndimage.sobel(data, axis=2)
    return np.hypot(sx, sy)

sobel_channel = map_batches(input_channel, apply_sobel, Channel)
```

### `map_batches_2`, `map_batches_3`, `map_batches_4`

These variants handle operations that combine 2, 3, or 4 inputs:

```python
def map_batches_2(input1, input2, func, output_type):
    def iterator_factory():
        for b1, b2 in zip(input1, input2):
            yield Batch(func(b1.data, b2.data))
    return output_type(iterator_factory)
```

#### Use Cases

| Function | Inputs | Example Operation |
|----------|--------|-------------------|
| `map_batches` | 1 | `sobel_filter(channel)` |
| `map_batches_2` | 2 | `concat_features_2(f1, f2)` |
| `map_batches_3` | 3 | `linear_combination(c1, c2, c3, w1, w2, w3)` |
| `map_batches_4` | 4 | `concat_features_4(f1, f2, f3, f4)` |

---

## Input Nodes (Terminals)

### The Global Image Store

```python
_CURRENT_IMAGE = None

def set_image(image: Image):
    global _CURRENT_IMAGE
    _CURRENT_IMAGE = image
```

Before executing any GP tree, you must call `set_image()` with the input data. This makes the image data available to all terminal nodes.

### Channel Extraction Functions

#### `_get_red_iter()`

```python
def _get_red_iter():
    if _CURRENT_IMAGE is None: raise ValueError("Image not set")
    for batch in _CURRENT_IMAGE:
        if isinstance(batch.data, list):
            yield Batch(batch.data[0])  # List format: [R, G, B]
        else:
            yield Batch(batch.data[:, 0])  # Tensor format: (N, C, H, W)
```

This function:
1. Checks that an image has been set
2. Iterates over batches in the image
3. Extracts the red channel (index 0)
4. Yields it as a new Batch

#### Why Two Formats?

The code handles two input formats:
- **List format**: `[red_array, green_array, blue_array]`
- **Tensor format**: `(N, 3, H, W)` where dimension 1 is channels

This flexibility allows different data loading strategies.

### `_get_gray_iter()` - Grayscale Conversion

```python
def _get_gray_iter():
    if _CURRENT_IMAGE is None: raise ValueError("Image not set")
    for batch in _CURRENT_IMAGE:
        if isinstance(batch.data, list):
            if len(batch.data) > 3:
                yield Batch(batch.data[3])  # Pre-computed gray
            else:
                r, g, b = batch.data[0], batch.data[1], batch.data[2]
                yield Batch(0.299 * r + 0.587 * g + 0.114 * b)
        else:
            # Similar logic for tensor format...
```

Uses the **ITU-R BT.601** standard for grayscale conversion:
- `Gray = 0.299 * R + 0.587 * G + 0.114 * B`

These weights reflect human eye sensitivity (most sensitive to green).

### Terminal Objects

```python
RedTerminal = Channel(_get_red_iter)
GreenTerminal = Channel(_get_green_iter)
BlueTerminal = Channel(_get_blue_iter)
GrayTerminal = Channel(_get_gray_iter)
```

These are pre-created `Channel` objects that DEAP uses as terminals:

```python
# In gp_setup.py:
pset.addTerminal(gp_ops.RedTerminal, Channel, name="GetRed")
```

---

## Image Filtering Operations

All filtering operations follow the pattern:
- **Input**: `Channel` (and possibly parameters)
- **Output**: `Channel`
- **Implementation**: Define a function, wrap with `map_batches`

### Basic Filters

#### `mean_filter(channel)`

```python
def mean_filter(channel: Channel) -> Channel:
    def func(data):
        return ndimage.uniform_filter(data, size=(1, 3, 3))
    return map_batches(channel, func, Channel)
```

**What it does**: Replaces each pixel with the average of its 3×3 neighborhood.

**Size `(1, 3, 3)`**: The `1` in the first position means "don't average across images in the batch" - only spatially.

#### `gaussian_filter(channel, sigma)`

```python
def gaussian_filter(channel: Channel, sigma: Sigma) -> Channel:
    def func(data):
        return ndimage.gaussian_filter(data, sigma=(0, sigma, sigma))
    return map_batches(channel, func, Channel)
```

**What it does**: Applies Gaussian blur (smoothing).

**Sigma parameter**: Controls blur strength (higher = more blur).

**`sigma=(0, sigma, sigma)`**: No blur across batch dimension, blur in both spatial dimensions.

### Edge Detection Filters

#### `sobel_filter(channel)`

```python
def sobel_filter(channel: Channel) -> Channel:
    def func(data):
        sx = ndimage.sobel(data, axis=1)  # Vertical edges
        sy = ndimage.sobel(data, axis=2)  # Horizontal edges
        return np.hypot(sx, sy)  # Magnitude
    return map_batches(channel, func, Channel)
```

**What it does**: Detects edges using Sobel operator.

**Process**:
1. Compute horizontal gradient (`sx`)
2. Compute vertical gradient (`sy`)
3. Combine with `hypot` (Euclidean magnitude): `√(sx² + sy²)`

#### `laplacian_filter(channel)`

```python
def laplacian_filter(channel: Channel) -> Channel:
    def func(data):
        return ndimage.laplace(data)
    return map_batches(channel, func, Channel)
```

**What it does**: Second-order edge detector that highlights regions of rapid intensity change.

### Advanced Filters

#### `gabor_filter(channel, theta, frequency)`

```python
def gabor_filter(channel: Channel, theta: Theta, frequency: Frequency) -> Channel:
    def func(data):
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            filt_real, filt_imag = gabor(data[i], frequency=frequency, theta=theta)
            out[i] = np.hypot(filt_real, filt_imag)
        return out
    return map_batches(channel, func, Channel)
```

**What it does**: Texture analysis filter tuned to a specific frequency and orientation.

**Parameters**:
- `theta`: Orientation angle (0 to π)
- `frequency`: Spatial frequency of the sinusoidal component

**Use case**: Detecting textures at specific orientations (e.g., stripes, fabric patterns).

#### `gaud_filter(channel, sigma, o1, o2)` - Gaussian Derivatives

```python
def gaud_filter(channel: Channel, sigma: Sigma, o1: Order, o2: Order) -> Channel:
    def func(data):
        return ndimage.gaussian_filter(data, sigma=(0, sigma, sigma), order=(0, o1, o2))
    return map_batches(channel, func, Channel)
```

**What it does**: Computes Gaussian derivatives (useful for edge detection with scale).

**Order parameters**:
- `o1`: Derivative order in y-direction (0, 1, or 2)
- `o2`: Derivative order in x-direction (0, 1, or 2)

**Examples**:
- `o1=0, o2=0`: Just Gaussian blur
- `o1=1, o2=0`: First derivative in y (vertical edges)
- `o1=0, o2=1`: First derivative in x (horizontal edges)
- `o1=1, o2=1`: Mixed second derivative

#### `log1_filter` and `log2_filter` - Laplacian of Gaussian

```python
def log1_filter(channel: Channel) -> Channel:
    def func(data):
        return ndimage.gaussian_laplace(data, sigma=(0, 1, 1))
    return map_batches(channel, func, Channel)
```

**What it does**: Blob detection filter. Combines Gaussian smoothing with Laplacian edge detection.

**`log1` vs `log2`**: Different sigma values (1 vs 2) for detecting blobs at different scales.

### Visualization Filters

#### `hog_filter(channel)` - HOG Visualization

```python
def hog_filter(channel: Channel) -> Channel:
    def func(data):
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            try:
                _, hog_img = hog(data[i], orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), visualize=True)
                out[i] = hog_img
            except:
                pass
        return out
    return map_batches(channel, func, Channel)
```

**What it does**: Generates a visualization of the HOG (Histogram of Oriented Gradients) descriptor.

**Output**: An image showing gradient orientations as little lines.

#### `lbp_filter(channel)` - Local Binary Pattern

```python
def lbp_filter(channel: Channel) -> Channel:
    def func(data):
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            out[i] = local_binary_pattern(data[i], P=8, R=1.5, method='uniform')
        return out
    return map_batches(channel, func, Channel)
```

**What it does**: Computes the Local Binary Pattern, a texture descriptor.

**Parameters**:
- `P=8`: Number of circularly symmetric neighbors
- `R=1.5`: Radius of the circle
- `method='uniform'`: Uses uniform LBP patterns (rotation invariant)

### Combination Filters

#### `add_max_pool(c1, c2)` and `sub_max_pool(c1, c2)`

```python
def add_max_pool(c1: Channel, c2: Channel) -> Channel:
    def func(d1, d2):
        p1 = skimage.measure.block_reduce(d1, (1, 2, 2), np.max)
        p2 = skimage.measure.block_reduce(d2, (1, 2, 2), np.max)
        min_h = min(p1.shape[1], p2.shape[1])
        min_w = min(p1.shape[2], p2.shape[2])
        return p1[:, :min_h, :min_w] + p2[:, :min_h, :min_w]
    return map_batches_2(c1, c2, func, Channel)
```

**What it does**: 
1. Applies 2×2 max pooling to both channels
2. Adds (or subtracts) the results

**Why min_h, min_w?**: Max pooling might produce slightly different sizes; we crop to match.

#### `linear_combination(c1, c2, c3, w1, w2, w3)`

```python
def linear_combination(c1: Channel, c2: Channel, c3: Channel,
                       w1: Weight, w2: Weight, w3: Weight) -> Channel:
    def func(d1, d2, d3):
        return w1 * d1 + w2 * d2 + w3 * d3
    return map_batches_3(c1, c2, c3, func, Channel)
```

**What it does**: Creates a weighted combination of three channels.

**Example**: `0.5 * Red + 0.3 * Green + 0.2 * Blue` creates a custom grayscale.

### Activation Functions

#### `relu(channel)`

```python
def relu(channel: Channel) -> Channel:
    def func(data):
        return np.maximum(0, data)
    return map_batches(channel, func, Channel)
```

**What it does**: Applies ReLU (Rectified Linear Unit) - sets negative values to 0.

**Common use**: After edge detectors to keep only positive responses.

#### `sqrt_op(channel)`

```python
def sqrt_op(channel: Channel) -> Channel:
    def func(data):
        d = data.copy()
        neg_mask = d < 0
        d[neg_mask] = 1  # Set negatives to 1 (sqrt(1) = 1)
        d[~neg_mask] = np.sqrt(d[~neg_mask])
        return d
    return map_batches(channel, func, Channel)
```

**What it does**: Square root transformation for contrast adjustment.

**Handling negatives**: Negative values become 1 (avoiding `sqrt` of negative).

---

## Feature Extraction Operations

These operations convert spatial data (images) into 1D feature vectors suitable for classification.

### `histogram_features(channel)`

```python
def histogram_features(channel: Channel) -> FeatureVector:
    def func(data):
        feats = []
        for img in data:
            hist, _ = np.histogram(img, bins=256, range=(0, 1))
            feats.append(hist)
        return np.array(feats)
    return map_batches(channel, func, FeatureVector)
```

**What it does**: Computes intensity histogram for each image.

**Output shape**: `(N, 256)` - 256 bins for each image.

**Use case**: Simple but effective for images with distinctive intensity distributions.

### `hog_features(channel)`

```python
def hog_features(channel: Channel) -> FeatureVector:
    def func(data):
        feats = []
        for img in data:
            try:
                fd = hog(img, orientations=9, pixels_per_cell=(4, 4),
                        cells_per_block=(2, 2))
                feats.append(fd)
            except:
                feats.append(np.zeros(10))
        return np.array(feats)
    return map_batches(channel, func, FeatureVector)
```

**What it does**: Extracts Histogram of Oriented Gradients descriptor.

**Parameters**:
- `orientations=9`: 9 orientation bins (20° each)
- `pixels_per_cell=(4, 4)`: Each cell is 4×4 pixels
- `cells_per_block=(2, 2)`: Normalize over 2×2 cell blocks

**Output size**: Depends on image size. For 32×32 images: thousands of features.

**Use case**: Excellent for object detection and recognition.

### `lbp_features(channel)`

```python
def lbp_features(channel: Channel) -> FeatureVector:
    def func(data):
        feats = []
        for img in data:
            lbp = local_binary_pattern(img, P=8, R=1.5, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10 + 3),
                                   range=(0, 10 + 2))
            feats.append(hist)
        return np.array(feats)
    return map_batches(channel, func, FeatureVector)
```

**What it does**: Extracts LBP texture descriptor histogram.

**Process**:
1. Compute LBP image
2. Create histogram of LBP patterns

**Output shape**: `(N, ~12)` - compact texture representation.

**Use case**: Texture classification, face recognition.

### `sift_features(channel)`

```python
def sift_features(channel: Channel) -> FeatureVector:
    def func(data):
        feats = []
        sift = cv2.SIFT_create()
        for img in data:
            img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            
            # Dense SIFT on a grid
            step_size = 8
            kp = [cv2.KeyPoint(x, y, step_size) 
                  for y in range(0, img_uint8.shape[0], step_size)
                  for x in range(0, img_uint8.shape[1], step_size)]
            
            _, des = sift.compute(img_uint8, kp)
            
            if des is None:
                feats.append(np.zeros(128))
            else:
                feats.append(np.mean(des, axis=0))  # Average all descriptors
        return np.array(feats)
    return map_batches(channel, func, FeatureVector)
```

**What it does**: Computes dense SIFT descriptors.

**Process**:
1. Convert to uint8 (SIFT requirement)
2. Create keypoints on a grid (every 8 pixels)
3. Compute SIFT descriptor at each keypoint
4. Average all descriptors into one 128-D vector

**Output shape**: `(N, 128)`

### `concat_images(c1, c2)`

```python
def concat_images(c1: Channel, c2: Channel) -> FeatureVector:
    def func(d1, d2):
        f1 = d1.reshape(d1.shape[0], -1)  # Flatten
        f2 = d2.reshape(d2.shape[0], -1)  # Flatten
        return np.hstack([f1, f2])        # Concatenate
    return map_batches_2(c1, c2, func, FeatureVector)
```

**What it does**: Flattens two channels and concatenates them as features.

**Output shape**: `(N, 2 * H * W)` - can be very large!

### FE Wrappers (Filter + Flatten)

These combine a filter with flattening into a single feature extraction operation.

#### `hog_image_features(channel)`

```python
def hog_image_features(channel: Channel) -> FeatureVector:
    def func(data):
        out = []
        for i in range(data.shape[0]):
            try:
                _, hog_img = hog(data[i], orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), visualize=True)
                out.append(hog_img.flatten())
            except:
                out.append(np.zeros(data.shape[1]*data.shape[2]))
        return np.array(out)
    return map_batches(channel, func, FeatureVector)
```

**What it does**: Gets the HOG visualization image and flattens it to features.

**Difference from `hog_features`**: Uses the visual image, not the descriptor.

#### `sobel_features(channel)`

```python
def sobel_features(channel: Channel) -> FeatureVector:
    def func(data):
        sx = ndimage.sobel(data, axis=1)
        sy = ndimage.sobel(data, axis=2)
        s = np.hypot(sx, sy)
        return s.reshape(s.shape[0], -1)  # Flatten
    return map_batches(channel, func, FeatureVector)
```

**What it does**: Sobel edge detection + flatten.

---

## Feature Concatenation

Operations for combining multiple feature vectors.

### `concat_features_2(f1, f2)`

```python
def concat_features_2(f1: FeatureVector, f2: FeatureVector) -> FeatureVector:
    def func(d1, d2):
        return np.hstack([d1, d2])
    return map_batches_2(f1, f2, func, FeatureVector)
```

**What it does**: Horizontally stacks two feature arrays.

**Example**:
```python
# f1: (N, 100), f2: (N, 50)
# Result: (N, 150)
```

### `concat_features_3` and `concat_features_4`

Same pattern for 3 and 4 feature vectors.

---

## Classification Heads

These are the **stateful** operations that train and apply machine learning classifiers.

### Common Pattern

All classifiers follow this pattern:

```python
def xxx_classification(features: FeatureVector, ...) -> Prediction:
    node_id = context.get_next_node_id()  # Get unique ID
    
    def iterator_factory():
        if context.mode == ExecutionMode.TRAIN:
            # 1. Collect all features
            # 2. Get labels from context
            # 3. Train classifier
            # 4. Run cross-validation for predictions
            # 5. Save classifier to context
            # 6. Yield predictions
        else:  # EVAL mode
            # 1. Load classifier from context
            # 2. For each batch, predict
            # 3. Yield predictions
    
    return Prediction(iterator_factory)
```

### `rf_classification(features, t, d)` - Random Forest

```python
def rf_classification(features: FeatureVector, t: Trees, d: Depth) -> Prediction:
    node_id = context.get_next_node_id()
    
    def iterator_factory():
        if context.mode == ExecutionMode.TRAIN:
            # Collect all feature batches
            X_list = []
            for batch in features:
                X_list.append(batch.data)
            if not X_list: return
            X = np.vstack(X_list)
            y = context.train_labels
            
            # Create and train classifier
            clf = RandomForestClassifier(n_estimators=t, max_depth=d, n_jobs=-1)
            try:
                # Cross-validation for better probability estimates
                probs = cross_val_predict(clf, X, y, cv=3, method='predict_proba')
                clf.fit(X, y)  # Final fit on all data
            except:
                clf.fit(X, y)
                probs = clf.predict_proba(X)
            
            # Save for later use
            context.save_model(node_id, clf)
            yield Batch(probs)
        else:
            # EVAL: Load and predict
            clf = context.get_model(node_id)
            if clf:
                for batch in features:
                    yield Batch(clf.predict_proba(batch.data))
            else:
                for batch in features:
                    yield Batch(np.zeros((batch.data.shape[0], 10)))

    return Prediction(iterator_factory)
```

**Parameters**:
- `t` (Trees): Number of trees (50-1000)
- `d` (Depth): Maximum tree depth (10-100)

**Why Cross-Validation?**

During training, we use `cross_val_predict` with 3-fold CV:
- Gives more realistic probability estimates
- Avoids overfitting to training data
- Makes fitness evaluation more meaningful

### `erf_classification(features, t, d)` - Extra Trees

Nearly identical to Random Forest but uses `ExtraTreesClassifier`:
- More random splits (doesn't find optimal split points)
- Often faster than RF
- Can be more accurate for some problems

### `lr_classification(features)` - Logistic Regression

```python
def lr_classification(features: FeatureVector) -> Prediction:
    node_id = context.get_next_node_id()
    
    def iterator_factory():
        if context.mode == ExecutionMode.TRAIN:
            X_list = []
            for batch in features:
                X_list.append(batch.data)
            if not X_list: return
            X = np.vstack(X_list)
            y = context.train_labels
            
            clf = LogisticRegression(max_iter=1000, n_jobs=-1)
            # ... similar pattern ...
```

**No hyperparameters**: Uses default settings with increased `max_iter`.

**Use case**: Fast, interpretable, works well when classes are linearly separable.

### `svm_classification(features)` - Support Vector Machine

```python
clf = SVC(probability=True)
```

**Key setting**: `probability=True` enables `predict_proba()`.

**Note**: SVM with probability is slower due to Platt scaling.

---

## Ensemble Operations

### `sum_prediction_2(p1, p2)`

```python
def sum_prediction_2(p1: Prediction, p2: Prediction) -> Prediction:
    def func(d1, d2):
        return d1 + d2
    return map_batches_2(p1, p2, func, Prediction)
```

**What it does**: Adds probability predictions from two classifiers.

**Why add?**: Creates an ensemble effect. The final prediction uses `argmax`, so adding probabilities boosts confident predictions.

**Example**:
```python
# Classifier 1: [0.8, 0.1, 0.1]  (confident class 0)
# Classifier 2: [0.6, 0.3, 0.1]  (also leans class 0)
# Sum:          [1.4, 0.4, 0.2]
# Prediction:   class 0 (argmax)
```

### `sum_prediction_3(p1, p2, p3)`

Same pattern for three classifiers.

---

## Cascade Classification

Cascade classifiers are **intermediate** classifiers that enrich features with their predictions.

### How It Works

```
Features → Classifier → Predictions
    │                       │
    └───────────────────────┴─→ Concatenated → New Features
```

The new features contain both:
1. Original features
2. Class probability predictions

### `cascade_rf(features, t, d)`

```python
def cascade_rf(features: FeatureVector, t: Trees, d: Depth) -> FeatureVector:
    pred = rf_classification(features, t, d)  # Get predictions
    return _cascade_helper(features, pred)     # Concatenate with features
```

### `_cascade_helper(features, pred)`

```python
def _cascade_helper(features: FeatureVector, pred: Prediction) -> FeatureVector:
    def iterator_factory():
        preds_iter = iter(pred)
        feats_iter = iter(features)
        
        if context.mode == ExecutionMode.TRAIN:
            # Training: one big batch
            all_feats = []
            for b in feats_iter:
                all_feats.append(b.data)
            if not all_feats: return
            X = np.vstack(all_feats)
            
            try:
                p_batch = next(preds_iter)
                yield Batch(np.hstack([X, p_batch.data]))
            except StopIteration:
                pass
        else:
            # Eval: batch by batch
            for f_batch, p_batch in zip(feats_iter, preds_iter):
                yield Batch(np.hstack([f_batch.data, p_batch.data]))

    return FeatureVector(iterator_factory)
```

**Why different for TRAIN vs EVAL?**

During training, classifiers collect ALL data before fitting (one big batch).
During eval, they process batch-by-batch.

---

## Complete Function Reference

### Input Terminals

| Name | Type | Description |
|------|------|-------------|
| `RedTerminal` | Channel | Red channel from input image |
| `GreenTerminal` | Channel | Green channel |
| `BlueTerminal` | Channel | Blue channel |
| `GrayTerminal` | Channel | Grayscale conversion |

### Filters (Channel → Channel)

| Function | Parameters | Description |
|----------|------------|-------------|
| `mean_filter` | - | 3×3 averaging |
| `median_filter` | - | 3×3 median |
| `min_filter` | - | 3×3 minimum |
| `max_filter` | - | 3×3 maximum |
| `gaussian_filter` | sigma | Gaussian blur |
| `laplacian_filter` | - | Laplacian edge detector |
| `sobel_filter` | - | Sobel edge detector |
| `relu` | - | ReLU activation |
| `sqrt_op` | - | Square root |
| `gaud_filter` | sigma, o1, o2 | Gaussian derivatives |
| `log1_filter` | - | LoG (σ=1) |
| `log2_filter` | - | LoG (σ=2) |
| `gabor_filter` | theta, freq | Gabor texture filter |
| `hog_filter` | - | HOG visualization |
| `lbp_filter` | - | LBP texture |
| `add_max_pool` | c2 | Add after max pooling |
| `sub_max_pool` | c2 | Subtract after max pooling |
| `linear_combination` | c2, c3, w1, w2, w3 | Weighted sum of 3 channels |

### Feature Extraction (Channel → FeatureVector)

| Function | Parameters | Description |
|----------|------------|-------------|
| `histogram_features` | - | Intensity histogram |
| `hog_features` | - | HOG descriptor |
| `lbp_features` | - | LBP histogram |
| `sift_features` | - | Dense SIFT |
| `concat_images` | c2 | Flatten and concat 2 channels |
| `hog_image_features` | - | HOG visualization flattened |
| `lbp_image_features` | - | LBP image flattened |
| `sobel_features` | - | Sobel edges flattened |
| `gabor_features` | theta, freq | Gabor response flattened |
| `gaussian_features` | sigma | Gaussian blur flattened |
| `gaud_features` | sigma, o1, o2 | Gaussian derivatives flattened |

### Feature Operations (FeatureVector → FeatureVector)

| Function | Parameters | Description |
|----------|------------|-------------|
| `concat_features_2` | f2 | Concatenate 2 feature vectors |
| `concat_features_3` | f2, f3 | Concatenate 3 |
| `concat_features_4` | f2, f3, f4 | Concatenate 4 |
| `cascade_rf` | t, d | RF predictions as features |
| `cascade_erf` | t, d | Extra Trees predictions as features |
| `cascade_lr` | - | LR predictions as features |
| `cascade_svm` | - | SVM predictions as features |

### Classification (FeatureVector → Prediction)

| Function | Parameters | Description |
|----------|------------|-------------|
| `rf_classification` | t, d | Random Forest classifier |
| `erf_classification` | t, d | Extra Trees classifier |
| `lr_classification` | - | Logistic Regression |
| `svm_classification` | - | Support Vector Machine |

### Ensemble (Prediction → Prediction)

| Function | Parameters | Description |
|----------|------------|-------------|
| `sum_prediction_2` | p2 | Sum 2 predictions |
| `sum_prediction_3` | p2, p3 | Sum 3 predictions |

---

## Summary

The `gp_ops.py` module provides a comprehensive library of image processing and machine learning primitives that GP can combine into classification pipelines. Key features:

1. **Lazy evaluation**: Operations don't execute until iteration
2. **Stateful classifiers**: Models are trained once, reused during evaluation
3. **Type safety**: Each operation has clear input/output types
4. **Flexibility**: Operations can be stacked, combined, and ensembled in countless ways

This module is the "vocabulary" of the GP language - every evolved tree is built from these building blocks.


## Overview

The `gp_utils.py` module contains utility functions for the Genetic Programming system. The primary function is `gen_safe`, a custom tree generation algorithm that handles complex type systems where some types have no terminals.

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE TERMINATION PROBLEM                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Standard DEAP tree generation:                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 1. Start at root with type Prediction                      │ │
│  │ 2. Need to terminate? Look for Prediction terminals        │ │
│  │ 3. PROBLEM: There are NO Prediction terminals!             │ │
│  │ 4. 💥 Generation fails or creates invalid trees            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Our gen_safe solution:                                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 1. Start at root with type Prediction                      │ │
│  │ 2. Need to terminate? No terminals? Find a primitive!      │ │
│  │ 3. Pick rf_classification(FeatureVector, Trees, Depth)     │ │
│  │ 4. Recursively generate children (with termination)        │ │
│  │ 5. ✅ Always generates valid trees!                         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Problem with Standard Tree Generation

### DEAP's Built-in Generators

DEAP provides tree generation functions like `genGrow` and `genFull`:

```python
# DEAP's genGrow (simplified):
def genGrow(pset, min_depth, max_depth, type_):
    if depth_reached or should_terminate:
        return pick_terminal(type_)  # <-- PROBLEM!
    else:
        return pick_primitive_and_recurse(type_)
```

### The Terminal Requirement

These generators **require terminals for every type** that might need to be generated. But in our system:

| Type | Has Terminals? | Example Terminals |
|------|----------------|-------------------|
| `Channel` | ✅ Yes | GetRed, GetGreen, GetBlue, GetGray |
| `Trees` | ✅ Yes | Ephemeral constants (50, 100, ...) |
| `Depth` | ✅ Yes | Ephemeral constants (10, 20, ...) |
| `Sigma` | ✅ Yes | Ephemeral constants (1, 2, 3) |
| `FeatureVector` | ❌ **No!** | - |
| `Prediction` | ❌ **No!** | - |

### Why No Terminals for Some Types?

We deliberately avoid terminals for `FeatureVector` and `Prediction`:

**If we had a `DefaultRF` Prediction terminal**:
```
DefaultRF  # Complete tree! But completely useless.
```

**If we had a `DefaultHOG` FeatureVector terminal**:
```
rf_classification(DefaultHOG, 100, 50)
# No image processing at all!
```

These "degenerate" trees bypass the whole point of GP: evolving interesting pipelines.

### The Crash

With standard DEAP generators:

```python
# Trying to generate a tree rooted at Prediction type
expr = gp.genGrow(pset, min_=1, max_=3, type_=Prediction)

# CRASH! Cannot find terminal for type Prediction!
```

---

## The `gen_safe` Function

### Signature

```python
def gen_safe(pset, min_, max_, type_):
    """
    Generates a tree ensuring that we can always terminate even if the root type
    has no direct terminals (by falling back to non-recursive primitives).
    """
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `pset` | `PrimitiveSetTyped` | The primitive set defining available operations |
| `min_` | `int` | Minimum depth (not strictly enforced) |
| `max_` | `int` | Maximum depth for recursive growth |
| `type_` | `type` | The type to generate (usually `Prediction`) |

### Return Value

Returns a **list of tree nodes** in prefix notation (the format DEAP expects).

### Key Innovation

When we need to terminate but have no terminals, we **find a primitive that can be terminated**:

```python
# For type Prediction:
# 1. No terminals exist
# 2. Find primitives that output Prediction:
#    - rf_classification(FeatureVector, Trees, Depth)
#    - lr_classification(FeatureVector)
#    - etc.
# 3. Pick one (e.g., lr_classification)
# 4. Generate its children recursively (forcing termination)
```

---

## Algorithm Deep Dive

### The Recursive `generate` Function

```python
def generate(type_, depth):
    terminals = pset.terminals.get(type_, [])
    primitives = pset.primitives.get(type_, [])
    
    # ... decision logic ...
```

This inner function does the actual generation. It's recursive, building the tree from root to leaves.

### Decision Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    GENERATION DECISION TREE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                    depth <= 0?                                   │
│                    (Must terminate)                              │
│                         │                                        │
│              ┌──────────┴──────────┐                            │
│              │                     │                             │
│             YES                   NO                             │
│              │                     │                             │
│              ▼                     ▼                             │
│     Has terminals?           Has terminals?                      │
│         │                         │                              │
│    ┌────┴────┐              ┌────┴────┐                         │
│   YES       NO             YES       NO                          │
│    │         │              │         │                          │
│    ▼         ▼              ▼         ▼                          │
│  Pick     Find safe      Random:   Pick                          │
│ terminal  primitive &   terminal  primitive &                    │
│           recurse(0)   OR prim?   recurse(depth-1)               │
│              │              │         │                          │
│              │         ┌────┴────┐    │                          │
│              │       term      prim   │                          │
│              │         │         │    │                          │
│              ▼         ▼         ▼    ▼                          │
│           TERMINATE  TERMINATE  GROW  GROW                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### The Two Terminal Helpers

#### `pick_terminal()`

```python
def pick_terminal():
    """Helper to properly instantiate a terminal."""
    term = random.choice(terminals)
    if inspect.isclass(term):
        # Ephemeral constant - needs instantiation
        return [term()]
    else:
        # Regular terminal - use as-is
        return [term]
```

**Why check `isclass`?**

DEAP stores ephemeral constants differently from regular terminals:
- **Regular terminal**: An actual object (e.g., `RedTerminal`)
- **Ephemeral constant**: A **class** that must be instantiated

```python
# Regular terminal
RedTerminal  # Object, use directly

# Ephemeral constant for Trees
<class 'deap.gp.t0'>  # Class! Must call t0() to create instance
t0()  # Returns an object with value like 350
```

#### `pick_primitive_and_recurse(available_depth)`

```python
def pick_primitive_and_recurse(available_depth):
    """Pick a primitive and recursively generate its arguments."""
    prim = random.choice(primitives)
    expr = [prim]
    for arg_type in prim.args:
        expr.extend(generate(arg_type, available_depth - 1))
    return expr
```

This:
1. Randomly picks a primitive that outputs our type
2. Starts the expression with that primitive
3. Recursively generates each argument

### The Safe Termination Fallback

When `depth <= 0` and no terminals exist:

```python
if depth <= 0:
    if terminals:
        return pick_terminal()
    else:
        # No terminals - must pick a non-recursive primitive
        safe_prims = [p for p in primitives if type_ not in p.args]
        
        if not safe_prims:
            raise ValueError(f"Cannot terminate type {type_}")
        
        prim = random.choice(safe_prims)
        expr = [prim]
        for arg_type in prim.args:
            expr.extend(generate(arg_type, 0))  # Force termination
        return expr
```

**Why filter out recursive primitives?**

If we picked a primitive that takes `type_` as input, we'd recurse forever:

```python
# BAD: sum_prediction_2 takes Prediction inputs!
# Prediction → sum_prediction_2(Prediction, Prediction)
#                                    ↓            ↓
#                     sum_prediction_2(...)  sum_prediction_2(...)
#                          ↓     ↓                 ↓     ↓
#                          ...  ...               ...   ...
# INFINITE LOOP!
```

**Safe primitives** don't take `type_` as input:
```python
# For Prediction type, safe primitives:
# - rf_classification(FeatureVector, Trees, Depth)  ✅ No Prediction input
# - lr_classification(FeatureVector)                ✅ No Prediction input

# Unsafe primitives:
# - sum_prediction_2(Prediction, Prediction)        ❌ Takes Prediction!
```

---

## Step-by-Step Execution Examples

### Example 1: Simple Tree Generation

```python
# Goal: Generate a tree of type Prediction with max_depth=2

# Call: gen_safe(pset, min_=1, max_=2, type_=Prediction)

# Step 1: generate(Prediction, depth=2)
#   - terminals = [] (none for Prediction)
#   - primitives = [rf_classification, lr_classification, ...]
#   - depth > 0, no terminals → pick primitive
#   - Randomly pick: lr_classification(FeatureVector)
#   - expr = [lr_classification]
#   - Recurse for FeatureVector with depth=1

# Step 2: generate(FeatureVector, depth=1)
#   - terminals = [] (none for FeatureVector)
#   - primitives = [hog_features, lbp_features, ...]
#   - depth > 0, no terminals → pick primitive
#   - Randomly pick: hog_features(Channel)
#   - expr = [hog_features]
#   - Recurse for Channel with depth=0

# Step 3: generate(Channel, depth=0)
#   - terminals = [GetRed, GetGreen, GetBlue, GetGray]
#   - depth <= 0, have terminals → pick terminal
#   - Randomly pick: GetGray
#   - return [GetGray]

# Unwinding:
# Step 2 result: [hog_features, GetGray]
# Step 1 result: [lr_classification, hog_features, GetGray]

# Final tree:
# lr_classification(hog_features(GetGray))
```

### Example 2: Tree with Parameters

```python
# Goal: Generate rf_classification with Trees and Depth

# Step 1: generate(Prediction, depth=2)
#   - Pick: rf_classification(FeatureVector, Trees, Depth)
#   - expr = [rf_classification]
#   - Recurse for FeatureVector (depth=1)
#   - Recurse for Trees (depth=1)
#   - Recurse for Depth (depth=1)

# Step 2a: generate(FeatureVector, depth=1)
#   - Pick: histogram_features(Channel)
#   - Recurse for Channel (depth=0)
#   - → [histogram_features, GetRed]

# Step 2b: generate(Trees, depth=1)
#   - terminals = [<ephemeral class>]
#   - depth > 0, have terminals, choose terminal
#   - Pick terminal, it's a class → instantiate
#   - t_instance = TreesEphemeral()  # value: 250
#   - return [t_instance]

# Step 2c: generate(Depth, depth=1)
#   - Similar to Trees
#   - return [d_instance]  # value: 40

# Final tree:
# rf_classification(histogram_features(GetRed), 250, 40)
```

### Example 3: Forced Termination

```python
# Goal: What happens when we need Prediction at depth=0?

# generate(Prediction, depth=0)
#   - terminals = [] (none!)
#   - primitives = [rf_classification, lr_classification, sum_prediction_2, ...]
#   - depth <= 0, NO terminals → find safe primitive
#   
#   - Filter primitives where Prediction not in args:
#     - rf_classification(FeatureVector, Trees, Depth) ✅
#     - lr_classification(FeatureVector) ✅
#     - sum_prediction_2(Prediction, Prediction) ❌ REMOVED
#     
#   - safe_prims = [rf_classification, lr_classification, ...]
#   - Pick: lr_classification
#   - expr = [lr_classification]
#   - Recurse for FeatureVector with depth=0 (forced termination!)

# generate(FeatureVector, depth=0)
#   - terminals = [] (none!)
#   - primitives = [hog_features, concat_features_2, ...]
#   - depth <= 0, NO terminals → find safe primitive
#   
#   - Filter: hog_features(Channel) ✅, concat_features_2(FV, FV) ❌
#   - safe_prims = [hog_features, lbp_features, ...]
#   - Pick: hog_features
#   - Recurse for Channel with depth=0

# generate(Channel, depth=0)
#   - terminals = [GetRed, GetGreen, GetBlue, GetGray] ✅
#   - Pick terminal: GetBlue
#   - return [GetBlue]

# Final tree:
# lr_classification(hog_features(GetBlue))
```

---

## Edge Cases and Error Handling

### Case 1: Type with No Terminals AND No Safe Primitives

```python
# Hypothetical: What if a type had no terminals and all primitives were recursive?

safe_prims = [p for p in primitives if type_ not in p.args]

if not safe_prims:
    raise ValueError(f"Cannot terminate type {type_}: No terminals and no safe primitives.")
```

This can't happen in our system because:
- `Prediction` has `rf_classification`, `lr_classification` (safe)
- `FeatureVector` has `hog_features`, `lbp_features`, etc. (safe)

### Case 2: Empty Primitive Set

```python
if not terminals:
    if not primitives:
        raise ValueError(f"No primitives or terminals for type {type_}")
```

Guards against misconfigured primitive sets.

### Case 3: Grow vs. Full Behavior

The algorithm uses genGrow-like behavior:

```python
n_terms = len(terminals)
n_prims = len(primitives)

if n_prims == 0 or random.random() < (n_terms / (n_terms + n_prims)):
    return pick_terminal()
else:
    return pick_primitive_and_recurse(depth)
```

This means:
- Probability of terminal = `n_terminals / (n_terminals + n_primitives)`
- More primitives → more likely to grow
- Avoids always-minimal or always-maximal trees

---

## Complete Code Walkthrough

```python
import random
from deap import gp
import inspect
```

**Lines 1-3**: Import random for stochastic choices, DEAP's gp module, and inspect for checking ephemeral constants.

---

```python
def gen_safe(pset, min_, max_, type_):
    """
    Generates a tree ensuring that we can always terminate even if the root type
    has no direct terminals (by falling back to non-recursive primitives).
    
    This handles the case where types like Ephemeral have no terminal but
    can be produced by primitives like rf_classification.
    """
```

**Lines 5-11**: Function definition with docstring. Note: `min_` is mentioned but not strictly used (it's for API compatibility with DEAP).

---

```python
    def generate(type_, depth):
        # Check if we have terminals and primitives for this type
        terminals = pset.terminals.get(type_, [])
        primitives = pset.primitives.get(type_, [])
```

**Lines 12-15**: The recursive inner function. Gets available terminals and primitives for the current type.

---

```python
        def pick_terminal():
            """Helper to properly instantiate a terminal (handles ephemeral constants)."""
            term = random.choice(terminals)
            # Ephemeral constants in DEAP are classes that inherit from a special base
            # They have a 'func' attribute and need to be instantiated
            if inspect.isclass(term):
                # This is an ephemeral constant class - instantiate it
                return [term()]
            else:
                # Regular terminal instance
                return [term]
```

**Lines 17-27**: Helper to pick and instantiate a terminal. The check for `isclass` distinguishes ephemeral constants (which are classes) from regular terminals (which are instances).

---

```python
        def pick_primitive_and_recurse(available_depth):
            """Helper to pick a primitive and recursively generate its arguments."""
            prim = random.choice(primitives)
            expr = [prim]
            for arg_type in prim.args:
                expr.extend(generate(arg_type, available_depth - 1))
            return expr
```

**Lines 29-35**: Helper to pick a primitive and build its subtree. The expression starts with the primitive, then extends with each recursively generated argument.

---

```python
        # If we MUST stop (depth limit reached or below)
        if depth <= 0:
            if terminals:
                return pick_terminal()
            else:
                # No terminals - must pick a non-recursive primitive
                # Filter for primitives that don't take 'type_' as argument
                safe_prims = [p for p in primitives if type_ not in p.args]
                
                if not safe_prims:
                    raise ValueError(f"Cannot terminate type {type_}: No terminals and no safe primitives.")
                
                prim = random.choice(safe_prims)
                expr = [prim]
                for arg_type in prim.args:
                    # Force termination for children
                    expr.extend(generate(arg_type, 0))
                return expr
```

**Lines 37-53**: The forced termination case. When depth is exhausted:
1. If terminals exist, use one
2. Otherwise, find a "safe" primitive (one that doesn't take this type as input)
3. Generate its children with depth=0 (forcing them to terminate too)

---

```python
        # If we can continue growing
        else:
            if not terminals:
                # No terminals for this type - must pick primitive
                if not primitives:
                    raise ValueError(f"No primitives or terminals for type {type_}")
                return pick_primitive_and_recurse(depth)
            else:
                # We have terminals - choose between terminal and primitive
                # Use genGrow-like behavior: uniform probability between all options
                n_terms = len(terminals)
                n_prims = len(primitives)
                
                if n_prims == 0 or random.random() < (n_terms / (n_terms + n_prims)):
                    return pick_terminal()
                else:
                    return pick_primitive_and_recurse(depth)
```

**Lines 55-70**: The normal growth case. When we still have depth remaining:
1. If no terminals, must use a primitive
2. If we have terminals, probabilistically choose between terminal and primitive
3. The probability formula gives equal weight to each option (terminal or primitive)

---

```python
    return generate(type_, max_)
```

**Line 72**: Start generation at the specified type with maximum depth.

---

## How It's Used

### In ExperimentRunner

```python
# In gp_experiment.py
self.toolbox.register("expr", gp_utils.gen_safe, pset=pset, min_=2, max_=5, type_=gp_types.Prediction)
self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
```

This tells DEAP:
1. To generate expressions using `gen_safe`
2. With minimum depth 2, maximum depth 5
3. Root type is `Prediction`
4. Wrap the expression in an `Individual`

### For Mutation

```python
self.toolbox.register("expr_mut", gp_utils.gen_safe, min_=0, max_=2)
self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=pset)
```

Mutation uses smaller trees (max depth 2) to replace subtrees.

---

## Summary

The `gp_utils.py` module provides a critical utility function:

| Function | Purpose |
|----------|---------|
| `gen_safe` | Generate valid trees even when types have no terminals |

Key features:
1. **Safe termination**: Falls back to non-recursive primitives when needed
2. **Ephemeral handling**: Correctly instantiates ephemeral constant classes
3. **Probabilistic growth**: Uses genGrow-like behavior for tree shape diversity
4. **Error handling**: Clear errors for impossible type configurations

Without this custom generator, DEAP couldn't create trees for our type system where `Prediction` and `FeatureVector` have no terminals.
