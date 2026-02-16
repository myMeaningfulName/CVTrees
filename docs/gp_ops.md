# GP Operations Module (`gp_ops.py`) - Complete Documentation

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
