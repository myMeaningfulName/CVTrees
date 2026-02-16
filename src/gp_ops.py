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
from .gp_types import (
    Image, Channel, FeatureVector, Prediction, EnsembleOutput, Weight, Sigma, Trees, Depth, 
    Frequency, Theta, Batch, Order,
    # Extended parameter types
    PixelsPerCell, CellsPerBlock, Orientations, LBP_P, LBP_R, KernelSize,
    NumKeypoints, Threshold, MorphKernel, CannyLow, CannyHigh
)
from .gp_context import context, ExecutionMode

# --- Helper for Batch Processing ---

def _ensure_float32(data: np.ndarray) -> np.ndarray:
    """Ensure data is float32 to avoid issues with cv2 and other libraries."""
    if data.dtype != np.float32:
        return data.astype(np.float32)
    return data

def map_batches(input_wrapper, func, output_type):
    """
    Helper to create a new DataWrapper that applies func to each batch of input_wrapper.
    """
    def iterator_factory():
        for batch in input_wrapper:
            result = func(batch.data)
            # Ensure Channel outputs are float32 to avoid dtype issues
            if output_type == Channel and result.dtype != np.float32:
                result = result.astype(np.float32)
            yield Batch(result)
    return output_type(iterator_factory)

def map_batches_2(input1, input2, func, output_type):
    def iterator_factory():
        for b1, b2 in zip(input1, input2):
            yield Batch(func(b1.data, b2.data))
    return output_type(iterator_factory)

def map_batches_3(input1, input2, input3, func, output_type):
    def iterator_factory():
        for b1, b2, b3 in zip(input1, input2, input3):
            yield Batch(func(b1.data, b2.data, b3.data))
    return output_type(iterator_factory)

def map_batches_4(input1, input2, input3, input4, func, output_type):
    def iterator_factory():
        for b1, b2, b3, b4 in zip(input1, input2, input3, input4):
            yield Batch(func(b1.data, b2.data, b3.data, b4.data))
    return output_type(iterator_factory)

# --- Input Nodes ---

_CURRENT_IMAGE = None

def set_image(image: Image):
    global _CURRENT_IMAGE
    _CURRENT_IMAGE = image

def _get_red_iter():
    if _CURRENT_IMAGE is None: raise ValueError("Image not set")
    for batch in _CURRENT_IMAGE:
        # batch.data is List[np.ndarray] (channels) or (N, C, H, W)
        # Assuming List[np.ndarray] where index 0 is Red
        if isinstance(batch.data, list):
            yield Batch(batch.data[0])
        else:
            yield Batch(batch.data[:, 0])

def _get_green_iter():
    if _CURRENT_IMAGE is None: raise ValueError("Image not set")
    for batch in _CURRENT_IMAGE:
        if isinstance(batch.data, list):
            yield Batch(batch.data[1])
        else:
            yield Batch(batch.data[:, 1])

def _get_blue_iter():
    if _CURRENT_IMAGE is None: raise ValueError("Image not set")
    for batch in _CURRENT_IMAGE:
        if isinstance(batch.data, list):
            yield Batch(batch.data[2])
        else:
            yield Batch(batch.data[:, 2])

def _get_gray_iter():
    if _CURRENT_IMAGE is None: raise ValueError("Image not set")
    for batch in _CURRENT_IMAGE:
        if isinstance(batch.data, list):
            if len(batch.data) > 3:
                yield Batch(batch.data[3])
            else:
                r, g, b = batch.data[0], batch.data[1], batch.data[2]
                yield Batch(0.299 * r + 0.587 * g + 0.114 * b)
        else:
            if batch.data.shape[1] > 3:
                yield Batch(batch.data[:, 3])
            else:
                r, g, b = batch.data[:, 0], batch.data[:, 1], batch.data[:, 2]
                yield Batch(0.299 * r + 0.587 * g + 0.114 * b)

# Terminals (Lazy Channels)
RedTerminal = Channel(_get_red_iter)
GreenTerminal = Channel(_get_green_iter)
BlueTerminal = Channel(_get_blue_iter)
GrayTerminal = Channel(_get_gray_iter)

def linear_combination(c1: Channel, c2: Channel, c3: Channel, w1: Weight, w2: Weight, w3: Weight) -> Channel:
    def func(d1, d2, d3):
        # Handle potential size mismatches from pooling operations
        # Resize all to the smallest common size
        h1, w1_img = d1.shape[1], d1.shape[2]
        h2, w2_img = d2.shape[1], d2.shape[2]
        h3, w3_img = d3.shape[1], d3.shape[2]
        
        target_h = min(h1, h2, h3)
        target_w = min(w1_img, w2_img, w3_img)
        
        # Resize using cv2 if sizes don't match
        if (h1, w1_img) != (target_h, target_w):
            d1_resized = np.zeros((d1.shape[0], target_h, target_w), dtype=d1.dtype)
            for i in range(d1.shape[0]):
                d1_resized[i] = cv2.resize(d1[i], (target_w, target_h))
            d1 = d1_resized
        
        if (h2, w2_img) != (target_h, target_w):
            d2_resized = np.zeros((d2.shape[0], target_h, target_w), dtype=d2.dtype)
            for i in range(d2.shape[0]):
                d2_resized[i] = cv2.resize(d2[i], (target_w, target_h))
            d2 = d2_resized
        
        if (h3, w3_img) != (target_h, target_w):
            d3_resized = np.zeros((d3.shape[0], target_h, target_w), dtype=d3.dtype)
            for i in range(d3.shape[0]):
                d3_resized[i] = cv2.resize(d3[i], (target_w, target_h))
            d3 = d3_resized
        
        return w1 * d1 + w2 * d2 + w3 * d3
    return map_batches_3(c1, c2, c3, func, Channel)

# --- Image Filtering Operations ---

def mean_filter(channel: Channel) -> Channel:
    def func(data):
        return ndimage.uniform_filter(data, size=(1, 3, 3))
    return map_batches(channel, func, Channel)

def median_filter(channel: Channel) -> Channel:
    def func(data):
        return ndimage.median_filter(data, size=(1, 3, 3))
    return map_batches(channel, func, Channel)

def min_filter(channel: Channel) -> Channel:
    def func(data):
        return ndimage.minimum_filter(data, size=(1, 3, 3))
    return map_batches(channel, func, Channel)

def max_filter(channel: Channel) -> Channel:
    def func(data):
        return ndimage.maximum_filter(data, size=(1, 3, 3))
    return map_batches(channel, func, Channel)

def gaussian_filter(channel: Channel, sigma: Sigma) -> Channel:
    def func(data):
        return ndimage.gaussian_filter(data, sigma=(0, sigma, sigma))
    return map_batches(channel, func, Channel)

def laplacian_filter(channel: Channel) -> Channel:
    def func(data):
        return ndimage.laplace(data)
    return map_batches(channel, func, Channel)

def sobel_filter(channel: Channel) -> Channel:
    def func(data):
        sx = ndimage.sobel(data, axis=1)
        sy = ndimage.sobel(data, axis=2)
        return np.hypot(sx, sy)
    return map_batches(channel, func, Channel)

def relu(channel: Channel) -> Channel:
    def func(data):
        return np.maximum(0, data)
    return map_batches(channel, func, Channel)

def sqrt_op(channel: Channel) -> Channel:
    def func(data):
        d = data.copy()
        neg_mask = d < 0
        d[neg_mask] = 1
        d[~neg_mask] = np.sqrt(d[~neg_mask])
        return d
    return map_batches(channel, func, Channel)

def add_max_pool(c1: Channel, c2: Channel) -> Channel:
    def func(d1, d2):
        p1 = skimage.measure.block_reduce(d1, (1, 2, 2), np.max)
        p2 = skimage.measure.block_reduce(d2, (1, 2, 2), np.max)
        min_h = min(p1.shape[1], p2.shape[1])
        min_w = min(p1.shape[2], p2.shape[2])
        return p1[:, :min_h, :min_w] + p2[:, :min_h, :min_w]
    return map_batches_2(c1, c2, func, Channel)

def sub_max_pool(c1: Channel, c2: Channel) -> Channel:
    def func(d1, d2):
        p1 = skimage.measure.block_reduce(d1, (1, 2, 2), np.max)
        p2 = skimage.measure.block_reduce(d2, (1, 2, 2), np.max)
        min_h = min(p1.shape[1], p2.shape[1])
        min_w = min(p1.shape[2], p2.shape[2])
        return p1[:, :min_h, :min_w] - p2[:, :min_h, :min_w]
    return map_batches_2(c1, c2, func, Channel)

def gaud_filter(channel: Channel, sigma: Sigma, o1: Order, o2: Order) -> Channel:
    def func(data):
        # Gaussian derivatives. Order is (batch_order, y_order, x_order)
        # We don't want derivative across batch, so 0.
        return ndimage.gaussian_filter(data, sigma=(0, sigma, sigma), order=(0, o1, o2))
    return map_batches(channel, func, Channel)

def log1_filter(channel: Channel) -> Channel:
    def func(data):
        # Laplacian of Gaussian with sigma=1
        return ndimage.gaussian_laplace(data, sigma=(0, 1, 1))
    return map_batches(channel, func, Channel)

def log2_filter(channel: Channel) -> Channel:
    def func(data):
        # Laplacian of Gaussian with sigma=2
        return ndimage.gaussian_laplace(data, sigma=(0, 2, 2))
    return map_batches(channel, func, Channel)

def gabor_filter(channel: Channel, theta: Theta, frequency: Frequency) -> Channel:
    def func(data):
        # Gabor filter. skimage.filters.gabor works on 2D images.
        # We need to apply it to each image in the batch.
        # data shape: (N, H, W)
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            # gabor returns (real, imag)
            filt_real, filt_imag = gabor(data[i], frequency=frequency, theta=theta)
            # We use magnitude
            out[i] = np.hypot(filt_real, filt_imag)
        return out
    return map_batches(channel, func, Channel)

def hog_filter(channel: Channel) -> Channel:
    def func(data):
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            try:
                # visualize=True returns (fd, hog_image)
                _, hog_img = hog(data[i], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
                out[i] = hog_img
            except:
                pass # Keep zero if fails
        return out
    return map_batches(channel, func, Channel)

def lbp_filter(channel: Channel) -> Channel:
    def func(data):
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            # local_binary_pattern returns the LBP image
            out[i] = local_binary_pattern(data[i], P=8, R=1.5, method='uniform')
        return out
    return map_batches(channel, func, Channel)

# --- Feature Extraction Operations ---

def histogram_features(channel: Channel) -> FeatureVector:
    def func(data):
        feats = []
        for img in data:
            hist, _ = np.histogram(img, bins=256, range=(0, 1))
            feats.append(hist)
        return np.array(feats)
    return map_batches(channel, func, FeatureVector)

def hog_features(channel: Channel) -> FeatureVector:
    def func(data):
        feats = []
        for img in data:
            try:
                # Updated to 4x4 pixels per cell as per requirements
                fd = hog(img, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2))
                feats.append(fd)
            except:
                feats.append(np.zeros(10))
        return np.array(feats)
    return map_batches(channel, func, FeatureVector)

def lbp_features(channel: Channel) -> FeatureVector:
    def func(data):
        feats = []
        for img in data:
            lbp = local_binary_pattern(img, P=8, R=1.5, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10 + 3), range=(0, 10 + 2))
            feats.append(hist)
        return np.array(feats)
    return map_batches(channel, func, FeatureVector)

def concat_images(c1: Channel, c2: Channel) -> FeatureVector:
    def func(d1, d2):
        f1 = d1.reshape(d1.shape[0], -1)
        f2 = d2.reshape(d2.shape[0], -1)
        return np.hstack([f1, f2])
    return map_batches_2(c1, c2, func, FeatureVector)

def sift_features(channel: Channel) -> FeatureVector:
    def func(data):
        feats = []
        # SIFT requires uint8
        sift = cv2.SIFT_create()
        for img in data:
            # Normalize to 0-255 and convert to uint8
            img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            
            # Dense SIFT: compute descriptors on a grid
            step_size = 8
            kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img_uint8.shape[0], step_size) 
                                                for x in range(0, img_uint8.shape[1], step_size)]
            
            if not kp:
                feats.append(np.zeros(128))
                continue
                
            _, des = sift.compute(img_uint8, kp)
            
            if des is None:
                feats.append(np.zeros(128))
            else:
                # Average descriptors to get 128 features
                feats.append(np.mean(des, axis=0))
        return np.array(feats)
    return map_batches(channel, func, FeatureVector)

# --- FE Wrappers (Filter + Flatten) ---

def _flatten_helper(data):
    return data.reshape(data.shape[0], -1)

def hog_image_features(channel: Channel) -> FeatureVector:
    # HOG FE: Generate HOG image and flatten
    # We can reuse hog_filter logic but we need to compose it.
    # Since we are inside ops, we can just call the implementation of hog_filter's func
    # But cleaner to just use the filter and map a flatten.
    # However, we need to return a FeatureVector.
    # Let's implement directly to avoid overhead of creating intermediate Channel objects if possible,
    # but reusing code is better.
    
    # Re-implementing logic to ensure single pass
    def func(data):
        out = []
        for i in range(data.shape[0]):
            try:
                _, hog_img = hog(data[i], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
                out.append(hog_img.flatten())
            except:
                out.append(np.zeros(data.shape[1]*data.shape[2])) # Fallback size?
        return np.array(out)
    return map_batches(channel, func, FeatureVector)

def lbp_image_features(channel: Channel) -> FeatureVector:
    def func(data):
        out = []
        for i in range(data.shape[0]):
            lbp = local_binary_pattern(data[i], P=8, R=1.5, method='uniform')
            out.append(lbp.flatten())
        return np.array(out)
    return map_batches(channel, func, FeatureVector)

def sobel_features(channel: Channel) -> FeatureVector:
    def func(data):
        sx = ndimage.sobel(data, axis=1)
        sy = ndimage.sobel(data, axis=2)
        s = np.hypot(sx, sy)
        return s.reshape(s.shape[0], -1)
    return map_batches(channel, func, FeatureVector)

def gabor_features(channel: Channel, theta: Theta, frequency: Frequency) -> FeatureVector:
    def func(data):
        out = []
        for i in range(data.shape[0]):
            filt_real, filt_imag = gabor(data[i], frequency=frequency, theta=theta)
            out.append(np.hypot(filt_real, filt_imag).flatten())
        return np.array(out)
    return map_batches(channel, func, FeatureVector)

def gaussian_features(channel: Channel, sigma: Sigma) -> FeatureVector:
    def func(data):
        g = ndimage.gaussian_filter(data, sigma=(0, sigma, sigma))
        return g.reshape(g.shape[0], -1)
    return map_batches(channel, func, FeatureVector)

def gaud_features(channel: Channel, sigma: Sigma, o1: Order, o2: Order) -> FeatureVector:
    def func(data):
        g = ndimage.gaussian_filter(data, sigma=(0, sigma, sigma), order=(0, o1, o2))
        return g.reshape(g.shape[0], -1)
    return map_batches(channel, func, FeatureVector)

# --- Feature Concatenation ---

def concat_features_2(f1: FeatureVector, f2: FeatureVector) -> FeatureVector:
    def func(d1, d2):
        return np.hstack([d1, d2])
    return map_batches_2(f1, f2, func, FeatureVector)

def concat_features_3(f1: FeatureVector, f2: FeatureVector, f3: FeatureVector) -> FeatureVector:
    def func(d1, d2, d3):
        return np.hstack([d1, d2, d3])
    return map_batches_3(f1, f2, f3, func, FeatureVector)

def concat_features_4(f1: FeatureVector, f2: FeatureVector, f3: FeatureVector, f4: FeatureVector) -> FeatureVector:
    def func(d1, d2, d3, d4):
        return np.hstack([d1, d2, d3, d4])
    return map_batches_4(f1, f2, f3, f4, func, FeatureVector)

# --- Classification Heads (Stateful) ---

def rf_classification(features: FeatureVector, t: Trees, d: Depth) -> Prediction:
    node_id = context.get_next_node_id()
    
    def iterator_factory():
        if context.mode == ExecutionMode.TRAIN:
            X_list = []
            for batch in features:
                X_list.append(batch.data)
            if not X_list: return
            X = np.vstack(X_list)
            y = context.train_labels
            
            clf = RandomForestClassifier(n_estimators=t, max_depth=d, n_jobs=1)
            try:
                probs = cross_val_predict(clf, X, y, cv=3, method='predict_proba')
                clf.fit(X, y)
            except:
                clf.fit(X, y)
                probs = clf.predict_proba(X)
            
            context.save_model(node_id, clf)
            yield Batch(probs)
        else:
            clf = context.get_model(node_id)
            if clf:
                for batch in features:
                    yield Batch(clf.predict_proba(batch.data))
            else:
                for batch in features:
                    yield Batch(np.zeros((batch.data.shape[0], 10)))

    return Prediction(iterator_factory)

def erf_classification(features: FeatureVector, t: Trees, d: Depth) -> Prediction:
    node_id = context.get_next_node_id()
    
    def iterator_factory():
        if context.mode == ExecutionMode.TRAIN:
            X_list = []
            for batch in features:
                X_list.append(batch.data)
            if not X_list: return
            X = np.vstack(X_list)
            y = context.train_labels
            
            clf = ExtraTreesClassifier(n_estimators=t, max_depth=d, n_jobs=1)
            try:
                probs = cross_val_predict(clf, X, y, cv=3, method='predict_proba')
                clf.fit(X, y)
            except:
                clf.fit(X, y)
                probs = clf.predict_proba(X)
            
            context.save_model(node_id, clf)
            yield Batch(probs)
        else:
            clf = context.get_model(node_id)
            if clf:
                for batch in features:
                    yield Batch(clf.predict_proba(batch.data))
            else:
                for batch in features:
                    yield Batch(np.zeros((batch.data.shape[0], 10)))

    return Prediction(iterator_factory)

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
            
            clf = LogisticRegression(max_iter=1000)
            try:
                probs = cross_val_predict(clf, X, y, cv=3, method='predict_proba')
                clf.fit(X, y)
            except:
                clf.fit(X, y)
                probs = clf.predict_proba(X)
            
            context.save_model(node_id, clf)
            yield Batch(probs)
        else:
            clf = context.get_model(node_id)
            if clf:
                for batch in features:
                    yield Batch(clf.predict_proba(batch.data))
            else:
                for batch in features:
                    yield Batch(np.zeros((batch.data.shape[0], 10)))

    return Prediction(iterator_factory)

def svm_classification(features: FeatureVector) -> Prediction:
    node_id = context.get_next_node_id()
    
    def iterator_factory():
        if context.mode == ExecutionMode.TRAIN:
            X_list = []
            for batch in features:
                X_list.append(batch.data)
            if not X_list: return
            X = np.vstack(X_list)
            y = context.train_labels
            
            clf = SVC(probability=True)
            try:
                probs = cross_val_predict(clf, X, y, cv=3, method='predict_proba')
                clf.fit(X, y)
            except:
                clf.fit(X, y)
                probs = clf.predict_proba(X)
            
            context.save_model(node_id, clf)
            yield Batch(probs)
        else:
            clf = context.get_model(node_id)
            if clf:
                for batch in features:
                    yield Batch(clf.predict_proba(batch.data))
            else:
                print(f"WARNING: SVM Model not found for node {node_id} in EVAL mode.")
                for batch in features:
                    yield Batch(np.zeros((batch.data.shape[0], 10)))

    return Prediction(iterator_factory)

# --- Ensemble Output (Final Layer) ---
# These operations produce EnsembleOutput, which is the ROOT type of all legal trees.
# This ensures that summation/ensemble operations can ONLY occur at the top of the tree.
# Nothing consumes EnsembleOutput, so you can't nest summations or add classifiers after.

def ensemble_single(p: Prediction) -> EnsembleOutput:
    """
    Wraps a single classifier's prediction as the final ensemble output.
    Use this when you have only one classification head (no summation needed).
    """
    def iterator_factory():
        for batch in p:
            yield Batch(batch.data)
    return EnsembleOutput(iterator_factory)

def ensemble_sum_2(p1: Prediction, p2: Prediction) -> EnsembleOutput:
    """
    Sums two classifiers' predictions and produces the final ensemble output.
    This is the ensemble summation layer.
    """
    def func(d1, d2):
        return d1 + d2
    return map_batches_2(p1, p2, func, EnsembleOutput)

def ensemble_sum_3(p1: Prediction, p2: Prediction, p3: Prediction) -> EnsembleOutput:
    """
    Sums three classifiers' predictions and produces the final ensemble output.
    This is the ensemble summation layer.
    """
    def func(d1, d2, d3):
        return d1 + d2 + d3
    return map_batches_3(p1, p2, p3, func, EnsembleOutput)

# --- Legacy Summation (kept for backwards compatibility, but not registered in pset) ---
# These return Prediction instead of EnsembleOutput, which allowed nested summations.
# They are deprecated and should not be used in new code.

def sum_prediction_2(p1: Prediction, p2: Prediction) -> Prediction:
    """DEPRECATED: Use ensemble_sum_2 instead. This allows illegal nested summations."""
    def func(d1, d2):
        return d1 + d2
    return map_batches_2(p1, p2, func, Prediction)

def sum_prediction_3(p1: Prediction, p2: Prediction, p3: Prediction) -> Prediction:
    """DEPRECATED: Use ensemble_sum_3 instead. This allows illegal nested summations."""
    def func(d1, d2, d3):
        return d1 + d2 + d3
    return map_batches_3(p1, p2, p3, func, Prediction)

# --- Cascade ---

def cascade_rf(features: FeatureVector, t: Trees, d: Depth) -> FeatureVector:
    pred = rf_classification(features, t, d)
    return _cascade_helper(features, pred)

def cascade_erf(features: FeatureVector, t: Trees, d: Depth) -> FeatureVector:
    pred = erf_classification(features, t, d)
    return _cascade_helper(features, pred)

def cascade_lr(features: FeatureVector) -> FeatureVector:
    pred = lr_classification(features)
    return _cascade_helper(features, pred)

def cascade_svm(features: FeatureVector) -> FeatureVector:
    pred = svm_classification(features)
    return _cascade_helper(features, pred)

def _cascade_helper(features: FeatureVector, pred: Prediction) -> FeatureVector:
    """
    Helper for cascade classifiers - concatenates predictions with original features.
    This allows the next layer to use both the original features and the classifier's output.
    
    IMPORTANT: In TRAIN mode, classifiers aggregate all data into one batch for cross-validation.
    We need to re-batch the output to match the original input batch structure, otherwise
    downstream operations like concat_features_* will fail due to mismatched batch sizes.
    """
    def iterator_factory():
        # First, collect all features and their batch sizes to know the original structure
        feats_batches = []
        batch_sizes = []
        for b in features:
            feats_batches.append(b.data)
            batch_sizes.append(b.data.shape[0])
        
        if not feats_batches:
            return
        
        # Concatenate all features
        all_feats = np.vstack(feats_batches)
        
        # Get the single prediction batch (in TRAIN mode, classifiers yield one big batch)
        all_preds = []
        for p_batch in pred:
            all_preds.append(p_batch.data)
        
        if not all_preds:
            return
        
        all_preds = np.vstack(all_preds)
        
        # Concatenate features with predictions
        cascaded = np.hstack([all_feats, all_preds])
        
        # Re-batch to match original structure
        idx = 0
        for size in batch_sizes:
            yield Batch(cascaded[idx:idx + size])
            idx += size

    return FeatureVector(iterator_factory)

# --- Identity Functions (kept for potential future use, but NOT registered in pset) ---
def identity_trees(t: Trees) -> Trees: return t
def identity_depth(d: Depth) -> Depth: return d
def identity_sigma(s: Sigma) -> Sigma: return s
def identity_frequency(f: Frequency) -> Frequency: return f
def identity_theta(t: Theta) -> Theta: return t
def identity_weight(w: Weight) -> Weight: return w
def identity_order(o: Order) -> Order: return o

# NOTE: DefaultHOG and get_default_rf have been removed.
# They were causing degenerate trees by allowing FeatureVector and Prediction
# terminals that bypass the proper Channel -> FeatureVector -> Prediction pipeline.


# =============================================================================
# EXTENDED OPERATIONS (for extended_ops mode)
# =============================================================================

# --- Extended Image Filtering Operations ---

def prewitt_filter(channel: Channel) -> Channel:
    """Prewitt edge detection filter."""
    def func(data):
        px = ndimage.prewitt(data, axis=1)
        py = ndimage.prewitt(data, axis=2)
        return np.hypot(px, py)
    return map_batches(channel, func, Channel)

def scharr_filter(channel: Channel) -> Channel:
    """Scharr edge detection filter (better rotation invariance than Sobel)."""
    def func(data):
        # Scharr kernels
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            sx = cv2.Scharr(data[i].astype(np.float32), cv2.CV_32F, 1, 0)
            sy = cv2.Scharr(data[i].astype(np.float32), cv2.CV_32F, 0, 1)
            out[i] = np.hypot(sx, sy)
        return out
    return map_batches(channel, func, Channel)

def canny_filter(channel: Channel) -> Channel:
    """Canny edge detection with fixed thresholds."""
    def func(data):
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            img_uint8 = cv2.normalize(data[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            edges = cv2.Canny(img_uint8, 50, 150)
            out[i] = edges.astype(np.float32) / 255.0
        return out
    return map_batches(channel, func, Channel)

def bilateral_filter(channel: Channel, sigma: Sigma) -> Channel:
    """Bilateral filter for edge-preserving smoothing."""
    def func(data):
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            img_uint8 = cv2.normalize(data[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            filtered = cv2.bilateralFilter(img_uint8, d=5, sigmaColor=sigma*25, sigmaSpace=sigma*25)
            out[i] = filtered.astype(np.float32) / 255.0
        return out
    return map_batches(channel, func, Channel)

def clahe_filter(channel: Channel) -> Channel:
    """Contrast Limited Adaptive Histogram Equalization."""
    def func(data):
        out = np.zeros_like(data, dtype=np.float32)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for i in range(data.shape[0]):
            img_uint8 = cv2.normalize(data[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            out[i] = clahe.apply(img_uint8).astype(np.float32) / 255.0
        return out
    return map_batches(channel, func, Channel)

def erosion_filter(channel: Channel) -> Channel:
    """Morphological erosion with 3x3 kernel."""
    def func(data):
        kernel = np.ones((3, 3), np.uint8)
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            img_uint8 = cv2.normalize(data[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            out[i] = cv2.erode(img_uint8, kernel).astype(np.float32) / 255.0
        return out
    return map_batches(channel, func, Channel)

def dilation_filter(channel: Channel) -> Channel:
    """Morphological dilation with 3x3 kernel."""
    def func(data):
        kernel = np.ones((3, 3), np.uint8)
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            img_uint8 = cv2.normalize(data[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            out[i] = cv2.dilate(img_uint8, kernel).astype(np.float32) / 255.0
        return out
    return map_batches(channel, func, Channel)

def opening_filter(channel: Channel) -> Channel:
    """Morphological opening (erosion followed by dilation)."""
    def func(data):
        kernel = np.ones((3, 3), np.uint8)
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            img_uint8 = cv2.normalize(data[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            out[i] = cv2.morphologyEx(img_uint8, cv2.MORPH_OPEN, kernel).astype(np.float32) / 255.0
        return out
    return map_batches(channel, func, Channel)

def closing_filter(channel: Channel) -> Channel:
    """Morphological closing (dilation followed by erosion)."""
    def func(data):
        kernel = np.ones((3, 3), np.uint8)
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            img_uint8 = cv2.normalize(data[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            out[i] = cv2.morphologyEx(img_uint8, cv2.MORPH_CLOSE, kernel).astype(np.float32) / 255.0
        return out
    return map_batches(channel, func, Channel)

def gradient_filter(channel: Channel) -> Channel:
    """Morphological gradient (difference between dilation and erosion)."""
    def func(data):
        kernel = np.ones((3, 3), np.uint8)
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            img_uint8 = cv2.normalize(data[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            out[i] = cv2.morphologyEx(img_uint8, cv2.MORPH_GRADIENT, kernel).astype(np.float32) / 255.0
        return out
    return map_batches(channel, func, Channel)

def tophat_filter(channel: Channel) -> Channel:
    """Top hat filter (difference between input and opening)."""
    def func(data):
        kernel = np.ones((5, 5), np.uint8)
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            img_uint8 = cv2.normalize(data[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            out[i] = cv2.morphologyEx(img_uint8, cv2.MORPH_TOPHAT, kernel).astype(np.float32) / 255.0
        return out
    return map_batches(channel, func, Channel)

def blackhat_filter(channel: Channel) -> Channel:
    """Black hat filter (difference between closing and input)."""
    def func(data):
        kernel = np.ones((5, 5), np.uint8)
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            img_uint8 = cv2.normalize(data[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            out[i] = cv2.morphologyEx(img_uint8, cv2.MORPH_BLACKHAT, kernel).astype(np.float32) / 255.0
        return out
    return map_batches(channel, func, Channel)

def sharpen_filter(channel: Channel) -> Channel:
    """Sharpening filter using unsharp masking."""
    def func(data):
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            blurred = ndimage.gaussian_filter(data[i], sigma=1)
            out[i] = data[i] + (data[i] - blurred)
        return np.clip(out, 0, 1)
    return map_batches(channel, func, Channel)

def emboss_filter(channel: Channel) -> Channel:
    """Emboss filter for edge emphasis with direction."""
    def func(data):
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            out[i] = ndimage.convolve(data[i], kernel)
        return out
    return map_batches(channel, func, Channel)

def dog_filter(channel: Channel, sigma: Sigma) -> Channel:
    """Difference of Gaussians filter for blob detection."""
    def func(data):
        g1 = ndimage.gaussian_filter(data, sigma=(0, sigma, sigma))
        g2 = ndimage.gaussian_filter(data, sigma=(0, sigma*1.6, sigma*1.6))
        return g1 - g2
    return map_batches(channel, func, Channel)

def normalize_filter(channel: Channel) -> Channel:
    """Normalize to 0-1 range per image."""
    def func(data):
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            min_val = data[i].min()
            max_val = data[i].max()
            if max_val > min_val:
                out[i] = (data[i] - min_val) / (max_val - min_val)
            else:
                out[i] = np.zeros_like(data[i])
        return out
    return map_batches(channel, func, Channel)

def invert_filter(channel: Channel) -> Channel:
    """Invert the image (1 - x for normalized images)."""
    def func(data):
        return 1.0 - np.clip(data, 0, 1)
    return map_batches(channel, func, Channel)

def threshold_filter(channel: Channel) -> Channel:
    """Binary threshold at 0.5."""
    def func(data):
        return (data > 0.5).astype(np.float32)
    return map_batches(channel, func, Channel)

def adaptive_threshold_filter(channel: Channel) -> Channel:
    """Adaptive threshold using Gaussian weighted mean."""
    def func(data):
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            img_uint8 = cv2.normalize(data[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            out[i] = cv2.adaptiveThreshold(img_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2).astype(np.float32) / 255.0
        return out
    return map_batches(channel, func, Channel)

# --- Extended Feature Extraction Operations ---

def orb_features(channel: Channel) -> FeatureVector:
    """ORB (Oriented FAST and Rotated BRIEF) features."""
    def func(data):
        feats = []
        orb = cv2.ORB_create(nfeatures=100)
        for img in data:
            img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            kp, des = orb.detectAndCompute(img_uint8, None)
            if des is None or len(des) == 0:
                feats.append(np.zeros(32))  # ORB descriptor is 32 bytes
            else:
                feats.append(np.mean(des, axis=0))
        return np.array(feats)
    return map_batches(channel, func, FeatureVector)

def brief_features(channel: Channel) -> FeatureVector:
    """BRIEF (Binary Robust Independent Elementary Features) descriptors."""
    def func(data):
        feats = []
        star = cv2.xfeatures2d.StarDetector_create() if hasattr(cv2, 'xfeatures2d') else None
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create() if hasattr(cv2, 'xfeatures2d') else None
        
        for img in data:
            if star is None or brief is None:
                # Fallback to ORB if BRIEF not available
                orb = cv2.ORB_create(nfeatures=100)
                img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                kp, des = orb.detectAndCompute(img_uint8, None)
                if des is None:
                    feats.append(np.zeros(32))
                else:
                    feats.append(np.mean(des, axis=0))
            else:
                img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                kp = star.detect(img_uint8, None)
                kp, des = brief.compute(img_uint8, kp)
                if des is None:
                    feats.append(np.zeros(32))
                else:
                    feats.append(np.mean(des, axis=0))
        return np.array(feats)
    return map_batches(channel, func, FeatureVector)

def color_histogram_features(channel: Channel) -> FeatureVector:
    """Extended histogram with more bins."""
    def func(data):
        feats = []
        for img in data:
            hist, _ = np.histogram(img.ravel(), bins=64, range=(0, 1))
            feats.append(hist.astype(np.float32) / hist.sum())
        return np.array(feats)
    return map_batches(channel, func, FeatureVector)

def glcm_features(channel: Channel) -> FeatureVector:
    """Gray Level Co-occurrence Matrix features (texture)."""
    def func(data):
        from skimage.feature import graycomatrix, graycoprops
        feats = []
        for img in data:
            # Quantize to fewer levels
            img_uint8 = (np.clip(img, 0, 1) * 63).astype(np.uint8)
            try:
                glcm = graycomatrix(img_uint8, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                                    levels=64, symmetric=True, normed=True)
                contrast = graycoprops(glcm, 'contrast').ravel()
                dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
                homogeneity = graycoprops(glcm, 'homogeneity').ravel()
                energy = graycoprops(glcm, 'energy').ravel()
                correlation = graycoprops(glcm, 'correlation').ravel()
                feats.append(np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation]))
            except:
                feats.append(np.zeros(20))
        return np.array(feats)
    return map_batches(channel, func, FeatureVector)

def hu_moments_features(channel: Channel) -> FeatureVector:
    """Hu invariant moments (shape features)."""
    def func(data):
        feats = []
        for img in data:
            img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            moments = cv2.moments(img_uint8)
            hu = cv2.HuMoments(moments).flatten()
            # Log transform for better scaling
            hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
            feats.append(hu_log)
        return np.array(feats)
    return map_batches(channel, func, FeatureVector)

def zernike_features(channel: Channel) -> FeatureVector:
    """Zernike moments for rotation invariant shape description."""
    def func(data):
        try:
            from mahotas.features import zernike_moments
            feats = []
            for img in data:
                img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                # Radius is half of minimum dimension
                radius = min(img_uint8.shape) // 2
                zm = zernike_moments(img_uint8, radius, degree=8)
                feats.append(zm)
            return np.array(feats)
        except ImportError:
            # Fallback to Hu moments if mahotas not available
            feats = []
            for img in data:
                img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                moments = cv2.moments(img_uint8)
                hu = cv2.HuMoments(moments).flatten()
                feats.append(np.concatenate([hu, np.zeros(18)]))  # Pad to consistent size
            return np.array(feats)
    return map_batches(channel, func, FeatureVector)

def edge_histogram_features(channel: Channel) -> FeatureVector:
    """Edge histogram descriptor for texture classification."""
    def func(data):
        feats = []
        for img in data:
            # Compute gradients
            gx = ndimage.sobel(img, axis=0)
            gy = ndimage.sobel(img, axis=1)
            magnitude = np.hypot(gx, gy)
            angle = np.arctan2(gy, gx)
            
            # Quantize angles into bins
            angle_bins = ((angle + np.pi) / (2 * np.pi) * 8).astype(int) % 8
            
            # Build histogram weighted by magnitude
            hist = np.zeros(8)
            for b in range(8):
                hist[b] = np.sum(magnitude[angle_bins == b])
            
            if hist.sum() > 0:
                hist = hist / hist.sum()
            feats.append(hist)
        return np.array(feats)
    return map_batches(channel, func, FeatureVector)

def statistical_features(channel: Channel) -> FeatureVector:
    """Statistical features: mean, std, skew, kurtosis, min, max, etc."""
    def func(data):
        from scipy import stats
        feats = []
        for img in data:
            flat = img.ravel()
            mean = np.mean(flat)
            std = np.std(flat)
            skew = stats.skew(flat)
            kurtosis = stats.kurtosis(flat)
            min_val = np.min(flat)
            max_val = np.max(flat)
            median = np.median(flat)
            percentile_25 = np.percentile(flat, 25)
            percentile_75 = np.percentile(flat, 75)
            entropy = stats.entropy(np.histogram(flat, bins=64)[0] + 1e-10)
            feats.append([mean, std, skew, kurtosis, min_val, max_val, median, 
                         percentile_25, percentile_75, entropy])
        return np.array(feats)
    return map_batches(channel, func, FeatureVector)

def fourier_features(channel: Channel) -> FeatureVector:
    """Fourier descriptor features."""
    def func(data):
        feats = []
        for img in data:
            # 2D FFT
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)
            
            # Take center region as features
            h, w = magnitude.shape
            ch, cw = h // 2, w // 2
            center = magnitude[ch-4:ch+4, cw-4:cw+4].flatten()
            feats.append(center)
        return np.array(feats)
    return map_batches(channel, func, FeatureVector)

def daisy_features(channel: Channel) -> FeatureVector:
    """DAISY dense feature descriptor."""
    def func(data):
        from skimage.feature import daisy
        feats = []
        for img in data:
            try:
                desc = daisy(img, step=8, radius=15, rings=2, histograms=6, orientations=8)
                feats.append(desc.flatten()[:200])  # Truncate to consistent size
            except:
                feats.append(np.zeros(200))
        return np.array(feats)
    return map_batches(channel, func, FeatureVector)


# =============================================================================
# PARAMETERIZED OPERATIONS (for extended_params mode)
# =============================================================================

# --- Parameterized HOG Operations ---

def hog_filter_param(channel: Channel, ppc: PixelsPerCell, cpb: CellsPerBlock, orient: Orientations) -> Channel:
    """HOG filter with configurable parameters."""
    def func(data):
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            try:
                _, hog_img = hog(data[i], orientations=orient, 
                                 pixels_per_cell=(ppc, ppc), 
                                 cells_per_block=(cpb, cpb), 
                                 visualize=True)
                # Resize back to original size
                hog_img_resized = cv2.resize(hog_img, (data.shape[2], data.shape[1]))
                out[i] = hog_img_resized
            except:
                pass
        return out
    return map_batches(channel, func, Channel)

def hog_features_param(channel: Channel, ppc: PixelsPerCell, cpb: CellsPerBlock, orient: Orientations) -> FeatureVector:
    """HOG features with configurable parameters."""
    def func(data):
        feats = []
        for img in data:
            try:
                fd = hog(img, orientations=orient, 
                        pixels_per_cell=(ppc, ppc), 
                        cells_per_block=(cpb, cpb))
                feats.append(fd)
            except:
                feats.append(np.zeros(10))
        return np.array(feats)
    return map_batches(channel, func, FeatureVector)

def hog_image_features_param(channel: Channel, ppc: PixelsPerCell, cpb: CellsPerBlock, orient: Orientations) -> FeatureVector:
    """HOG image features with configurable parameters."""
    def func(data):
        out = []
        for i in range(data.shape[0]):
            try:
                _, hog_img = hog(data[i], orientations=orient, 
                                 pixels_per_cell=(ppc, ppc), 
                                 cells_per_block=(cpb, cpb), 
                                 visualize=True)
                out.append(hog_img.flatten())
            except:
                out.append(np.zeros(data.shape[1]*data.shape[2]))
        return np.array(out)
    return map_batches(channel, func, FeatureVector)

# --- Parameterized LBP Operations ---

def lbp_filter_param(channel: Channel, p: LBP_P, r: LBP_R) -> Channel:
    """LBP filter with configurable P and R parameters."""
    def func(data):
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            out[i] = local_binary_pattern(data[i], P=p, R=r, method='uniform')
        return out
    return map_batches(channel, func, Channel)

def lbp_features_param(channel: Channel, p: LBP_P, r: LBP_R) -> FeatureVector:
    """LBP features with configurable P and R parameters."""
    def func(data):
        feats = []
        n_bins = p + 2  # uniform LBP has P+2 patterns
        for img in data:
            lbp = local_binary_pattern(img, P=p, R=r, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
            feats.append(hist)
        return np.array(feats)
    return map_batches(channel, func, FeatureVector)

def lbp_image_features_param(channel: Channel, p: LBP_P, r: LBP_R) -> FeatureVector:
    """LBP image features with configurable P and R parameters."""
    def func(data):
        out = []
        for i in range(data.shape[0]):
            lbp = local_binary_pattern(data[i], P=p, R=r, method='uniform')
            out.append(lbp.flatten())
        return np.array(out)
    return map_batches(channel, func, FeatureVector)

# --- Parameterized Filter Operations ---

def mean_filter_param(channel: Channel, ksize: KernelSize) -> Channel:
    """Mean filter with configurable kernel size."""
    def func(data):
        return ndimage.uniform_filter(data, size=(1, ksize, ksize))
    return map_batches(channel, func, Channel)

def median_filter_param(channel: Channel, ksize: KernelSize) -> Channel:
    """Median filter with configurable kernel size."""
    def func(data):
        return ndimage.median_filter(data, size=(1, ksize, ksize))
    return map_batches(channel, func, Channel)

def min_filter_param(channel: Channel, ksize: KernelSize) -> Channel:
    """Min filter with configurable kernel size."""
    def func(data):
        return ndimage.minimum_filter(data, size=(1, ksize, ksize))
    return map_batches(channel, func, Channel)

def max_filter_param(channel: Channel, ksize: KernelSize) -> Channel:
    """Max filter with configurable kernel size."""
    def func(data):
        return ndimage.maximum_filter(data, size=(1, ksize, ksize))
    return map_batches(channel, func, Channel)

# --- Parameterized Morphological Operations ---

def erosion_filter_param(channel: Channel, ksize: MorphKernel) -> Channel:
    """Morphological erosion with configurable kernel size."""
    def func(data):
        kernel = np.ones((ksize, ksize), np.uint8)
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            img_uint8 = cv2.normalize(data[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            out[i] = cv2.erode(img_uint8, kernel).astype(np.float32) / 255.0
        return out
    return map_batches(channel, func, Channel)

def dilation_filter_param(channel: Channel, ksize: MorphKernel) -> Channel:
    """Morphological dilation with configurable kernel size."""
    def func(data):
        kernel = np.ones((ksize, ksize), np.uint8)
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            img_uint8 = cv2.normalize(data[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            out[i] = cv2.dilate(img_uint8, kernel).astype(np.float32) / 255.0
        return out
    return map_batches(channel, func, Channel)

def opening_filter_param(channel: Channel, ksize: MorphKernel) -> Channel:
    """Morphological opening with configurable kernel size."""
    def func(data):
        kernel = np.ones((ksize, ksize), np.uint8)
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            img_uint8 = cv2.normalize(data[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            out[i] = cv2.morphologyEx(img_uint8, cv2.MORPH_OPEN, kernel).astype(np.float32) / 255.0
        return out
    return map_batches(channel, func, Channel)

def closing_filter_param(channel: Channel, ksize: MorphKernel) -> Channel:
    """Morphological closing with configurable kernel size."""
    def func(data):
        kernel = np.ones((ksize, ksize), np.uint8)
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            img_uint8 = cv2.normalize(data[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            out[i] = cv2.morphologyEx(img_uint8, cv2.MORPH_CLOSE, kernel).astype(np.float32) / 255.0
        return out
    return map_batches(channel, func, Channel)

# --- Parameterized Canny ---

def canny_filter_param(channel: Channel, low: CannyLow, high: CannyHigh) -> Channel:
    """Canny edge detection with configurable thresholds."""
    def func(data):
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            img_uint8 = cv2.normalize(data[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            edges = cv2.Canny(img_uint8, int(low), int(high))
            out[i] = edges.astype(np.float32) / 255.0
        return out
    return map_batches(channel, func, Channel)

# --- Parameterized SIFT ---

def sift_features_param(channel: Channel, nkp: NumKeypoints) -> FeatureVector:
    """SIFT features with configurable number of keypoints."""
    def func(data):
        feats = []
        sift = cv2.SIFT_create(nfeatures=nkp)
        for img in data:
            img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            kp, des = sift.detectAndCompute(img_uint8, None)
            if des is None or len(des) == 0:
                feats.append(np.zeros(128))
            else:
                feats.append(np.mean(des, axis=0))
        return np.array(feats)
    return map_batches(channel, func, FeatureVector)

