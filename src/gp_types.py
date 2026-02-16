import numpy as np
from typing import List, Union, Iterator, Callable

class Batch:
    def __init__(self, data: np.ndarray):
        self.data = data

class DataWrapper:
    """
    Base class for all data types flowing through the GP tree.
    Supports lazy iteration over batches.
    """
    def __init__(self, iterator_factory: Callable[[], Iterator[Batch]]):
        self._iterator_factory = iterator_factory

    def __iter__(self) -> Iterator[Batch]:
        return self._iterator_factory()

class Image(DataWrapper):
    """
    Represents a stream of image batches.
    Each batch.data is List[np.ndarray] (channels) or np.ndarray (N, C, H, W).
    """
    pass

class Channel(DataWrapper):
    """
    Represents a stream of single channel batches.
    Each batch.data is (N, H, W).
    """
    pass

class FeatureVector(DataWrapper):
    """
    Represents a stream of feature vector batches.
    Each batch.data is (N, Features).
    """
    pass

class Prediction(DataWrapper):
    """
    Represents a stream of prediction batches from a single classifier.
    Each batch.data is (N, Classes).
    
    NOTE: This is the output of individual classification heads (rf_classification, etc.)
    and cascade classifiers. For the final ensemble output, see EnsembleOutput.
    """
    pass

class EnsembleOutput(DataWrapper):
    """
    Represents the final ensemble output (summation of predictions + majority voting).
    Each batch.data is (N, Classes).
    
    This is the ROOT TYPE of all legal GP trees. It can only be produced by:
    - ensemble_single: Wraps a single classifier's Prediction
    - ensemble_sum_2: Sums two Predictions
    - ensemble_sum_3: Sums three Predictions
    
    This type enforces that summation/ensemble operations can ONLY occur at the 
    top of the tree (since nothing consumes EnsembleOutput except the final output).
    This prevents illegal structures like nested summations or classification after summation.
    """
    pass

# Terminal Types
class Weight:
    pass

class Trees:
    pass

class Depth:
    pass

class Frequency:
    pass

class Theta:
    pass

class Sigma:
    pass

class Order:
    pass

# --- Extended Parameter Types (for extended_params mode) ---

class PixelsPerCell:
    """HOG pixels per cell dimension (e.g., 4, 8, 16)."""
    pass

class CellsPerBlock:
    """HOG cells per block dimension (e.g., 1, 2, 3)."""
    pass

class Orientations:
    """HOG number of orientation bins (e.g., 6, 9, 12)."""
    pass

class LBP_P:
    """LBP number of circularly symmetric neighbour points."""
    pass

class LBP_R:
    """LBP radius of circle."""
    pass

class KernelSize:
    """Kernel size for filter operations (e.g., 3, 5, 7)."""
    pass

class NumKeypoints:
    """Number of keypoints for feature detectors."""
    pass

class Threshold:
    """Threshold value for various operations."""
    pass

class MorphKernel:
    """Morphological operation kernel size."""
    pass

class CannyLow:
    """Canny edge detector low threshold."""
    pass

class CannyHigh:
    """Canny edge detector high threshold."""
    pass
