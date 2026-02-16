import random
import math
from deap import gp
from .gp_types import (
    Image, Channel, FeatureVector, Prediction, EnsembleOutput, Weight, Sigma, Trees, Depth, 
    Frequency, Theta, Order,
    # Extended parameter types
    PixelsPerCell, CellsPerBlock, Orientations, LBP_P, LBP_R, KernelSize,
    NumKeypoints, Threshold, MorphKernel, CannyLow, CannyHigh
)
from . import gp_ops


def create_primitive_set(extended_ops: bool = False, extended_params: bool = False):
    """
    Creates the primitive set defining the GP language for image classification trees.
    
    Args:
        extended_ops: If True, includes extended filtering and feature extraction operations
                      (morphological ops, edge detectors, texture features, etc.)
        extended_params: If True, includes parameterized versions of operations with
                         configurable parameters (pixels_per_cell, kernel sizes, etc.)
    
    Legal Tree Structure (data flow) - enforced by type system:
    ============================================================
    1. Input Layer: Channel terminals (GetRed, GetGreen, GetBlue, GetGray)
    2. Image Filtering Layer: Channel -> Channel (optional, can be stacked)
    3. Feature Extraction Layer: Channel -> FeatureVector (required)
    4. Feature Concatenation Layer: FeatureVector x N -> FeatureVector (optional)
    5. Classification & Cascade Layer: FeatureVector -> FeatureVector (optional, can be stacked)
    6. Feature Concatenation Layer: FeatureVector x N -> FeatureVector (optional)
    7. Final Classification Layer: FeatureVector -> Prediction (required)
    8. Ensemble/Summation Layer: Prediction x N -> EnsembleOutput (required, at root)
    
    The key constraint is that EnsembleOutput is the ROOT TYPE and nothing consumes it.
    This prevents:
    - Nested summations (sum(sum(...), ...))
    - Classification after summation
    - Any operations after the final ensemble layer
    
    Parameter types (Trees, Depth, Sigma, etc.) are only used as arguments to primitives,
    never as intermediate data flow types.
    """
    # Root type is EnsembleOutput - this enforces that ensemble/summation is at the top
    pset = gp.PrimitiveSetTyped("MAIN", [], EnsembleOutput)

    # === TERMINALS ===
    
    # --- Basic Parameters (always included) ---
    pset.addEphemeralConstant("t", lambda: random.randrange(50, 1001, 50), Trees)
    pset.addEphemeralConstant("d", lambda: random.randrange(10, 101, 10), Depth)
    pset.addEphemeralConstant("f", lambda: random.choice([math.pi/8 * i for i in range(1, 5)]), Frequency)
    pset.addEphemeralConstant("theta", lambda: random.choice([math.pi/8 * i for i in range(0, 8)]), Theta)
    pset.addEphemeralConstant("sigma", lambda: random.randint(1, 3), Sigma)
    pset.addEphemeralConstant("weight", lambda: random.random(), Weight)
    pset.addEphemeralConstant("order", lambda: random.randint(0, 2), Order)
    
    # --- Extended Parameters (only when extended_params=True) ---
    if extended_params:
        # HOG parameters
        pset.addEphemeralConstant("ppc", lambda: random.choice([4, 8, 16]), PixelsPerCell)
        pset.addEphemeralConstant("cpb", lambda: random.choice([1, 2, 3]), CellsPerBlock)
        pset.addEphemeralConstant("orient", lambda: random.choice([6, 9, 12]), Orientations)
        
        # LBP parameters
        pset.addEphemeralConstant("lbp_p", lambda: random.choice([8, 16, 24]), LBP_P)
        pset.addEphemeralConstant("lbp_r", lambda: random.choice([1, 1.5, 2, 3]), LBP_R)
        
        # Kernel sizes
        pset.addEphemeralConstant("ksize", lambda: random.choice([3, 5, 7]), KernelSize)
        pset.addEphemeralConstant("mkernel", lambda: random.choice([3, 5, 7]), MorphKernel)
        
        # Canny thresholds
        pset.addEphemeralConstant("canny_low", lambda: random.choice([30, 50, 70, 100]), CannyLow)
        pset.addEphemeralConstant("canny_high", lambda: random.choice([100, 150, 200, 250]), CannyHigh)
        
        # Feature detector parameters
        pset.addEphemeralConstant("nkp", lambda: random.choice([50, 100, 200, 500]), NumKeypoints)

    # --- Input Nodes (Terminals) ---
    pset.addTerminal(gp_ops.RedTerminal, Channel, name="GetRed")
    pset.addTerminal(gp_ops.GreenTerminal, Channel, name="GetGreen")
    pset.addTerminal(gp_ops.BlueTerminal, Channel, name="GetBlue")
    pset.addTerminal(gp_ops.GrayTerminal, Channel, name="GetGray")

    # === PRIMITIVES ===
    
    # --- Channel Combination ---
    pset.addPrimitive(gp_ops.linear_combination, [Channel, Channel, Channel, Weight, Weight, Weight], Channel)

    # --- Basic Image Filtering (Channel -> Channel) ---
    pset.addPrimitive(gp_ops.mean_filter, [Channel], Channel)
    pset.addPrimitive(gp_ops.median_filter, [Channel], Channel)
    pset.addPrimitive(gp_ops.min_filter, [Channel], Channel)
    pset.addPrimitive(gp_ops.max_filter, [Channel], Channel)
    pset.addPrimitive(gp_ops.gaussian_filter, [Channel, Sigma], Channel)
    pset.addPrimitive(gp_ops.laplacian_filter, [Channel], Channel)
    pset.addPrimitive(gp_ops.sobel_filter, [Channel], Channel)
    pset.addPrimitive(gp_ops.relu, [Channel], Channel)
    pset.addPrimitive(gp_ops.sqrt_op, [Channel], Channel)
    pset.addPrimitive(gp_ops.gaud_filter, [Channel, Sigma, Order, Order], Channel)
    pset.addPrimitive(gp_ops.log1_filter, [Channel], Channel)
    pset.addPrimitive(gp_ops.log2_filter, [Channel], Channel)
    pset.addPrimitive(gp_ops.gabor_filter, [Channel, Theta, Frequency], Channel)
    pset.addPrimitive(gp_ops.hog_filter, [Channel], Channel)
    pset.addPrimitive(gp_ops.lbp_filter, [Channel], Channel)
    pset.addPrimitive(gp_ops.add_max_pool, [Channel, Channel], Channel)
    pset.addPrimitive(gp_ops.sub_max_pool, [Channel, Channel], Channel)

    # --- Extended Image Filtering (only when extended_ops=True) ---
    if extended_ops:
        pset.addPrimitive(gp_ops.prewitt_filter, [Channel], Channel)
        pset.addPrimitive(gp_ops.scharr_filter, [Channel], Channel)
        pset.addPrimitive(gp_ops.canny_filter, [Channel], Channel)
        pset.addPrimitive(gp_ops.bilateral_filter, [Channel, Sigma], Channel)
        pset.addPrimitive(gp_ops.clahe_filter, [Channel], Channel)
        pset.addPrimitive(gp_ops.erosion_filter, [Channel], Channel)
        pset.addPrimitive(gp_ops.dilation_filter, [Channel], Channel)
        pset.addPrimitive(gp_ops.opening_filter, [Channel], Channel)
        pset.addPrimitive(gp_ops.closing_filter, [Channel], Channel)
        pset.addPrimitive(gp_ops.gradient_filter, [Channel], Channel)
        pset.addPrimitive(gp_ops.tophat_filter, [Channel], Channel)
        pset.addPrimitive(gp_ops.blackhat_filter, [Channel], Channel)
        pset.addPrimitive(gp_ops.sharpen_filter, [Channel], Channel)
        pset.addPrimitive(gp_ops.emboss_filter, [Channel], Channel)
        pset.addPrimitive(gp_ops.dog_filter, [Channel, Sigma], Channel)
        pset.addPrimitive(gp_ops.normalize_filter, [Channel], Channel)
        pset.addPrimitive(gp_ops.invert_filter, [Channel], Channel)
        pset.addPrimitive(gp_ops.threshold_filter, [Channel], Channel)
        pset.addPrimitive(gp_ops.adaptive_threshold_filter, [Channel], Channel)

    # --- Parameterized Image Filtering (only when extended_params=True) ---
    if extended_params:
        pset.addPrimitive(gp_ops.hog_filter_param, [Channel, PixelsPerCell, CellsPerBlock, Orientations], Channel)
        pset.addPrimitive(gp_ops.lbp_filter_param, [Channel, LBP_P, LBP_R], Channel)
        pset.addPrimitive(gp_ops.mean_filter_param, [Channel, KernelSize], Channel)
        pset.addPrimitive(gp_ops.median_filter_param, [Channel, KernelSize], Channel)
        pset.addPrimitive(gp_ops.min_filter_param, [Channel, KernelSize], Channel)
        pset.addPrimitive(gp_ops.max_filter_param, [Channel, KernelSize], Channel)
        
    # --- Parameterized Morphological Filters (requires both extended_ops and extended_params) ---
    if extended_ops and extended_params:
        pset.addPrimitive(gp_ops.erosion_filter_param, [Channel, MorphKernel], Channel)
        pset.addPrimitive(gp_ops.dilation_filter_param, [Channel, MorphKernel], Channel)
        pset.addPrimitive(gp_ops.opening_filter_param, [Channel, MorphKernel], Channel)
        pset.addPrimitive(gp_ops.closing_filter_param, [Channel, MorphKernel], Channel)
        pset.addPrimitive(gp_ops.canny_filter_param, [Channel, CannyLow, CannyHigh], Channel)

    # --- Basic Feature Extraction (Channel -> FeatureVector) ---
    pset.addPrimitive(gp_ops.histogram_features, [Channel], FeatureVector)
    pset.addPrimitive(gp_ops.hog_features, [Channel], FeatureVector)
    pset.addPrimitive(gp_ops.lbp_features, [Channel], FeatureVector)
    pset.addPrimitive(gp_ops.concat_images, [Channel, Channel], FeatureVector)
    pset.addPrimitive(gp_ops.sift_features, [Channel], FeatureVector)
    pset.addPrimitive(gp_ops.hog_image_features, [Channel], FeatureVector)
    pset.addPrimitive(gp_ops.lbp_image_features, [Channel], FeatureVector)
    pset.addPrimitive(gp_ops.sobel_features, [Channel], FeatureVector)
    pset.addPrimitive(gp_ops.gabor_features, [Channel, Theta, Frequency], FeatureVector)
    pset.addPrimitive(gp_ops.gaussian_features, [Channel, Sigma], FeatureVector)
    pset.addPrimitive(gp_ops.gaud_features, [Channel, Sigma, Order, Order], FeatureVector)

    # --- Extended Feature Extraction (only when extended_ops=True) ---
    if extended_ops:
        pset.addPrimitive(gp_ops.orb_features, [Channel], FeatureVector)
        pset.addPrimitive(gp_ops.brief_features, [Channel], FeatureVector)
        pset.addPrimitive(gp_ops.color_histogram_features, [Channel], FeatureVector)
        pset.addPrimitive(gp_ops.glcm_features, [Channel], FeatureVector)
        pset.addPrimitive(gp_ops.hu_moments_features, [Channel], FeatureVector)
        pset.addPrimitive(gp_ops.zernike_features, [Channel], FeatureVector)
        pset.addPrimitive(gp_ops.edge_histogram_features, [Channel], FeatureVector)
        pset.addPrimitive(gp_ops.statistical_features, [Channel], FeatureVector)
        pset.addPrimitive(gp_ops.fourier_features, [Channel], FeatureVector)
        pset.addPrimitive(gp_ops.daisy_features, [Channel], FeatureVector)

    # --- Parameterized Feature Extraction (only when extended_params=True) ---
    if extended_params:
        pset.addPrimitive(gp_ops.hog_features_param, [Channel, PixelsPerCell, CellsPerBlock, Orientations], FeatureVector)
        pset.addPrimitive(gp_ops.hog_image_features_param, [Channel, PixelsPerCell, CellsPerBlock, Orientations], FeatureVector)
        pset.addPrimitive(gp_ops.lbp_features_param, [Channel, LBP_P, LBP_R], FeatureVector)
        pset.addPrimitive(gp_ops.lbp_image_features_param, [Channel, LBP_P, LBP_R], FeatureVector)
        pset.addPrimitive(gp_ops.sift_features_param, [Channel, NumKeypoints], FeatureVector)

    # --- Feature Concatenation (FeatureVector x N -> FeatureVector) ---
    pset.addPrimitive(gp_ops.concat_features_2, [FeatureVector, FeatureVector], FeatureVector)
    pset.addPrimitive(gp_ops.concat_features_3, [FeatureVector, FeatureVector, FeatureVector], FeatureVector)
    pset.addPrimitive(gp_ops.concat_features_4, [FeatureVector, FeatureVector, FeatureVector, FeatureVector], FeatureVector)

    # --- Cascade Classification (FeatureVector -> FeatureVector) ---
    pset.addPrimitive(gp_ops.cascade_rf, [FeatureVector, Trees, Depth], FeatureVector)
    pset.addPrimitive(gp_ops.cascade_erf, [FeatureVector, Trees, Depth], FeatureVector)
    pset.addPrimitive(gp_ops.cascade_lr, [FeatureVector], FeatureVector)
    pset.addPrimitive(gp_ops.cascade_svm, [FeatureVector], FeatureVector)

    # --- Final Classification (FeatureVector -> Prediction) ---
    # These produce Prediction, which must be consumed by ensemble operations
    pset.addPrimitive(gp_ops.rf_classification, [FeatureVector, Trees, Depth], Prediction)
    pset.addPrimitive(gp_ops.erf_classification, [FeatureVector, Trees, Depth], Prediction)
    pset.addPrimitive(gp_ops.lr_classification, [FeatureVector], Prediction)
    pset.addPrimitive(gp_ops.svm_classification, [FeatureVector], Prediction)

    # --- Ensemble Output Layer (Prediction -> EnsembleOutput) ---
    # These are the ONLY operations that produce EnsembleOutput (the root type).
    # This enforces that the tree structure ends with: Classification -> Ensemble -> Output
    # Since nothing consumes EnsembleOutput, summation can ONLY occur at the root.
    pset.addPrimitive(gp_ops.ensemble_single, [Prediction], EnsembleOutput)
    pset.addPrimitive(gp_ops.ensemble_sum_2, [Prediction, Prediction], EnsembleOutput)
    pset.addPrimitive(gp_ops.ensemble_sum_3, [Prediction, Prediction, Prediction], EnsembleOutput)

    return pset


def get_mode_description(extended_ops: bool, extended_params: bool) -> str:
    """Returns a human-readable description of the current mode."""
    ops_mode = "Extended" if extended_ops else "Basic"
    params_mode = "Extended" if extended_params else "Basic"
    return f"Operations: {ops_mode}, Parameters: {params_mode}"


def count_primitives(pset) -> dict:
    """Returns a count of primitives and terminals in the primitive set."""
    n_primitives = sum(len(prims) for prims in pset.primitives.values())
    n_terminals = sum(len(terms) for terms in pset.terminals.values())
    return {
        "primitives": n_primitives,
        "terminals": n_terminals,
        "total": n_primitives + n_terminals
    }
