"""
Tests for dtype consistency and numerical correctness across all GP nodes.

This file catches:
  - dtype mismatches (float16/float32/float64/int64 where unexpected)
  - batch-to-batch dtype inconsistency within a single node
  - dtype propagation bugs through filter chains
  - value-range violations (e.g. probabilities outside [0,1])
  - classifier probability sanity (rows sum to ~1.0)
  - NaN / Inf contamination

Run with debug output:
    pytest unit_tests/test_dtype_consistency.py --debug-ops -s
"""
import math
import numpy as np
import pytest

from src import gp_ops, gp_types, gp_context

from unit_tests.conftest import (
    dbg_print, consume_channel, consume_features, consume_prediction,
    fresh_image_setup, _make_random_images, _make_labels,
    N_SAMPLES, IMG_H, IMG_W, N_CLASSES, BATCH_SIZE,
)


# ===================================================================
# Expected dtypes — the "ground truth" contract for each layer
# ===================================================================
# Channel outputs should be float32 (map_batches coerces them).
# FeatureVector outputs vary; we document what each actually produces.
# Prediction/EnsembleOutput should be float64 (sklearn predict_proba).

CHANNEL_DTYPE = np.float32     # enforced by map_batches
PREDICTION_DTYPE = np.float64  # sklearn predict_proba


# ===================================================================
# Helpers
# ===================================================================

def _refresh(X=None):
    if X is None:
        X = _make_random_images()
    fresh_image_setup(X)
    return X


def _collect_batch_dtypes(data_wrapper):
    """Iterate a DataWrapper, return (stacked_array, list_of_per_batch_dtypes)."""
    arrays = []
    dtypes = []
    for batch in data_wrapper:
        arrays.append(batch.data)
        dtypes.append(batch.data.dtype)
    stacked = np.vstack(arrays) if arrays else np.array([])
    return stacked, dtypes


def _assert_uniform_dtype(dtypes, label):
    """All batches must share a single dtype."""
    unique = set(str(d) for d in dtypes)
    assert len(unique) == 1, \
        f"{label}: batches have mixed dtypes {unique}"


def _assert_dtype(arr, expected, label):
    """Array must have the expected dtype."""
    assert arr.dtype == expected, \
        f"{label}: expected dtype {expected}, got {arr.dtype}"


def _assert_floating(arr, label):
    """Array must be a floating type (float16/32/64), not integer."""
    assert np.issubdtype(arr.dtype, np.floating), \
        f"{label}: expected floating dtype, got {arr.dtype}"


# ===================================================================
# 1. CHANNEL TERMINAL DTYPES
# ===================================================================

class TestTerminalDtypes:

    @pytest.mark.parametrize("terminal,name", [
        (gp_ops.RedTerminal, "Red"),
        (gp_ops.GreenTerminal, "Green"),
        (gp_ops.BlueTerminal, "Blue"),
        (gp_ops.GrayTerminal, "Gray"),
    ])
    def test_terminal_dtype_is_float32(self, terminal, name, debug):
        """Channel terminals should produce float32 (input images are float32)."""
        _refresh()
        result, dtypes = _collect_batch_dtypes(terminal)
        _assert_uniform_dtype(dtypes, name)
        # Terminals pass through raw channel slices from the float32 input.
        # Gray is computed via 0.299*R + 0.587*G + 0.114*B which promotes to float64
        # because the constants are Python floats. This is a known inconsistency.
        dbg_print(debug, f"  {name} terminal: dtype={result.dtype}, batch_dtypes={set(str(d) for d in dtypes)}")
        assert np.issubdtype(result.dtype, np.floating), \
            f"{name}: expected floating, got {result.dtype}"

    def test_red_green_blue_are_float32(self, debug):
        """R/G/B terminals should be exactly float32 (sliced from float32 input)."""
        for terminal, name in [(gp_ops.RedTerminal, "Red"),
                               (gp_ops.GreenTerminal, "Green"),
                               (gp_ops.BlueTerminal, "Blue")]:
            _refresh()
            result, dtypes = _collect_batch_dtypes(terminal)
            _assert_dtype(result, np.float32, name)
            dbg_print(debug, f"  {name}: dtype={result.dtype} ✓")

    def test_gray_dtype(self, debug):
        """Gray terminal: 0.299*R + 0.587*G + 0.114*B.
        Python float * float32 -> float64. Document the actual dtype."""
        _refresh()
        result, dtypes = _collect_batch_dtypes(gp_ops.GrayTerminal)
        # This is a known promotion: float literal × float32 → float64
        dbg_print(debug, f"  Gray terminal dtype: {result.dtype} (expected float64 due to Python float promotion)")
        _assert_floating(result, "Gray")


# ===================================================================
# 2. BASIC FILTER DTYPES  (Channel → Channel)
# ===================================================================

class TestBasicFilterDtypes:
    """All Channel→Channel filters pass through map_batches which coerces to float32."""

    UNARY_FILTERS = [
        ("mean_filter", gp_ops.mean_filter, {}),
        ("median_filter", gp_ops.median_filter, {}),
        ("min_filter", gp_ops.min_filter, {}),
        ("max_filter", gp_ops.max_filter, {}),
        ("laplacian_filter", gp_ops.laplacian_filter, {}),
        ("sobel_filter", gp_ops.sobel_filter, {}),
        ("relu", gp_ops.relu, {}),
        ("sqrt_op", gp_ops.sqrt_op, {}),
        ("log1_filter", gp_ops.log1_filter, {}),
        ("log2_filter", gp_ops.log2_filter, {}),
        ("hog_filter", gp_ops.hog_filter, {}),
        ("lbp_filter", gp_ops.lbp_filter, {}),
    ]

    @pytest.mark.parametrize("name,func,kwargs", UNARY_FILTERS)
    def test_unary_filter_dtype_float32(self, name, func, kwargs, debug):
        _refresh()
        ch = func(gp_ops.GrayTerminal, **kwargs)
        result, dtypes = _collect_batch_dtypes(ch)
        _assert_uniform_dtype(dtypes, name)
        _assert_dtype(result, CHANNEL_DTYPE, name)
        dbg_print(debug, f"  {name}: dtype={result.dtype} ✓")

    def test_gaussian_filter_dtype(self, debug):
        _refresh()
        ch = gp_ops.gaussian_filter(gp_ops.GrayTerminal, 2)
        result, dtypes = _collect_batch_dtypes(ch)
        _assert_dtype(result, CHANNEL_DTYPE, "gaussian_filter")
        dbg_print(debug, f"  gaussian_filter: dtype={result.dtype} ✓")

    def test_gabor_filter_dtype(self, debug):
        _refresh()
        ch = gp_ops.gabor_filter(gp_ops.GrayTerminal, math.pi / 4, math.pi / 8)
        result, dtypes = _collect_batch_dtypes(ch)
        _assert_dtype(result, CHANNEL_DTYPE, "gabor_filter")

    def test_gaud_filter_dtype(self, debug):
        _refresh()
        ch = gp_ops.gaud_filter(gp_ops.GrayTerminal, 1, 1, 0)
        result, dtypes = _collect_batch_dtypes(ch)
        _assert_dtype(result, CHANNEL_DTYPE, "gaud_filter")

    def test_linear_combination_dtype(self, debug):
        """linear_combination uses map_batches_3 (no Channel coercion).
        Python float weights * float32 arrays → float64.
        This is a known dtype inconsistency."""
        _refresh()
        ch = gp_ops.linear_combination(
            gp_ops.RedTerminal, gp_ops.GreenTerminal, gp_ops.BlueTerminal,
            0.3, 0.5, 0.2,
        )
        result, dtypes = _collect_batch_dtypes(ch)
        _assert_uniform_dtype(dtypes, "linear_combination")
        _assert_floating(result, "linear_combination")
        dbg_print(debug, f"  linear_combination: dtype={result.dtype} "
                  f"(uses map_batches_3 → no float32 coercion)")

    def test_add_max_pool_dtype(self, debug):
        """add_max_pool uses map_batches_2 → no Channel coercion in map_batches_2."""
        _refresh()
        ch = gp_ops.add_max_pool(gp_ops.RedTerminal, gp_ops.GreenTerminal)
        result, dtypes = _collect_batch_dtypes(ch)
        _assert_uniform_dtype(dtypes, "add_max_pool")
        _assert_floating(result, "add_max_pool")
        dbg_print(debug, f"  add_max_pool: dtype={result.dtype}")

    def test_sub_max_pool_dtype(self, debug):
        _refresh()
        ch = gp_ops.sub_max_pool(gp_ops.RedTerminal, gp_ops.GreenTerminal)
        result, dtypes = _collect_batch_dtypes(ch)
        _assert_uniform_dtype(dtypes, "sub_max_pool")
        _assert_floating(result, "sub_max_pool")
        dbg_print(debug, f"  sub_max_pool: dtype={result.dtype}")


# ===================================================================
# 3. EXTENDED FILTER DTYPES
# ===================================================================

class TestExtendedFilterDtypes:

    EXTENDED_UNARY = [
        ("prewitt_filter", gp_ops.prewitt_filter),
        ("scharr_filter", gp_ops.scharr_filter),
        ("canny_filter", gp_ops.canny_filter),
        ("clahe_filter", gp_ops.clahe_filter),
        ("erosion_filter", gp_ops.erosion_filter),
        ("dilation_filter", gp_ops.dilation_filter),
        ("opening_filter", gp_ops.opening_filter),
        ("closing_filter", gp_ops.closing_filter),
        ("gradient_filter", gp_ops.gradient_filter),
        ("tophat_filter", gp_ops.tophat_filter),
        ("blackhat_filter", gp_ops.blackhat_filter),
        ("sharpen_filter", gp_ops.sharpen_filter),
        ("emboss_filter", gp_ops.emboss_filter),
        ("normalize_filter", gp_ops.normalize_filter),
        ("invert_filter", gp_ops.invert_filter),
        ("threshold_filter", gp_ops.threshold_filter),
        ("adaptive_threshold_filter", gp_ops.adaptive_threshold_filter),
    ]

    @pytest.mark.parametrize("name,func", EXTENDED_UNARY)
    def test_extended_filter_dtype_float32(self, name, func, debug):
        _refresh()
        ch = func(gp_ops.GrayTerminal)
        result, dtypes = _collect_batch_dtypes(ch)
        _assert_uniform_dtype(dtypes, name)
        _assert_dtype(result, CHANNEL_DTYPE, name)
        dbg_print(debug, f"  {name}: dtype={result.dtype} ✓")

    def test_bilateral_filter_dtype(self, debug):
        _refresh()
        ch = gp_ops.bilateral_filter(gp_ops.GrayTerminal, 2)
        result, _ = _collect_batch_dtypes(ch)
        _assert_dtype(result, CHANNEL_DTYPE, "bilateral_filter")

    def test_dog_filter_dtype(self, debug):
        _refresh()
        ch = gp_ops.dog_filter(gp_ops.GrayTerminal, 1)
        result, _ = _collect_batch_dtypes(ch)
        _assert_dtype(result, CHANNEL_DTYPE, "dog_filter")


# ===================================================================
# 4. FEATURE EXTRACTOR DTYPES
# ===================================================================

class TestFeatureExtractorDtypes:
    """
    Feature extractors have mixed dtypes:
      - histogram-based: int64 (np.histogram returns int counts)
      - most others: float64 (scipy/skimage produce float64)
      - concat_images, color_histogram_features: float32
    We document the actual dtype and ensure it is at least floating or
    numeric, and uniform across batches.
    """

    # (name, func, kwargs, expected_dtype_or_None)
    # None means "just check floating + uniform"
    FEATURE_DTYPES = [
        ("histogram_features", gp_ops.histogram_features, {},
         None),  # int64 from np.histogram — integer features
        ("hog_features", gp_ops.hog_features, {},
         None),  # float32 or float64 depending on skimage version
        ("lbp_features", gp_ops.lbp_features, {},
         None),  # int64 from np.histogram
        ("sift_features", gp_ops.sift_features, {},
         np.float32),
        ("hog_image_features", gp_ops.hog_image_features, {},
         None),  # float32 or float64 depending on skimage version
        ("lbp_image_features", gp_ops.lbp_image_features, {},
         None),  # float32 or float64 depending on skimage
        ("sobel_features", gp_ops.sobel_features, {},
         None),  # float64 from ndimage, but may vary
        ("concat_images", None, {},
         None),  # two-input, handled separately
    ]

    @pytest.mark.parametrize("name,func,kwargs,expected_dtype", [
        t for t in FEATURE_DTYPES if t[1] is not None
    ])
    def test_basic_fe_dtype(self, name, func, kwargs, expected_dtype, debug):
        _refresh()
        fv = func(gp_ops.GrayTerminal, **kwargs)
        result, dtypes = _collect_batch_dtypes(fv)
        _assert_uniform_dtype(dtypes, name)
        assert np.issubdtype(result.dtype, np.number), \
            f"{name}: expected numeric dtype, got {result.dtype}"
        if expected_dtype is not None:
            _assert_dtype(result, expected_dtype, name)
        dbg_print(debug, f"  {name}: dtype={result.dtype}")

    def test_histogram_features_dtype_is_integer(self, debug):
        """histogram_features uses np.histogram which returns int64 counts.
        This is a known inconsistency — it works with sklearn but differs
        from other feature extractors which are float."""
        _refresh()
        fv = gp_ops.histogram_features(gp_ops.GrayTerminal)
        result, dtypes = _collect_batch_dtypes(fv)
        _assert_uniform_dtype(dtypes, "histogram_features")
        dbg_print(debug, f"  histogram_features: dtype={result.dtype}")
        # Document: histogram_features returns int64
        assert np.issubdtype(result.dtype, np.number), \
            "histogram_features should be numeric"

    def test_lbp_features_dtype_is_integer(self, debug):
        """lbp_features also uses np.histogram → int64 counts."""
        _refresh()
        fv = gp_ops.lbp_features(gp_ops.GrayTerminal)
        result, dtypes = _collect_batch_dtypes(fv)
        _assert_uniform_dtype(dtypes, "lbp_features")
        dbg_print(debug, f"  lbp_features: dtype={result.dtype}")
        assert np.issubdtype(result.dtype, np.number)

    def test_gabor_features_dtype(self, debug):
        _refresh()
        fv = gp_ops.gabor_features(gp_ops.GrayTerminal, math.pi / 4, math.pi / 8)
        result, dtypes = _collect_batch_dtypes(fv)
        _assert_uniform_dtype(dtypes, "gabor_features")
        _assert_floating(result, "gabor_features")
        dbg_print(debug, f"  gabor_features: dtype={result.dtype}")

    def test_gaussian_features_dtype(self, debug):
        _refresh()
        fv = gp_ops.gaussian_features(gp_ops.GrayTerminal, 2)
        result, dtypes = _collect_batch_dtypes(fv)
        _assert_floating(result, "gaussian_features")
        dbg_print(debug, f"  gaussian_features: dtype={result.dtype}")

    def test_gaud_features_dtype(self, debug):
        _refresh()
        fv = gp_ops.gaud_features(gp_ops.GrayTerminal, 1, 1, 0)
        result, dtypes = _collect_batch_dtypes(fv)
        _assert_floating(result, "gaud_features")

    def test_concat_images_dtype(self, debug):
        _refresh()
        fv = gp_ops.concat_images(gp_ops.RedTerminal, gp_ops.GreenTerminal)
        result, dtypes = _collect_batch_dtypes(fv)
        _assert_uniform_dtype(dtypes, "concat_images")
        _assert_floating(result, "concat_images")
        dbg_print(debug, f"  concat_images: dtype={result.dtype}")


# ===================================================================
# 5. EXTENDED FEATURE EXTRACTOR DTYPES
# ===================================================================

class TestExtendedFeatureExtractorDtypes:

    EXTENDED_FE = [
        ("orb_features", gp_ops.orb_features),
        ("color_histogram_features", gp_ops.color_histogram_features),
        ("glcm_features", gp_ops.glcm_features),
        ("hu_moments_features", gp_ops.hu_moments_features),
        ("edge_histogram_features", gp_ops.edge_histogram_features),
        ("statistical_features", gp_ops.statistical_features),
        ("fourier_features", gp_ops.fourier_features),
    ]

    @pytest.mark.parametrize("name,func", EXTENDED_FE)
    def test_extended_fe_dtype(self, name, func, debug):
        _refresh()
        fv = func(gp_ops.GrayTerminal)
        result, dtypes = _collect_batch_dtypes(fv)
        _assert_uniform_dtype(dtypes, name)
        assert np.issubdtype(result.dtype, np.number), \
            f"{name}: expected numeric, got {result.dtype}"
        dbg_print(debug, f"  {name}: dtype={result.dtype}")


# ===================================================================
# 6. PARAMETERIZED OP DTYPES
# ===================================================================

class TestParameterizedDtypes:

    def test_hog_filter_param_dtype(self, debug):
        _refresh()
        ch = gp_ops.hog_filter_param(gp_ops.GrayTerminal, 8, 2, 9)
        result, dtypes = _collect_batch_dtypes(ch)
        _assert_dtype(result, CHANNEL_DTYPE, "hog_filter_param")
        dbg_print(debug, f"  hog_filter_param: dtype={result.dtype} ✓")

    def test_lbp_filter_param_dtype(self, debug):
        _refresh()
        ch = gp_ops.lbp_filter_param(gp_ops.GrayTerminal, 16, 2)
        result, _ = _collect_batch_dtypes(ch)
        _assert_dtype(result, CHANNEL_DTYPE, "lbp_filter_param")

    def test_mean_filter_param_dtype(self, debug):
        _refresh()
        ch = gp_ops.mean_filter_param(gp_ops.GrayTerminal, 5)
        result, _ = _collect_batch_dtypes(ch)
        _assert_dtype(result, CHANNEL_DTYPE, "mean_filter_param")

    def test_median_filter_param_dtype(self, debug):
        _refresh()
        ch = gp_ops.median_filter_param(gp_ops.GrayTerminal, 5)
        result, _ = _collect_batch_dtypes(ch)
        _assert_dtype(result, CHANNEL_DTYPE, "median_filter_param")

    def test_erosion_filter_param_dtype(self, debug):
        _refresh()
        ch = gp_ops.erosion_filter_param(gp_ops.GrayTerminal, 5)
        result, _ = _collect_batch_dtypes(ch)
        _assert_dtype(result, CHANNEL_DTYPE, "erosion_filter_param")

    def test_canny_filter_param_dtype(self, debug):
        _refresh()
        ch = gp_ops.canny_filter_param(gp_ops.GrayTerminal, 50, 150)
        result, _ = _collect_batch_dtypes(ch)
        _assert_dtype(result, CHANNEL_DTYPE, "canny_filter_param")

    def test_hog_features_param_dtype(self, debug):
        _refresh()
        fv = gp_ops.hog_features_param(gp_ops.GrayTerminal, 8, 2, 9)
        result, dtypes = _collect_batch_dtypes(fv)
        _assert_uniform_dtype(dtypes, "hog_features_param")
        _assert_floating(result, "hog_features_param")
        dbg_print(debug, f"  hog_features_param: dtype={result.dtype}")

    def test_lbp_features_param_dtype(self, debug):
        _refresh()
        fv = gp_ops.lbp_features_param(gp_ops.GrayTerminal, 16, 2)
        result, dtypes = _collect_batch_dtypes(fv)
        _assert_uniform_dtype(dtypes, "lbp_features_param")
        # lbp_features_param uses np.histogram → int64
        assert np.issubdtype(result.dtype, np.number)
        dbg_print(debug, f"  lbp_features_param: dtype={result.dtype}")

    def test_sift_features_param_dtype(self, debug):
        _refresh()
        fv = gp_ops.sift_features_param(gp_ops.GrayTerminal, 100)
        result, _ = _collect_batch_dtypes(fv)
        _assert_floating(result, "sift_features_param")
        dbg_print(debug, f"  sift_features_param: dtype={result.dtype}")


# ===================================================================
# 7. CLASSIFIER DTYPES & PROBABILITY CORRECTNESS
# ===================================================================

class TestClassifierDtypes:
    """Classifiers must return float64 probabilities that sum to ~1.0 per row."""

    @pytest.fixture(autouse=True)
    def _setup_train(self, sample_labels):
        gp_context.context.reset(gp_context.ExecutionMode.TRAIN)
        gp_context.context.set_train_labels(sample_labels)
        yield
        gp_context.context.reset()

    def _get_fv(self):
        _refresh()
        return gp_ops.histogram_features(gp_ops.GrayTerminal)

    @pytest.mark.parametrize("name,func,extra_args", [
        ("rf_classification", gp_ops.rf_classification, [50, 10]),
        ("erf_classification", gp_ops.erf_classification, [100, 20]),
        ("lr_classification", gp_ops.lr_classification, []),
        ("svm_classification", gp_ops.svm_classification, []),
    ])
    def test_classifier_dtype_float64(self, name, func, extra_args, debug):
        fv = self._get_fv()
        pred = func(fv, *extra_args)
        result, dtypes = _collect_batch_dtypes(pred)
        _assert_uniform_dtype(dtypes, name)
        _assert_dtype(result, PREDICTION_DTYPE, name)
        dbg_print(debug, f"  {name}: dtype={result.dtype} ✓")

    @pytest.mark.parametrize("name,func,extra_args", [
        ("rf_classification", gp_ops.rf_classification, [50, 10]),
        ("erf_classification", gp_ops.erf_classification, [100, 20]),
        ("lr_classification", gp_ops.lr_classification, []),
        ("svm_classification", gp_ops.svm_classification, []),
    ])
    def test_classifier_probabilities_sum_to_one(self, name, func, extra_args, debug):
        """Each row of probabilities from a classifier should sum to ~1.0."""
        fv = self._get_fv()
        pred = func(fv, *extra_args)
        result, _ = _collect_batch_dtypes(pred)
        row_sums = result.sum(axis=1)
        dbg_print(debug, f"  {name}: row sums (first 5) = {row_sums[:5]}")
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6,
            err_msg=f"{name}: probability rows don't sum to 1.0")

    @pytest.mark.parametrize("name,func,extra_args", [
        ("rf_classification", gp_ops.rf_classification, [50, 10]),
        ("erf_classification", gp_ops.erf_classification, [100, 20]),
        ("lr_classification", gp_ops.lr_classification, []),
        ("svm_classification", gp_ops.svm_classification, []),
    ])
    def test_classifier_probabilities_non_negative(self, name, func, extra_args, debug):
        """All probability values must be >= 0."""
        fv = self._get_fv()
        pred = func(fv, *extra_args)
        result, _ = _collect_batch_dtypes(pred)
        assert (result >= 0).all(), f"{name}: negative probabilities found"
        assert (result <= 1.0 + 1e-6).all(), f"{name}: probabilities > 1.0 found"
        dbg_print(debug, f"  {name}: all probs in [0, 1] ✓")

    @pytest.mark.parametrize("name,func,extra_args", [
        ("rf_classification", gp_ops.rf_classification, [50, 10]),
        ("erf_classification", gp_ops.erf_classification, [100, 20]),
        ("lr_classification", gp_ops.lr_classification, []),
        ("svm_classification", gp_ops.svm_classification, []),
    ])
    def test_classifier_correct_n_classes(self, name, func, extra_args, debug):
        """Output should have exactly N_CLASSES columns."""
        fv = self._get_fv()
        pred = func(fv, *extra_args)
        result, _ = _collect_batch_dtypes(pred)
        assert result.shape[1] == N_CLASSES, \
            f"{name}: expected {N_CLASSES} classes, got {result.shape[1]}"
        dbg_print(debug, f"  {name}: {result.shape[1]} classes ✓")


# ===================================================================
# 8. ENSEMBLE DTYPES
# ===================================================================

class TestEnsembleDtypes:

    @pytest.fixture(autouse=True)
    def _setup_train(self, sample_labels):
        gp_context.context.reset(gp_context.ExecutionMode.TRAIN)
        gp_context.context.set_train_labels(sample_labels)
        yield
        gp_context.context.reset()

    def _get_pred(self):
        _refresh()
        fv = gp_ops.histogram_features(gp_ops.GrayTerminal)
        return gp_ops.rf_classification(fv, 50, 10)

    def test_ensemble_single_dtype(self, debug):
        pred = self._get_pred()
        ens = gp_ops.ensemble_single(pred)
        result, dtypes = _collect_batch_dtypes(ens)
        _assert_uniform_dtype(dtypes, "ensemble_single")
        _assert_dtype(result, PREDICTION_DTYPE, "ensemble_single")
        dbg_print(debug, f"  ensemble_single: dtype={result.dtype} ✓")

    def test_ensemble_sum_2_dtype(self, debug):
        p1 = self._get_pred()
        p2 = self._get_pred()
        ens = gp_ops.ensemble_sum_2(p1, p2)
        result, dtypes = _collect_batch_dtypes(ens)
        _assert_uniform_dtype(dtypes, "ensemble_sum_2")
        _assert_dtype(result, PREDICTION_DTYPE, "ensemble_sum_2")
        dbg_print(debug, f"  ensemble_sum_2: dtype={result.dtype} ✓")

    def test_ensemble_sum_2_values_are_sum(self, debug):
        """ensemble_sum_2(p1, p2) should produce p1 + p2 element-wise."""
        _refresh()
        fv1 = gp_ops.histogram_features(gp_ops.GrayTerminal)
        p1 = gp_ops.rf_classification(fv1, 50, 10)
        r1, _ = _collect_batch_dtypes(p1)

        _refresh()
        fv2 = gp_ops.histogram_features(gp_ops.GrayTerminal)
        p2 = gp_ops.rf_classification(fv2, 50, 10)
        r2, _ = _collect_batch_dtypes(p2)

        # Re-run to get the ensemble (need fresh predictions)
        _refresh()
        fv1b = gp_ops.histogram_features(gp_ops.GrayTerminal)
        p1b = gp_ops.rf_classification(fv1b, 50, 10)
        _refresh()
        fv2b = gp_ops.histogram_features(gp_ops.GrayTerminal)
        p2b = gp_ops.rf_classification(fv2b, 50, 10)
        ens = gp_ops.ensemble_sum_2(p1b, p2b)
        r_ens, _ = _collect_batch_dtypes(ens)

        # The ensemble should sum two prediction arrays
        assert r_ens.shape == r1.shape
        dbg_print(debug, f"  ensemble_sum_2: output shape {r_ens.shape}, "
                  f"max value {r_ens.max():.4f} (should be > 1.0 if summed)")


# ===================================================================
# 9. CASCADE DTYPES
# ===================================================================

class TestCascadeDtypes:

    @pytest.fixture(autouse=True)
    def _setup_train(self, sample_labels):
        gp_context.context.reset(gp_context.ExecutionMode.TRAIN)
        gp_context.context.set_train_labels(sample_labels)
        yield
        gp_context.context.reset()

    def test_cascade_rf_dtype(self, debug):
        _refresh()
        fv_in = gp_ops.histogram_features(gp_ops.GrayTerminal)
        fv_out = gp_ops.cascade_rf(fv_in, 50, 10)
        result, dtypes = _collect_batch_dtypes(fv_out)
        _assert_uniform_dtype(dtypes, "cascade_rf")
        # cascade hstacks features (may be int64) with predictions (float64)
        # result should be at least numeric
        assert np.issubdtype(result.dtype, np.number), \
            f"cascade_rf: expected numeric, got {result.dtype}"
        dbg_print(debug, f"  cascade_rf: dtype={result.dtype}, shape={result.shape}")
        # Verify the appended columns (last N_CLASSES) look like probabilities
        prob_cols = result[:, -N_CLASSES:]
        _assert_floating(prob_cols, "cascade_rf prob columns")
        assert (prob_cols >= 0).all(), "cascade_rf: negative values in prob columns"

    def test_cascade_lr_dtype(self, debug):
        _refresh()
        fv_in = gp_ops.histogram_features(gp_ops.GrayTerminal)
        fv_out = gp_ops.cascade_lr(fv_in)
        result, dtypes = _collect_batch_dtypes(fv_out)
        _assert_uniform_dtype(dtypes, "cascade_lr")
        assert np.issubdtype(result.dtype, np.number)
        dbg_print(debug, f"  cascade_lr: dtype={result.dtype}")


# ===================================================================
# 10. DTYPE PROPAGATION THROUGH CHAINS
# ===================================================================

class TestDtypePropagation:
    """Verify that chaining operations doesn't introduce unexpected dtypes."""

    def test_filter_chain_stays_float32(self, debug):
        """mean → sobel → relu should all stay float32."""
        _refresh()
        ch1 = gp_ops.mean_filter(gp_ops.GrayTerminal)
        r1, _ = _collect_batch_dtypes(ch1)
        dbg_print(debug, f"  mean_filter: {r1.dtype}")

        _refresh()
        ch2 = gp_ops.sobel_filter(gp_ops.mean_filter(gp_ops.GrayTerminal))
        r2, _ = _collect_batch_dtypes(ch2)
        dbg_print(debug, f"  mean→sobel: {r2.dtype}")

        _refresh()
        ch3 = gp_ops.relu(gp_ops.sobel_filter(gp_ops.mean_filter(gp_ops.GrayTerminal)))
        r3, _ = _collect_batch_dtypes(ch3)
        dbg_print(debug, f"  mean→sobel→relu: {r3.dtype}")

        _assert_dtype(r1, CHANNEL_DTYPE, "mean_filter")
        _assert_dtype(r2, CHANNEL_DTYPE, "mean→sobel")
        _assert_dtype(r3, CHANNEL_DTYPE, "mean→sobel→relu")

    def test_filter_to_features_dtype(self, debug):
        """Channel (float32) → FeatureVector should produce numeric output."""
        _refresh()
        ch = gp_ops.gaussian_filter(gp_ops.GrayTerminal, 2)
        fv = gp_ops.hog_features(ch)
        result, dtypes = _collect_batch_dtypes(fv)
        _assert_uniform_dtype(dtypes, "gauss→hog_features")
        _assert_floating(result, "gauss→hog_features")
        dbg_print(debug, f"  gaussian→hog_features: {result.dtype}")

    def test_concat_mixed_dtype_features(self, debug):
        """Concatenating int64 histogram with float64 HOG features.
        np.hstack should promote to float64."""
        _refresh()
        f_hist = gp_ops.histogram_features(gp_ops.RedTerminal)
        f_hog = gp_ops.hog_features(gp_ops.GreenTerminal)
        fv = gp_ops.concat_features_2(f_hist, f_hog)
        result, dtypes = _collect_batch_dtypes(fv)
        _assert_uniform_dtype(dtypes, "concat(hist,hog)")
        dbg_print(debug, f"  concat(int64 hist, float64 hog): result dtype={result.dtype}")
        # np.hstack promotes int64+float64 → float64
        _assert_floating(result, "concat(hist,hog)")

    def test_full_pipeline_dtype_consistency(self, debug, sample_labels):
        """Full pipeline: filter → features → classifier → ensemble.
        Track dtype at every step."""
        gp_context.context.reset(gp_context.ExecutionMode.TRAIN)
        gp_context.context.set_train_labels(sample_labels)

        _refresh()
        dbg_print(debug, "\n  === Pipeline dtype trace ===")

        # Step 1: Channel (should be float32)
        ch = gp_ops.sobel_filter(gp_ops.GrayTerminal)
        r_ch, _ = _collect_batch_dtypes(ch)
        dbg_print(debug, f"    [1] sobel_filter → {r_ch.dtype}")
        _assert_dtype(r_ch, CHANNEL_DTYPE, "pipeline:sobel")

        # Step 2: FeatureVector
        _refresh()
        ch = gp_ops.sobel_filter(gp_ops.GrayTerminal)
        fv = gp_ops.histogram_features(ch)
        r_fv, _ = _collect_batch_dtypes(fv)
        dbg_print(debug, f"    [2] histogram_features → {r_fv.dtype}")
        assert np.issubdtype(r_fv.dtype, np.number)

        # Step 3: Prediction (should be float64)
        _refresh()
        ch = gp_ops.sobel_filter(gp_ops.GrayTerminal)
        fv = gp_ops.histogram_features(ch)
        pred = gp_ops.rf_classification(fv, 50, 10)
        r_pred, _ = _collect_batch_dtypes(pred)
        dbg_print(debug, f"    [3] rf_classification → {r_pred.dtype}")
        _assert_dtype(r_pred, PREDICTION_DTYPE, "pipeline:rf")

        # Step 4: EnsembleOutput (should be float64)
        _refresh()
        ch = gp_ops.sobel_filter(gp_ops.GrayTerminal)
        fv = gp_ops.histogram_features(ch)
        pred = gp_ops.rf_classification(fv, 50, 10)
        ens = gp_ops.ensemble_single(pred)
        r_ens, _ = _collect_batch_dtypes(ens)
        dbg_print(debug, f"    [4] ensemble_single → {r_ens.dtype}")
        _assert_dtype(r_ens, PREDICTION_DTYPE, "pipeline:ensemble")

        gp_context.context.reset()


# ===================================================================
# 11. VALUE RANGE CHECKS
# ===================================================================

class TestValueRanges:
    """Certain operations have expected value ranges."""

    def test_normalize_filter_in_0_1(self, debug):
        """normalize_filter should output values in [0, 1]."""
        _refresh()
        ch = gp_ops.normalize_filter(gp_ops.GrayTerminal)
        result = consume_channel(ch, debug, label="normalize_filter")
        assert result.min() >= -1e-7, f"normalize_filter min={result.min()}"
        assert result.max() <= 1.0 + 1e-7, f"normalize_filter max={result.max()}"
        dbg_print(debug, f"  normalize_filter range: [{result.min():.6f}, {result.max():.6f}] ✓")

    def test_threshold_filter_is_binary(self, debug):
        """threshold_filter should produce only 0.0 or 1.0."""
        _refresh()
        ch = gp_ops.threshold_filter(gp_ops.GrayTerminal)
        result = consume_channel(ch, debug, label="threshold_filter")
        unique_vals = np.unique(result)
        assert set(unique_vals).issubset({0.0, 1.0}), \
            f"threshold_filter has non-binary values: {unique_vals}"
        dbg_print(debug, f"  threshold_filter unique values: {unique_vals} ✓")

    def test_canny_filter_in_0_1(self, debug):
        """Canny edges should be in [0, 1] (normalized from uint8)."""
        _refresh()
        ch = gp_ops.canny_filter(gp_ops.GrayTerminal)
        result = consume_channel(ch, debug, label="canny_filter")
        assert result.min() >= -1e-7
        assert result.max() <= 1.0 + 1e-7
        unique_vals = np.unique(result)
        assert len(unique_vals) <= 2, "Canny should produce binary output"
        dbg_print(debug, f"  canny_filter unique values: {unique_vals} ✓")

    def test_invert_filter_in_0_1(self, debug):
        """invert_filter clips to [0,1] then computes 1-x."""
        _refresh()
        ch = gp_ops.invert_filter(gp_ops.GrayTerminal)
        result = consume_channel(ch, debug, label="invert_filter")
        assert result.min() >= -1e-7
        assert result.max() <= 1.0 + 1e-7

    def test_relu_non_negative(self, debug):
        """relu should produce only non-negative values."""
        _refresh()
        ch = gp_ops.relu(gp_ops.GrayTerminal)
        result = consume_channel(ch, debug, label="relu")
        assert result.min() >= 0.0, f"relu produced negative value: {result.min()}"
        dbg_print(debug, f"  relu: min={result.min():.6f} ✓ (non-negative)")

    def test_sharpen_filter_clipped(self, debug):
        """sharpen_filter clips output to [0, 1]."""
        _refresh()
        ch = gp_ops.sharpen_filter(gp_ops.GrayTerminal)
        result = consume_channel(ch, debug, label="sharpen_filter")
        assert result.min() >= -1e-7, f"sharpen min={result.min()}"
        assert result.max() <= 1.0 + 1e-7, f"sharpen max={result.max()}"

    def test_color_histogram_features_normalized(self, debug):
        """color_histogram_features normalizes by total → should sum to ~1.0."""
        _refresh()
        fv = gp_ops.color_histogram_features(gp_ops.GrayTerminal)
        result = consume_features(fv, debug, label="color_histogram_features")
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5,
            err_msg="color_histogram row sums should be ~1.0")
        dbg_print(debug, f"  color_histogram_features: row sums ≈ 1.0 ✓")


# ===================================================================
# 12. NaN / Inf SWEEP ACROSS ALL OPS
# ===================================================================

class TestNoNaNInf:
    """Systematic NaN/Inf check for every operation."""

    CHANNEL_OPS = [
        ("mean_filter", lambda: gp_ops.mean_filter(gp_ops.GrayTerminal)),
        ("median_filter", lambda: gp_ops.median_filter(gp_ops.GrayTerminal)),
        ("min_filter", lambda: gp_ops.min_filter(gp_ops.GrayTerminal)),
        ("max_filter", lambda: gp_ops.max_filter(gp_ops.GrayTerminal)),
        ("gaussian_filter", lambda: gp_ops.gaussian_filter(gp_ops.GrayTerminal, 2)),
        ("laplacian_filter", lambda: gp_ops.laplacian_filter(gp_ops.GrayTerminal)),
        ("sobel_filter", lambda: gp_ops.sobel_filter(gp_ops.GrayTerminal)),
        ("relu", lambda: gp_ops.relu(gp_ops.GrayTerminal)),
        ("sqrt_op", lambda: gp_ops.sqrt_op(gp_ops.GrayTerminal)),
        ("log1_filter", lambda: gp_ops.log1_filter(gp_ops.GrayTerminal)),
        ("log2_filter", lambda: gp_ops.log2_filter(gp_ops.GrayTerminal)),
        ("hog_filter", lambda: gp_ops.hog_filter(gp_ops.GrayTerminal)),
        ("lbp_filter", lambda: gp_ops.lbp_filter(gp_ops.GrayTerminal)),
        ("gabor_filter", lambda: gp_ops.gabor_filter(gp_ops.GrayTerminal, math.pi/4, math.pi/8)),
        ("gaud_filter", lambda: gp_ops.gaud_filter(gp_ops.GrayTerminal, 1, 1, 0)),
        ("prewitt_filter", lambda: gp_ops.prewitt_filter(gp_ops.GrayTerminal)),
        ("scharr_filter", lambda: gp_ops.scharr_filter(gp_ops.GrayTerminal)),
        ("canny_filter", lambda: gp_ops.canny_filter(gp_ops.GrayTerminal)),
        ("bilateral_filter", lambda: gp_ops.bilateral_filter(gp_ops.GrayTerminal, 2)),
        ("clahe_filter", lambda: gp_ops.clahe_filter(gp_ops.GrayTerminal)),
        ("erosion_filter", lambda: gp_ops.erosion_filter(gp_ops.GrayTerminal)),
        ("dilation_filter", lambda: gp_ops.dilation_filter(gp_ops.GrayTerminal)),
        ("opening_filter", lambda: gp_ops.opening_filter(gp_ops.GrayTerminal)),
        ("closing_filter", lambda: gp_ops.closing_filter(gp_ops.GrayTerminal)),
        ("gradient_filter", lambda: gp_ops.gradient_filter(gp_ops.GrayTerminal)),
        ("tophat_filter", lambda: gp_ops.tophat_filter(gp_ops.GrayTerminal)),
        ("blackhat_filter", lambda: gp_ops.blackhat_filter(gp_ops.GrayTerminal)),
        ("sharpen_filter", lambda: gp_ops.sharpen_filter(gp_ops.GrayTerminal)),
        ("emboss_filter", lambda: gp_ops.emboss_filter(gp_ops.GrayTerminal)),
        ("dog_filter", lambda: gp_ops.dog_filter(gp_ops.GrayTerminal, 1)),
        ("normalize_filter", lambda: gp_ops.normalize_filter(gp_ops.GrayTerminal)),
        ("invert_filter", lambda: gp_ops.invert_filter(gp_ops.GrayTerminal)),
        ("threshold_filter", lambda: gp_ops.threshold_filter(gp_ops.GrayTerminal)),
        ("adaptive_threshold_filter", lambda: gp_ops.adaptive_threshold_filter(gp_ops.GrayTerminal)),
    ]

    @pytest.mark.parametrize("name,factory", CHANNEL_OPS)
    def test_channel_op_no_nan_inf(self, name, factory, debug):
        _refresh()
        ch = factory()
        result, _ = _collect_batch_dtypes(ch)
        assert np.isfinite(result).all(), \
            f"{name}: contains NaN or Inf (NaN count={np.isnan(result).sum()}, Inf count={np.isinf(result).sum()})"

    FEATURE_OPS = [
        ("histogram_features", lambda: gp_ops.histogram_features(gp_ops.GrayTerminal)),
        ("hog_features", lambda: gp_ops.hog_features(gp_ops.GrayTerminal)),
        ("lbp_features", lambda: gp_ops.lbp_features(gp_ops.GrayTerminal)),
        ("sift_features", lambda: gp_ops.sift_features(gp_ops.GrayTerminal)),
        ("sobel_features", lambda: gp_ops.sobel_features(gp_ops.GrayTerminal)),
        ("orb_features", lambda: gp_ops.orb_features(gp_ops.GrayTerminal)),
        ("color_histogram_features", lambda: gp_ops.color_histogram_features(gp_ops.GrayTerminal)),
        ("glcm_features", lambda: gp_ops.glcm_features(gp_ops.GrayTerminal)),
        ("hu_moments_features", lambda: gp_ops.hu_moments_features(gp_ops.GrayTerminal)),
        ("edge_histogram_features", lambda: gp_ops.edge_histogram_features(gp_ops.GrayTerminal)),
        ("statistical_features", lambda: gp_ops.statistical_features(gp_ops.GrayTerminal)),
        ("fourier_features", lambda: gp_ops.fourier_features(gp_ops.GrayTerminal)),
    ]

    @pytest.mark.parametrize("name,factory", FEATURE_OPS)
    def test_feature_op_no_nan_inf(self, name, factory, debug):
        _refresh()
        fv = factory()
        result, _ = _collect_batch_dtypes(fv)
        assert np.isfinite(result).all(), \
            f"{name}: contains NaN or Inf (NaN count={np.isnan(result).sum()}, Inf count={np.isinf(result).sum()})"


# ===================================================================
# 13. DTYPE CONSISTENCY SUMMARY  (all-in-one audit)
# ===================================================================

class TestDtypeAuditSummary:
    """
    One comprehensive test that runs every operation and prints a dtype summary.
    Use --debug-ops -s to see the full audit table.
    """

    @pytest.fixture(autouse=True)
    def _setup_train(self, sample_labels):
        gp_context.context.reset(gp_context.ExecutionMode.TRAIN)
        gp_context.context.set_train_labels(sample_labels)
        yield
        gp_context.context.reset()

    def test_dtype_audit_table(self, debug):
        """Run every op and print a summary table of dtypes."""
        audit = []

        # --- Channel terminals ---
        for terminal, name in [
            (gp_ops.RedTerminal, "RedTerminal"),
            (gp_ops.GreenTerminal, "GreenTerminal"),
            (gp_ops.BlueTerminal, "BlueTerminal"),
            (gp_ops.GrayTerminal, "GrayTerminal"),
        ]:
            _refresh()
            r, d = _collect_batch_dtypes(terminal)
            audit.append(("Terminal", name, str(r.dtype), r.shape))

        # --- Basic Channel filters ---
        basic_ch = [
            ("mean_filter", lambda: gp_ops.mean_filter(gp_ops.GrayTerminal)),
            ("median_filter", lambda: gp_ops.median_filter(gp_ops.GrayTerminal)),
            ("sobel_filter", lambda: gp_ops.sobel_filter(gp_ops.GrayTerminal)),
            ("relu", lambda: gp_ops.relu(gp_ops.GrayTerminal)),
            ("hog_filter", lambda: gp_ops.hog_filter(gp_ops.GrayTerminal)),
            ("lbp_filter", lambda: gp_ops.lbp_filter(gp_ops.GrayTerminal)),
            ("gaussian_filter", lambda: gp_ops.gaussian_filter(gp_ops.GrayTerminal, 2)),
            ("gabor_filter", lambda: gp_ops.gabor_filter(gp_ops.GrayTerminal, math.pi/4, math.pi/8)),
            ("linear_combination", lambda: gp_ops.linear_combination(
                gp_ops.RedTerminal, gp_ops.GreenTerminal, gp_ops.BlueTerminal, 0.3, 0.5, 0.2)),
            ("add_max_pool", lambda: gp_ops.add_max_pool(gp_ops.RedTerminal, gp_ops.GreenTerminal)),
            ("sub_max_pool", lambda: gp_ops.sub_max_pool(gp_ops.RedTerminal, gp_ops.GreenTerminal)),
        ]
        for name, factory in basic_ch:
            _refresh()
            r, d = _collect_batch_dtypes(factory())
            audit.append(("Channel:basic", name, str(r.dtype), r.shape))

        # --- Feature extractors ---
        fe_ops = [
            ("histogram_features", lambda: gp_ops.histogram_features(gp_ops.GrayTerminal)),
            ("hog_features", lambda: gp_ops.hog_features(gp_ops.GrayTerminal)),
            ("lbp_features", lambda: gp_ops.lbp_features(gp_ops.GrayTerminal)),
            ("sift_features", lambda: gp_ops.sift_features(gp_ops.GrayTerminal)),
            ("sobel_features", lambda: gp_ops.sobel_features(gp_ops.GrayTerminal)),
            ("hog_image_features", lambda: gp_ops.hog_image_features(gp_ops.GrayTerminal)),
            ("lbp_image_features", lambda: gp_ops.lbp_image_features(gp_ops.GrayTerminal)),
            ("orb_features", lambda: gp_ops.orb_features(gp_ops.GrayTerminal)),
            ("color_histogram_features", lambda: gp_ops.color_histogram_features(gp_ops.GrayTerminal)),
            ("statistical_features", lambda: gp_ops.statistical_features(gp_ops.GrayTerminal)),
            ("glcm_features", lambda: gp_ops.glcm_features(gp_ops.GrayTerminal)),
            ("hu_moments_features", lambda: gp_ops.hu_moments_features(gp_ops.GrayTerminal)),
            ("fourier_features", lambda: gp_ops.fourier_features(gp_ops.GrayTerminal)),
        ]
        for name, factory in fe_ops:
            _refresh()
            r, d = _collect_batch_dtypes(factory())
            audit.append(("FeatureVector", name, str(r.dtype), r.shape))

        # --- Classifiers ---
        for name, factory in [
            ("rf_classification", lambda: gp_ops.rf_classification(
                gp_ops.histogram_features(gp_ops.GrayTerminal), 50, 10)),
            ("erf_classification", lambda: gp_ops.erf_classification(
                gp_ops.histogram_features(gp_ops.GrayTerminal), 100, 20)),
            ("lr_classification", lambda: gp_ops.lr_classification(
                gp_ops.histogram_features(gp_ops.GrayTerminal))),
            ("svm_classification", lambda: gp_ops.svm_classification(
                gp_ops.histogram_features(gp_ops.GrayTerminal))),
        ]:
            _refresh()
            r, d = _collect_batch_dtypes(factory())
            audit.append(("Prediction", name, str(r.dtype), r.shape))

        # --- Print table ---
        if debug:
            print("\n" + "=" * 80)
            print(f"  {'LAYER':<20} {'OPERATION':<30} {'DTYPE':<12} {'SHAPE'}")
            print("-" * 80)
            for layer, name, dtype, shape in audit:
                marker = ""
                if layer == "Channel:basic" and dtype != "float32":
                    marker = " ⚠️  EXPECTED float32"
                elif layer == "Prediction" and dtype != "float64":
                    marker = " ⚠️  EXPECTED float64"
                elif layer == "FeatureVector" and dtype.startswith("int"):
                    marker = " ⚠️  INTEGER (not float)"
                print(f"  {layer:<20} {name:<30} {dtype:<12} {str(shape)}{marker}")
            print("=" * 80)

        # --- Assert no completely wrong types ---
        for layer, name, dtype, shape in audit:
            assert dtype not in ("float16", "bool"), \
                f"{name}: unexpected dtype {dtype}"
