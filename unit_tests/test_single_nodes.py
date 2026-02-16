"""
Tests for individual GP nodes: filters, feature extractors, classifiers, ensembles.

Each test feeds a known input to a single node and checks:
  - output type (Channel / FeatureVector / Prediction / EnsembleOutput)
  - output shape  (N preserved, spatial dims reasonable, feature dim > 0)
  - no NaN / Inf

Run with debug output:
    pytest unit_tests/test_single_nodes.py --debug-ops -s
"""
import math
import numpy as np
import pytest

from src import gp_ops, gp_types, gp_context

# Re-use helpers from conftest (pytest auto-discovers them)
from unit_tests.conftest import (
    dbg_print, consume_channel, consume_features, consume_prediction,
    fresh_image_setup, _make_random_images, _make_labels,
    N_SAMPLES, IMG_H, IMG_W, N_CLASSES, BATCH_SIZE,
)


# ===================================================================
# Helper: refresh image so terminals are re-iterable
# ===================================================================
def _refresh(X=None):
    if X is None:
        X = _make_random_images()
    fresh_image_setup(X)
    return X


# ===================================================================
# 1. CHANNEL TERMINALS
# ===================================================================

class TestChannelTerminals:
    """Verify that GetRed / GetGreen / GetBlue / GetGray produce (N, H, W)."""

    @pytest.mark.parametrize("terminal,name", [
        (gp_ops.RedTerminal, "Red"),
        (gp_ops.GreenTerminal, "Green"),
        (gp_ops.BlueTerminal, "Blue"),
        (gp_ops.GrayTerminal, "Gray"),
    ])
    def test_terminal_shape(self, terminal, name, debug):
        X = _refresh()
        dbg_print(debug, f"\n--- {name} Terminal ---")
        result = consume_channel(terminal, debug, label=name)
        assert result.shape == (N_SAMPLES, IMG_H, IMG_W), \
            f"{name} terminal shape mismatch: {result.shape}"
        assert np.isfinite(result).all(), f"{name} contains NaN/Inf"

    def test_gray_is_weighted_sum(self, debug):
        X = _refresh()
        gray = consume_channel(gp_ops.GrayTerminal, debug, label="Gray")
        _refresh(X)
        red = consume_channel(gp_ops.RedTerminal, debug, label="Red")
        _refresh(X)
        green = consume_channel(gp_ops.GreenTerminal, debug, label="Green")
        _refresh(X)
        blue = consume_channel(gp_ops.BlueTerminal, debug, label="Blue")
        expected = 0.299 * red + 0.587 * green + 0.114 * blue
        np.testing.assert_allclose(gray, expected, atol=1e-5)
        dbg_print(debug, "  Gray = 0.299R + 0.587G + 0.114B ✓")


# ===================================================================
# 2. IMAGE FILTERING OPERATIONS  (Channel → Channel)
# ===================================================================

class TestBasicFilters:
    """Each filter should preserve (N, H, W) and be finite."""

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
    def test_unary_filter(self, name, func, kwargs, debug):
        _refresh()
        dbg_print(debug, f"\n--- Filter: {name} ---")
        channel_in = gp_ops.GrayTerminal
        channel_out = func(channel_in, **kwargs)
        assert isinstance(channel_out, gp_types.Channel)
        result = consume_channel(channel_out, debug, label=name)
        assert result.shape[0] == N_SAMPLES, f"{name}: N mismatch"
        assert result.ndim == 3, f"{name}: expected 3-d output"
        assert np.isfinite(result).all(), f"{name}: NaN/Inf detected"

    def test_gaussian_filter(self, debug):
        _refresh()
        dbg_print(debug, "\n--- Filter: gaussian_filter ---")
        ch = gp_ops.gaussian_filter(gp_ops.GrayTerminal, 2)
        result = consume_channel(ch, debug, label="gaussian_filter")
        assert result.shape == (N_SAMPLES, IMG_H, IMG_W)

    def test_gabor_filter(self, debug):
        _refresh()
        dbg_print(debug, "\n--- Filter: gabor_filter ---")
        theta = math.pi / 4
        freq = math.pi / 8
        ch = gp_ops.gabor_filter(gp_ops.GrayTerminal, theta, freq)
        result = consume_channel(ch, debug, label="gabor_filter")
        assert result.shape == (N_SAMPLES, IMG_H, IMG_W)

    def test_gaud_filter(self, debug):
        _refresh()
        dbg_print(debug, "\n--- Filter: gaud_filter ---")
        ch = gp_ops.gaud_filter(gp_ops.GrayTerminal, 1, 1, 0)
        result = consume_channel(ch, debug, label="gaud_filter")
        assert result.shape == (N_SAMPLES, IMG_H, IMG_W)

    def test_linear_combination(self, debug):
        _refresh()
        dbg_print(debug, "\n--- linear_combination ---")
        ch = gp_ops.linear_combination(
            gp_ops.RedTerminal, gp_ops.GreenTerminal, gp_ops.BlueTerminal,
            0.3, 0.5, 0.2,
        )
        result = consume_channel(ch, debug, label="linear_combination")
        assert result.shape == (N_SAMPLES, IMG_H, IMG_W)

    def test_add_max_pool(self, debug):
        _refresh()
        dbg_print(debug, "\n--- add_max_pool ---")
        ch = gp_ops.add_max_pool(gp_ops.RedTerminal, gp_ops.GreenTerminal)
        result = consume_channel(ch, debug, label="add_max_pool")
        # max-pool halves spatial dims
        assert result.shape[0] == N_SAMPLES
        assert result.shape[1] == IMG_H // 2
        assert result.shape[2] == IMG_W // 2

    def test_sub_max_pool(self, debug):
        _refresh()
        dbg_print(debug, "\n--- sub_max_pool ---")
        ch = gp_ops.sub_max_pool(gp_ops.RedTerminal, gp_ops.GreenTerminal)
        result = consume_channel(ch, debug, label="sub_max_pool")
        assert result.shape[0] == N_SAMPLES


# ===================================================================
# 3. EXTENDED IMAGE FILTERS  (Channel → Channel)
# ===================================================================

class TestExtendedFilters:
    """Extended filter ops (morphological, edge, etc.)."""

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
    def test_extended_unary(self, name, func, debug):
        _refresh()
        dbg_print(debug, f"\n--- Extended Filter: {name} ---")
        ch = func(gp_ops.GrayTerminal)
        result = consume_channel(ch, debug, label=name)
        assert result.shape[0] == N_SAMPLES
        assert result.ndim == 3
        assert np.isfinite(result).all(), f"{name}: NaN/Inf"

    def test_bilateral_filter(self, debug):
        _refresh()
        dbg_print(debug, "\n--- bilateral_filter ---")
        ch = gp_ops.bilateral_filter(gp_ops.GrayTerminal, 2)
        result = consume_channel(ch, debug, label="bilateral_filter")
        assert result.shape == (N_SAMPLES, IMG_H, IMG_W)

    def test_dog_filter(self, debug):
        _refresh()
        dbg_print(debug, "\n--- dog_filter ---")
        ch = gp_ops.dog_filter(gp_ops.GrayTerminal, 1)
        result = consume_channel(ch, debug, label="dog_filter")
        assert result.shape == (N_SAMPLES, IMG_H, IMG_W)


# ===================================================================
# 4. FEATURE EXTRACTION  (Channel → FeatureVector)
# ===================================================================

class TestFeatureExtraction:
    """Each feature extractor should produce (N, D) with D > 0."""

    BASIC_FE = [
        ("histogram_features", gp_ops.histogram_features, {}, 256),
        ("hog_features", gp_ops.hog_features, {}, None),      # HOG dim depends on image size
        ("lbp_features", gp_ops.lbp_features, {}, None),
        ("sift_features", gp_ops.sift_features, {}, 128),
        ("hog_image_features", gp_ops.hog_image_features, {}, None),
        ("lbp_image_features", gp_ops.lbp_image_features, {}, IMG_H * IMG_W),
        ("sobel_features", gp_ops.sobel_features, {}, IMG_H * IMG_W),
    ]

    @pytest.mark.parametrize("name,func,kwargs,expected_dim", BASIC_FE)
    def test_basic_fe(self, name, func, kwargs, expected_dim, debug):
        _refresh()
        dbg_print(debug, f"\n--- Feature Extraction: {name} ---")
        fv = func(gp_ops.GrayTerminal, **kwargs)
        assert isinstance(fv, gp_types.FeatureVector)
        result = consume_features(fv, debug, label=name)
        assert result.shape[0] == N_SAMPLES, f"{name}: N mismatch"
        assert result.ndim == 2, f"{name}: expected 2-d"
        assert result.shape[1] > 0, f"{name}: feature dim is 0"
        if expected_dim is not None:
            assert result.shape[1] == expected_dim, \
                f"{name}: expected dim {expected_dim}, got {result.shape[1]}"
        dbg_print(debug, f"  feature_dim = {result.shape[1]}")

    def test_gabor_features(self, debug):
        _refresh()
        dbg_print(debug, "\n--- gabor_features ---")
        fv = gp_ops.gabor_features(gp_ops.GrayTerminal, math.pi / 4, math.pi / 8)
        result = consume_features(fv, debug, label="gabor_features")
        assert result.shape == (N_SAMPLES, IMG_H * IMG_W)

    def test_gaussian_features(self, debug):
        _refresh()
        dbg_print(debug, "\n--- gaussian_features ---")
        fv = gp_ops.gaussian_features(gp_ops.GrayTerminal, 2)
        result = consume_features(fv, debug, label="gaussian_features")
        assert result.shape == (N_SAMPLES, IMG_H * IMG_W)

    def test_gaud_features(self, debug):
        _refresh()
        dbg_print(debug, "\n--- gaud_features ---")
        fv = gp_ops.gaud_features(gp_ops.GrayTerminal, 1, 1, 0)
        result = consume_features(fv, debug, label="gaud_features")
        assert result.shape == (N_SAMPLES, IMG_H * IMG_W)

    def test_concat_images(self, debug):
        _refresh()
        dbg_print(debug, "\n--- concat_images ---")
        fv = gp_ops.concat_images(gp_ops.RedTerminal, gp_ops.GreenTerminal)
        result = consume_features(fv, debug, label="concat_images")
        assert result.shape == (N_SAMPLES, 2 * IMG_H * IMG_W)


# ===================================================================
# 5. EXTENDED FEATURE EXTRACTION
# ===================================================================

class TestExtendedFeatureExtraction:
    EXTENDED_FE = [
        ("orb_features", gp_ops.orb_features, 32),
        ("color_histogram_features", gp_ops.color_histogram_features, 64),
        ("glcm_features", gp_ops.glcm_features, 20),
        ("hu_moments_features", gp_ops.hu_moments_features, 7),
        ("edge_histogram_features", gp_ops.edge_histogram_features, 8),
        ("statistical_features", gp_ops.statistical_features, 10),
        ("fourier_features", gp_ops.fourier_features, 64),
    ]

    @pytest.mark.parametrize("name,func,expected_dim", EXTENDED_FE)
    def test_extended_fe(self, name, func, expected_dim, debug):
        _refresh()
        dbg_print(debug, f"\n--- Extended FE: {name} ---")
        fv = func(gp_ops.GrayTerminal)
        result = consume_features(fv, debug, label=name)
        assert result.shape[0] == N_SAMPLES
        assert result.shape[1] == expected_dim, \
            f"{name}: expected {expected_dim}, got {result.shape[1]}"


# ===================================================================
# 6. FEATURE CONCATENATION
# ===================================================================

class TestFeatureConcatenation:

    def test_concat_features_2(self, debug):
        _refresh()
        dbg_print(debug, "\n--- concat_features_2 ---")
        f1 = gp_ops.histogram_features(gp_ops.RedTerminal)
        f2 = gp_ops.histogram_features(gp_ops.GreenTerminal)
        fv = gp_ops.concat_features_2(f1, f2)
        result = consume_features(fv, debug, label="concat_2")
        assert result.shape == (N_SAMPLES, 256 + 256)

    def test_concat_features_3(self, debug):
        _refresh()
        dbg_print(debug, "\n--- concat_features_3 ---")
        f1 = gp_ops.histogram_features(gp_ops.RedTerminal)
        f2 = gp_ops.histogram_features(gp_ops.GreenTerminal)
        f3 = gp_ops.histogram_features(gp_ops.BlueTerminal)
        fv = gp_ops.concat_features_3(f1, f2, f3)
        result = consume_features(fv, debug, label="concat_3")
        assert result.shape == (N_SAMPLES, 256 * 3)

    def test_concat_features_4(self, debug):
        _refresh()
        dbg_print(debug, "\n--- concat_features_4 ---")
        f1 = gp_ops.histogram_features(gp_ops.RedTerminal)
        f2 = gp_ops.histogram_features(gp_ops.GreenTerminal)
        f3 = gp_ops.histogram_features(gp_ops.BlueTerminal)
        f4 = gp_ops.histogram_features(gp_ops.GrayTerminal)
        fv = gp_ops.concat_features_4(f1, f2, f3, f4)
        result = consume_features(fv, debug, label="concat_4")
        assert result.shape == (N_SAMPLES, 256 * 4)


# ===================================================================
# 7. CLASSIFICATION HEADS  (FeatureVector → Prediction)
# ===================================================================

class TestClassifiers:
    """
    Each classifier trains on features + labels, produces (N, n_classes) probs.
    """

    @pytest.fixture(autouse=True)
    def _setup_train(self, sample_labels):
        gp_context.context.reset(gp_context.ExecutionMode.TRAIN)
        gp_context.context.set_train_labels(sample_labels)
        yield
        gp_context.context.reset()

    def _get_histogram_fv(self):
        """Quick feature vector: histogram of gray channel."""
        _refresh()
        return gp_ops.hog_features(gp_ops.GrayTerminal)

    def test_rf_classification(self, debug):
        dbg_print(debug, "\n--- rf_classification ---")
        fv = self._get_histogram_fv()
        pred = gp_ops.rf_classification(fv, 50, 10)
        assert isinstance(pred, gp_types.Prediction)
        result = consume_prediction(pred, debug, label="rf")
        assert result.shape[0] == N_SAMPLES
        assert result.shape[1] == N_CLASSES
        assert np.isfinite(result).all()
        dbg_print(debug, f"  prob sums per sample: {result.sum(axis=1)[:5]}")

    def test_erf_classification(self, debug):
        dbg_print(debug, "\n--- erf_classification ---")
        fv = self._get_histogram_fv()
        pred = gp_ops.erf_classification(fv, 100, 20)
        result = consume_prediction(pred, debug, label="erf")
        assert result.shape == (N_SAMPLES, N_CLASSES)

    def test_lr_classification(self, debug):
        dbg_print(debug, "\n--- lr_classification ---")
        fv = self._get_histogram_fv()
        pred = gp_ops.lr_classification(fv)
        result = consume_prediction(pred, debug, label="lr")
        assert result.shape == (N_SAMPLES, N_CLASSES)

    def test_svm_classification(self, debug):
        dbg_print(debug, "\n--- svm_classification ---")
        fv = self._get_histogram_fv()
        pred = gp_ops.svm_classification(fv)
        result = consume_prediction(pred, debug, label="svm")
        assert result.shape == (N_SAMPLES, N_CLASSES)


# ===================================================================
# 8. ENSEMBLE OUTPUT  (Prediction → EnsembleOutput)
# ===================================================================

class TestEnsemble:

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

    def test_ensemble_single(self, debug):
        dbg_print(debug, "\n--- ensemble_single ---")
        pred = self._get_pred()
        ens = gp_ops.ensemble_single(pred)
        assert isinstance(ens, gp_types.EnsembleOutput)
        result = consume_prediction(ens, debug, label="ens_single")
        assert result.shape == (N_SAMPLES, N_CLASSES)

    def test_ensemble_sum_2(self, debug):
        dbg_print(debug, "\n--- ensemble_sum_2 ---")
        p1 = self._get_pred()
        p2 = self._get_pred()
        ens = gp_ops.ensemble_sum_2(p1, p2)
        assert isinstance(ens, gp_types.EnsembleOutput)
        result = consume_prediction(ens, debug, label="ens_sum_2")
        assert result.shape == (N_SAMPLES, N_CLASSES)
        dbg_print(debug, f"  max prob per sample (summed): {result.max(axis=1)[:5]}")

    def test_ensemble_sum_3(self, debug):
        dbg_print(debug, "\n--- ensemble_sum_3 ---")
        p1 = self._get_pred()
        p2 = self._get_pred()
        p3 = self._get_pred()
        ens = gp_ops.ensemble_sum_3(p1, p2, p3)
        assert isinstance(ens, gp_types.EnsembleOutput)
        result = consume_prediction(ens, debug, label="ens_sum_3")
        assert result.shape == (N_SAMPLES, N_CLASSES)


# ===================================================================
# 9. CASCADE CLASSIFIERS  (FeatureVector → FeatureVector)
# ===================================================================

class TestCascade:

    @pytest.fixture(autouse=True)
    def _setup_train(self, sample_labels):
        gp_context.context.reset(gp_context.ExecutionMode.TRAIN)
        gp_context.context.set_train_labels(sample_labels)
        yield
        gp_context.context.reset()

    def test_cascade_rf(self, debug):
        _refresh()
        dbg_print(debug, "\n--- cascade_rf ---")
        fv_in = gp_ops.histogram_features(gp_ops.GrayTerminal)
        fv_out = gp_ops.cascade_rf(fv_in, 50, 10)
        assert isinstance(fv_out, gp_types.FeatureVector)
        result = consume_features(fv_out, debug, label="cascade_rf")
        # cascade appends n_classes columns to the original features
        assert result.shape[0] == N_SAMPLES
        assert result.shape[1] == 256 + N_CLASSES
        dbg_print(debug, f"  cascade output dim = {result.shape[1]} (256 feats + {N_CLASSES} probs)")

    def test_cascade_lr(self, debug):
        _refresh()
        dbg_print(debug, "\n--- cascade_lr ---")
        fv_in = gp_ops.histogram_features(gp_ops.GrayTerminal)
        fv_out = gp_ops.cascade_lr(fv_in)
        result = consume_features(fv_out, debug, label="cascade_lr")
        assert result.shape == (N_SAMPLES, 256 + N_CLASSES)


# ===================================================================
# 10. PARAMETERIZED OPERATIONS
# ===================================================================

class TestParameterizedOps:

    def test_hog_filter_param(self, debug):
        _refresh()
        dbg_print(debug, "\n--- hog_filter_param ---")
        ch = gp_ops.hog_filter_param(gp_ops.GrayTerminal, 8, 2, 9)
        result = consume_channel(ch, debug, label="hog_filter_param")
        assert result.shape[0] == N_SAMPLES

    def test_lbp_filter_param(self, debug):
        _refresh()
        dbg_print(debug, "\n--- lbp_filter_param ---")
        ch = gp_ops.lbp_filter_param(gp_ops.GrayTerminal, 16, 2)
        result = consume_channel(ch, debug, label="lbp_filter_param")
        assert result.shape == (N_SAMPLES, IMG_H, IMG_W)

    def test_mean_filter_param(self, debug):
        _refresh()
        dbg_print(debug, "\n--- mean_filter_param ---")
        ch = gp_ops.mean_filter_param(gp_ops.GrayTerminal, 5)
        result = consume_channel(ch, debug, label="mean_filter_param")
        assert result.shape == (N_SAMPLES, IMG_H, IMG_W)

    def test_hog_features_param(self, debug):
        _refresh()
        dbg_print(debug, "\n--- hog_features_param ---")
        fv = gp_ops.hog_features_param(gp_ops.GrayTerminal, 8, 2, 9)
        result = consume_features(fv, debug, label="hog_features_param")
        assert result.shape[0] == N_SAMPLES
        assert result.shape[1] > 0

    def test_lbp_features_param(self, debug):
        _refresh()
        dbg_print(debug, "\n--- lbp_features_param ---")
        fv = gp_ops.lbp_features_param(gp_ops.GrayTerminal, 16, 2)
        result = consume_features(fv, debug, label="lbp_features_param")
        assert result.shape[0] == N_SAMPLES
        assert result.shape[1] == 18  # P + 2 = 16 + 2

    def test_canny_filter_param(self, debug):
        _refresh()
        dbg_print(debug, "\n--- canny_filter_param ---")
        ch = gp_ops.canny_filter_param(gp_ops.GrayTerminal, 50, 150)
        result = consume_channel(ch, debug, label="canny_filter_param")
        assert result.shape == (N_SAMPLES, IMG_H, IMG_W)

    def test_erosion_filter_param(self, debug):
        _refresh()
        dbg_print(debug, "\n--- erosion_filter_param ---")
        ch = gp_ops.erosion_filter_param(gp_ops.GrayTerminal, 5)
        result = consume_channel(ch, debug, label="erosion_filter_param")
        assert result.shape == (N_SAMPLES, IMG_H, IMG_W)

    def test_sift_features_param(self, debug):
        _refresh()
        dbg_print(debug, "\n--- sift_features_param ---")
        fv = gp_ops.sift_features_param(gp_ops.GrayTerminal, 100)
        result = consume_features(fv, debug, label="sift_features_param")
        assert result.shape == (N_SAMPLES, 128)


# ===================================================================
# 11. FILTER CHAINING  (filter → filter → feature)
# ===================================================================

class TestFilterChaining:
    """Verify that stacking filters still produces valid output."""

    def test_sobel_then_relu_then_histogram(self, debug):
        _refresh()
        dbg_print(debug, "\n--- sobel → relu → histogram ---")
        ch1 = gp_ops.sobel_filter(gp_ops.GrayTerminal)
        dbg_print(debug, "  after sobel:")
        if debug:
            consume_channel(ch1, True, label="sobel")
            _refresh()
            ch1 = gp_ops.sobel_filter(gp_ops.GrayTerminal)
        ch2 = gp_ops.relu(ch1)
        fv = gp_ops.histogram_features(ch2)
        result = consume_features(fv, debug, label="final histogram")
        assert result.shape == (N_SAMPLES, 256)

    def test_gaussian_then_lbp_features(self, debug):
        _refresh()
        dbg_print(debug, "\n--- gaussian → lbp_features ---")
        ch = gp_ops.gaussian_filter(gp_ops.GrayTerminal, 2)
        fv = gp_ops.lbp_features(ch)
        result = consume_features(fv, debug, label="gauss→lbp")
        assert result.shape[0] == N_SAMPLES
        assert result.shape[1] > 0

    def test_double_filter_preserves_shape(self, debug):
        _refresh()
        dbg_print(debug, "\n--- mean → median (double filter) ---")
        ch1 = gp_ops.mean_filter(gp_ops.GrayTerminal)
        ch2 = gp_ops.median_filter(ch1)
        result = consume_channel(ch2, debug, label="mean→median")
        assert result.shape == (N_SAMPLES, IMG_H, IMG_W)


# ===================================================================
# 12. END-TO-END MINI PIPELINE  (Channel → FV → Pred → Ensemble)
# ===================================================================

class TestMiniPipeline:
    """Build a small pipeline manually and run it end to end."""

    @pytest.fixture(autouse=True)
    def _setup_train(self, sample_labels):
        gp_context.context.reset(gp_context.ExecutionMode.TRAIN)
        gp_context.context.set_train_labels(sample_labels)
        yield
        gp_context.context.reset()

    def test_full_pipeline(self, debug):
        _refresh()
        dbg_print(debug, "\n=== Full Mini Pipeline ===")

        # Step 1: filter
        ch = gp_ops.sobel_filter(gp_ops.GrayTerminal)
        dbg_print(debug, "  [1] sobel_filter applied")

        # Step 2: extract features
        fv = gp_ops.histogram_features(ch)
        dbg_print(debug, "  [2] histogram_features extracted")

        # Step 3: classify
        pred = gp_ops.rf_classification(fv, 100, 20)
        dbg_print(debug, "  [3] rf_classification")

        # Step 4: ensemble
        ens = gp_ops.ensemble_single(pred)
        dbg_print(debug, "  [4] ensemble_single")

        result = consume_prediction(ens, debug, label="pipeline output")
        assert result.shape == (N_SAMPLES, N_CLASSES)
        preds_cls = np.argmax(result, axis=1)
        dbg_print(debug, f"  predicted classes: {preds_cls}")
        assert all(0 <= p < N_CLASSES for p in preds_cls)
