"""
Shared fixtures, helpers, and the DEBUG flag for CVTrees unit tests.

Usage:
    # Run all tests silently (CI mode):
    pytest unit_tests/

    # Run with verbose debug prints:
    pytest unit_tests/ --debug-ops

    # Run a single test file with debug:
    pytest unit_tests/test_single_nodes.py --debug-ops -s
"""
import sys
import os
import math
import random
import warnings

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make the project importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import gp_ops, gp_types, gp_context, gp_setup, gp_utils

# Suppress noisy sklearn / skimage warnings during tests
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Custom pytest option: --debug-ops
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--debug-ops",
        action="store_true",
        default=False,
        help="Enable verbose debug printing inside tests",
    )


@pytest.fixture
def debug(request):
    """
    Returns True when the user passes --debug-ops on the command line.
    Tests can use this to gate print statements.
    """
    return True
    return request.config.getoption("--debug-ops")


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def seed_everything():
    """Fix random seeds so tests are reproducible."""
    random.seed(42)
    np.random.seed(42)


# ---------------------------------------------------------------------------
# Image / data fixtures
# ---------------------------------------------------------------------------

IMG_H, IMG_W = 32, 32
N_SAMPLES = 20
N_CLASSES = 3
BATCH_SIZE = 10


def _make_random_images(n=N_SAMPLES, h=IMG_H, w=IMG_W):
    """Return (N, 3, H, W) float32 images in [0, 1]."""
    return np.random.rand(n, 3, h, w).astype(np.float32)


def _make_labels(n=N_SAMPLES, n_classes=N_CLASSES):
    """Return balanced integer labels."""
    return np.array([i % n_classes for i in range(n)])


@pytest.fixture
def sample_images():
    """(N, 3, H, W) float32 images."""
    return _make_random_images()


@pytest.fixture
def sample_labels():
    """Integer labels array."""
    return _make_labels()


def make_image_iterator(X, batch_size=BATCH_SIZE):
    """
    Given X of shape (N, 3, H, W) return an iterator factory that yields
    Batch objects with channel-list data (matching gp_ops expectations).
    """
    def iterator():
        for i in range(0, len(X), batch_size):
            chunk = X[i:i + batch_size]
            r = chunk[:, 0, :, :]
            g = chunk[:, 1, :, :]
            b = chunk[:, 2, :, :]
            yield gp_types.Batch([r, g, b])
    return iterator


@pytest.fixture
def setup_image(sample_images):
    """
    Sets up gp_ops._CURRENT_IMAGE from sample_images so that
    Channel terminals (GetRed, GetGreen, …) work.
    Returns the images array for further inspection.
    """
    img = gp_types.Image(make_image_iterator(sample_images))
    gp_ops.set_image(img)
    return sample_images


@pytest.fixture
def gray_channel(setup_image):
    """A Channel wrapping the gray terminal (lazy – not yet consumed)."""
    # We need to re-set the image each time we want to iterate because
    # terminals reuse the global _CURRENT_IMAGE.
    return gp_ops.GrayTerminal


@pytest.fixture
def red_channel(setup_image):
    return gp_ops.RedTerminal


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def train_context(sample_labels):
    """Reset context to TRAIN mode and set labels."""
    gp_context.context.reset(gp_context.ExecutionMode.TRAIN)
    gp_context.context.set_train_labels(sample_labels)
    yield
    gp_context.context.reset()


# ---------------------------------------------------------------------------
# Primitive set fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pset_basic():
    """Basic primitive set (no extended ops/params)."""
    return gp_setup.create_primitive_set(extended_ops=False, extended_params=False)


@pytest.fixture
def pset_extended():
    """Extended primitive set (extended ops + extended params)."""
    return gp_setup.create_primitive_set(extended_ops=True, extended_params=True)


# ---------------------------------------------------------------------------
# Debug-printing helpers (used inside tests)
# ---------------------------------------------------------------------------

def dbg_print(debug_flag, *args, **kwargs):
    """Only print if debug_flag is True."""
    if debug_flag:
        print(*args, **kwargs)


def consume_channel(channel, debug_flag=False, label="Channel"):
    """
    Consume a Channel, return the stacked (N, H, W) array.
    Optionally print shape / stats when debug is on.
    Also collects per-batch dtype info for consistency checking.
    """
    batches = []
    batch_dtypes = []
    for batch in channel:
        batches.append(batch.data)
        batch_dtypes.append(batch.data.dtype)
    result = np.vstack(batches) if batches else np.array([])
    if debug_flag:
        unique_dtypes = set(str(d) for d in batch_dtypes)
        print(f"  [{label}] shape={result.shape}, dtype={result.dtype}, "
              f"batch_dtypes={unique_dtypes}, "
              f"min={result.min():.4f}, max={result.max():.4f}, mean={result.mean():.4f}")
    return result


def consume_features(fv, debug_flag=False, label="FeatureVector"):
    """
    Consume a FeatureVector, return the stacked (N, D) array.
    Also collects per-batch dtype info for consistency checking.
    """
    batches = []
    batch_dtypes = []
    for batch in fv:
        batches.append(batch.data)
        batch_dtypes.append(batch.data.dtype)
    result = np.vstack(batches) if batches else np.array([])
    if debug_flag:
        unique_dtypes = set(str(d) for d in batch_dtypes)
        print(f"  [{label}] shape={result.shape}, dtype={result.dtype}, "
              f"batch_dtypes={unique_dtypes}, "
              f"min={result.min():.4f}, max={result.max():.4f}")
    return result


def consume_prediction(pred, debug_flag=False, label="Prediction"):
    """
    Consume a Prediction / EnsembleOutput, return the stacked (N, C) array.
    Also collects per-batch dtype info for consistency checking.
    """
    batches = []
    batch_dtypes = []
    for batch in pred:
        batches.append(batch.data)
        batch_dtypes.append(batch.data.dtype)
    result = np.vstack(batches) if batches else np.array([])
    if debug_flag:
        unique_dtypes = set(str(d) for d in batch_dtypes)
        print(f"  [{label}] shape={result.shape}, dtype={result.dtype}, "
              f"batch_dtypes={unique_dtypes}")
        if result.ndim == 2:
            print(f"    predicted classes: {np.argmax(result, axis=1)[:10]}...")
    return result


def fresh_image_setup(X):
    """
    Re-initialise _CURRENT_IMAGE from X (N,3,H,W).  Must be called before
    each independent pipeline evaluation because terminals are consumed once.
    """
    img = gp_types.Image(make_image_iterator(X))
    gp_ops.set_image(img)
