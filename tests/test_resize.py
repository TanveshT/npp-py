import numpy as np
import pytest

try:
    import npp_wrapper
except ImportError:  # pragma: no cover - exercised via skip
    npp_wrapper = None


@pytest.fixture(scope="module")
def resize():
    """Provide the resize function or skip if the extension is unavailable."""
    if npp_wrapper is None:
        pytest.skip("npp_wrapper extension is not built")
    return npp_wrapper.resize_32f


def test_invalid_output_dimensions(resize):
    img = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        resize(img, 0, 2)


def test_invalid_input_rank(resize):
    img = np.zeros((1, 4, 4, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="img must be 2D"):
        resize(img, 2, 2)


def test_resize_changes_shape(resize):
    img = np.arange(16, dtype=np.float32).reshape(4, 4)
    try:
        out = resize(img, 2, 8, "linear")
    except RuntimeError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"CUDA/NPP runtime unavailable: {exc}")
    assert out.shape == (2, 8)
    assert out.dtype == np.float32


def test_resize_unknown_interpolation(resize):
    img = np.arange(9, dtype=np.float32).reshape(3, 3)
    try:
        out = resize(img, 6, 6, "bogus")
    except RuntimeError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"CUDA/NPP runtime unavailable: {exc}")
    assert out.shape == (6, 6)
    assert out.dtype == np.float32


def test_resize_hwc(resize):
    img = np.arange(3 * 5 * 5, dtype=np.float32).reshape(5, 5, 3)
    try:
        out = resize(img, 2, 4, inter="nearest")
    except RuntimeError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"CUDA/NPP runtime unavailable: {exc}")
    assert out.shape == (2, 4, 3)
    assert out.dtype == np.float32


def test_resize_chw(resize):
    img = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    try:
        out = resize(img, 2, 6, inter="nearest", layout="chw")
    except RuntimeError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"CUDA/NPP runtime unavailable: {exc}")
    assert out.shape == (3, 2, 6)
    assert out.dtype == np.float32
