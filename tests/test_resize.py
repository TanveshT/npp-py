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


@pytest.fixture(scope="module")
def resize_sqr_pixel():
    if npp_wrapper is None:
        pytest.skip("npp_wrapper extension is not built")
    return npp_wrapper.resize_sqr_pixel_32f


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
        out = resize(img, 2, 8, "bilinear")
    except RuntimeError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"CUDA/NPP runtime unavailable: {exc}")
    assert out.shape == (2, 8)
    assert out.dtype == np.float32


def test_resize_unknown_interpolation(resize):
    img = np.arange(9, dtype=np.float32).reshape(3, 3)
    with pytest.raises(ValueError):
        resize(img, 6, 6, "bogus")


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


def test_resize_sqr_pixel_default(resize_sqr_pixel):
    img = np.arange(16, dtype=np.float32).reshape(4, 4)
    try:
        out = resize_sqr_pixel(img, 8, 8)
    except RuntimeError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"CUDA/NPP runtime unavailable: {exc}")
    assert out.shape == (8, 8)
    assert out.dtype == np.float32


def test_resize_sqr_pixel_hwc_params(resize_sqr_pixel):
    img = np.arange(3 * 5 * 7, dtype=np.float32).reshape(5, 7, 3)
    out_h, out_w = 10, 14
    scale_y = out_h / img.shape[0]
    scale_x = out_w / img.shape[1]
    try:
        out = resize_sqr_pixel(
            img,
            out_h,
            out_w,
            scale_y=scale_y,
            scale_x=scale_x,
            shift_y=0.5,
            shift_x=-0.5,
            inter="bilinear",
            layout="hwc",
        )
    except RuntimeError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"CUDA/NPP runtime unavailable: {exc}")
    assert out.shape == (out_h, out_w, 3)
    assert out.dtype == np.float32
