# npp-py

Thin Python wrapper around NVIDIA's NPP routines using CUDA + pybind11. The project currently focuses on resize helpers (e.g., `resize_32f`) and is structured so additional NPP-backed utilities can be layered on later.

## Prerequisites

- CUDA Toolkit **12.x** installed locally with the `libnppig`, `libnppisu`, and `libnppc` components. `/usr/local/cuda-*/lib64` must be on your runtime library path.
- A compatible NVIDIA GPU with the proper driver installed.
- Python **3.9+** with `pip`.
- A C++17-capable compiler (gcc/clang) and `cmake >= 3.22`.

Optional (but recommended):

- Create and activate a Python virtual environment so the build installs into an isolated prefix.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## Installation

1. Install build dependencies:

   ```bash
   pip install -r <(printf 'scikit-build-core>=0.8.0\npybind11>=2.11.0\nnumpy>=1.20\n')
   ```

   (If your shell does not support process substitution, place those lines into a temporary `requirements.txt` file instead.)

2. Build and install the extension:

   ```bash
   pip install . --no-cache-dir --no-build-isolation
   ```

   You should see a wheel built for your platform and installed into the virtual environment.

3. Verify the module loads:

   ```bash
   python -c "import npp_wrapper; print(npp_wrapper.resize_32f)"
   ```

   If you encounter an `ImportError` mentioning an undefined symbol like `nppiResize_32f_C1R`, ensure `libnppig.so` (from the CUDA toolkit) is reachable via `LD_LIBRARY_PATH`.

## Usage

### Resize (current export)

Below is a minimal example for the resize helper:

```python
import numpy as np
import npp_wrapper

img = np.random.rand(1080, 1920, 3).astype("float32")
resized = npp_wrapper.resize_32f(img, 640, 640, inter="linear", layout="hwc")
```

- `inter` accepts `"nearest"`, `"linear"`, or `"cubic"`.
- `layout` decides how channel dimensions are interpreted (`"auto"`, `"hwc"`, `"chw"`).

### Extending the Module

As you add more NPP-backed helpers, follow the same pattern:

1. Extend the C++ module in `src/npp_wrapper.cpp`.
2. Update `src/npp_wrapper.pyi` with type hints for IntelliSense.
3. Document usage here—consider dedicating a short subsection per function (e.g., “Color Conversion”, “Filtering”) with its signature, expected numpy dtypes/layouts, and sample code.

## Testing

Install dev extras and run pytest:

```bash
pip install .[dev]
python -m pytest
```

Tests will skip gracefully if the extension fails to import or CUDA/NPP is unavailable. To exercise them fully, run on a machine with CUDA 12.x and the NPP libraries installed.

## Troubleshooting

- **Undefined NPP symbols**: Reinstall after confirming `/usr/local/cuda/lib64` (or your CUDA path) is in `LD_LIBRARY_PATH`. Ensure `libnppig` is present; earlier builds used `libnppif`.
- **Build fails looking for scikit-build-core**: Pre-install `scikit-build-core` inside your virtualenv (`pip install scikit-build-core pybind11`).
- **Tests skip**: Check that the compiled `.so` in your environment matches the current source (`readelf -d $(python -c 'import npp_wrapper, pathlib; print(pathlib.Path(npp_wrapper.__file__))') | grep libnppig`).
