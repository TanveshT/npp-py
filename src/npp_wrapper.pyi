from __future__ import annotations

from typing import Literal

import numpy as np

def resize_32f(
    img: np.ndarray,
    out_h: int,
    out_w: int,
    inter: Literal["nearest", "bilinear", "cubic"] = "bilinear",
    layout: Literal["auto", "hwc", "chw"] = "auto",
) -> np.ndarray: ...

def resize_sqr_pixel_32f(
    img: np.ndarray,
    out_h: int,
    out_w: int,
    scale_y: float | None = None,
    scale_x: float | None = None,
    shift_y: float = 0.0,
    shift_x: float = 0.0,
    inter: Literal["nearest", "bilinear", "cubic"] = "bilinear",
    layout: Literal["auto", "hwc", "chw"] = "auto",
) -> np.ndarray: ...
