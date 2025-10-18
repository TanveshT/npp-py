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
