from __future__ import annotations

import numpy as np


def apply_quantized_layer(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.matmul(inputs, weights)

