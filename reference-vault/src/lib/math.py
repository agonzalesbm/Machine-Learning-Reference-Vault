from numpy.typing import NDArray
import numpy as np


def sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    if not isinstance(z, np.ndarray):
        raise TypeError("Expected numpy.ndarray")
    if z.dtype != np.float64:
        raise TypeError("Expected dtype float64")

    return 1 / (1 + np.exp(-z))
