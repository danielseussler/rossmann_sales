import numpy as np

def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Square Percentage Error (RMSPE) between two NumPy arrays."""

    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape.")

    percentage_error = (y_true - y_pred) / y_true
    percentage_error[y_true == 0] = 0

    return np.sqrt(np.mean(percentage_error**2))


def mad(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the Mean Absolute Deviation (MAD) between two numpy arrays.

    Parameters:
    - x, y: numpy arrays of the same length.

    Returns:
    - mad_value: Mean Absolute Deviation between x and y.
    """
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape.")

    return float(np.median(np.abs(np.subtract(x, y))))