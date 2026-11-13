"""Shared validation helpers and small utilities for numcompute_stream.

Every public estimator validates its inputs through these helpers, so error
messages and shape rules are the same across the package:

- ``X`` is a 2-D float array of shape (n_samples, n_features).
- ``y`` is a 1-D array of shape (n_samples,).
- Missing values are written as NaN.
"""

from collections import deque

import numpy as np

__all__ = ["NotFittedError", "check_array", "check_X_y", "check_y", "check_positive_int",
           "memory_usage", "rng_from"]


class NotFittedError(ValueError, AttributeError):
    """Raised when a method needs a fitted estimator but none is available."""


def check_array(X, name="X", allow_nan=True, allow_empty=True):
    """Validate a 2-D numeric array and return it as float64.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Input data. A 1-D input is rejected because it is ambiguous
        (one sample or one feature?).
    name : str, default "X"
        Name used in error messages.
    allow_nan : bool, default True
        If False, raise when X contains NaN.
    allow_empty : bool, default True
        If False, raise when X has zero rows.

    Returns
    -------
    X : ndarray of float64, shape (n_samples, n_features)

    Raises
    ------
    TypeError
        If X is not numeric.
    ValueError
        If X is not 2-D, has no features, contains +-inf, or violates
        ``allow_nan`` / ``allow_empty``.

    Complexity
    ----------
    Time O(n_samples * n_features), space O(n_samples * n_features) if a
    copy is needed.
    """
    arr = np.asarray(X)
    if arr.dtype == object or not (np.issubdtype(arr.dtype, np.number)
                                   or arr.dtype == bool):
        raise TypeError(f"{name} must be numeric, got dtype {arr.dtype}")
    if arr.ndim != 2:
        hint = " Use X.reshape(-1, 1) for one feature or X.reshape(1, -1) for one sample." \
            if arr.ndim == 1 else ""
        raise ValueError(f"{name} must be 2-D (n_samples, n_features), got shape {arr.shape}.{hint}")
    if arr.shape[1] == 0:
        raise ValueError(f"{name} must have at least one feature, got shape {arr.shape}")
    if not allow_empty and arr.shape[0] == 0:
        raise ValueError(f"{name} is empty (0 samples)")
    arr = arr.astype(np.float64, copy=False)
    if np.isinf(arr).any():
        raise ValueError(f"{name} contains infinite values")
    if not allow_nan and np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN; impute missing values first")
    return arr


def check_y(y, n_samples=None, name="y"):
    """Validate a 1-D label array.

    Parameters
    ----------
    y : array_like, shape (n_samples,)
        Labels (numbers, strings or booleans). Missing labels are not allowed.
    n_samples : int or None
        Expected length, usually ``X.shape[0]``.
    name : str, default "y"

    Returns
    -------
    y : ndarray, shape (n_samples,)

    Raises
    ------
    ValueError
        If y is not 1-D, has the wrong length, or contains NaN / None.

    Complexity
    ----------
    Time O(n_samples), space O(1) extra.
    """
    arr = np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D (n_samples,), got shape {arr.shape}")
    if n_samples is not None and arr.shape[0] != n_samples:
        raise ValueError(f"{name} has {arr.shape[0]} samples but X has {n_samples}")
    if arr.dtype == object and any(v is None for v in arr):
        raise ValueError(f"{name} contains missing labels (None)")
    if np.issubdtype(arr.dtype, np.floating) and np.isnan(arr).any():
        raise ValueError(f"{name} contains missing labels (NaN)")
    return arr


def check_X_y(X, y, allow_nan=True):
    """Validate a feature matrix and its labels together.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
    y : array_like, shape (n_samples,)
    allow_nan : bool, default True

    Returns
    -------
    X : ndarray of float64, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)

    Raises
    ------
    TypeError, ValueError
        See :func:`check_array` and :func:`check_y`.

    Complexity
    ----------
    Time O(n_samples * n_features).
    """
    X = check_array(X, allow_nan=allow_nan)
    return X, check_y(y, X.shape[0])


def check_positive_int(value, name, allow_none=False):
    """Validate a positive integer hyperparameter.

    Parameters
    ----------
    value : object
    name : str
        Parameter name used in the error message.
    allow_none : bool, default False

    Returns
    -------
    value : int or None

    Raises
    ------
    ValueError
        If ``value`` is not a positive int (bool and float are rejected).

    Complexity
    ----------
    O(1).
    """
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        expected = "None or a positive int" if allow_none else "a positive int"
        raise ValueError(f"{name} must be {expected}, got {value!r}")
    return int(value)


def rng_from(random_state):
    """Return a ``numpy.random.Generator`` from a seed, generator or None.

    Parameters
    ----------
    random_state : int, Generator or None

    Returns
    -------
    rng : numpy.random.Generator

    Complexity
    ----------
    O(1).
    """
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def memory_usage(obj, _seen=None):
    """Estimate the memory held by an object's NumPy arrays, in bytes.

    Walks the object's attributes, lists, tuples, dicts and deques and sums
    ``ndarray.nbytes``. It is used to log the memory footprint of streaming
    models (for example the size of a tree's data buffer).

    Parameters
    ----------
    obj : object

    Returns
    -------
    n_bytes : int

    Complexity
    ----------
    Time O(number of reachable objects), space O(number of reachable objects).
    """
    if _seen is None:
        _seen = set()
    if id(obj) in _seen:
        return 0
    _seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return sum(memory_usage(v, _seen) for v in obj.values())
    if isinstance(obj, (list, tuple, deque, set)):
        return sum(memory_usage(v, _seen) for v in obj)
    if hasattr(obj, "__dict__"):
        return memory_usage(vars(obj), _seen)
    return 0
