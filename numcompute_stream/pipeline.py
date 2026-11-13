"""Chained transformers + model with streaming support.

Protocol
--------
- Transformer: ``partial_fit(X[, y])``, ``fit(X[, y])``, ``transform(X)``
- Model (final step): ``partial_fit(X, y)``, ``fit(X, y)``, ``predict(X)``

``y`` is passed to a step only if its method accepts a ``y`` argument, so
transformers written as ``partial_fit(self, X)`` work as well as
``partial_fit(self, X, y=None)``.

Example
-------
>>> pipe = Pipeline([
...     ("scale", StandardScaler()),
...     ("model", DecisionTreeClassifier(max_depth=5, buffer_size=500)),
... ])
>>> for X_chunk, y_chunk in stream:
...     pipe.partial_fit(X_chunk, y_chunk)
>>> pipe.predict(X_new)
"""

import inspect

import numpy as np

from .utils import check_y

__all__ = ["Pipeline"]


def _call(est, method, X, y):
    """Call ``est.method(X, y)`` if the method takes a second argument, else ``est.method(X)``."""
    fn = getattr(est, method)
    return fn(X, y) if len(inspect.signature(fn).parameters) >= 2 else fn(X)


class Pipeline:
    """Apply transformers in order, then a final model.

    Parameters
    ----------
    steps : list of (name, estimator)
        All steps but the last must implement ``partial_fit`` and
        ``transform``. The last step must implement ``partial_fit``; it may be
        a model (``predict``) or another transformer (``transform``).

    Attributes
    ----------
    named_steps : dict
        Name -> estimator.
    final_estimator : object
        The last step.

    Raises
    ------
    ValueError
        If ``steps`` is empty or names are not unique.
    TypeError
        If a step does not implement the required methods.

    Notes
    -----
    ``partial_fit`` updates each transformer with the chunk *before*
    transforming it, so every chunk is transformed with statistics that
    include that chunk. Because a buffered tree retrains on chunks that were
    transformed at different times, prefer transformers whose output is
    stable over time (imputation, scaling) before a tree; trees split on
    thresholds and are not affected by monotone rescaling within a buffer.
    """

    def __init__(self, steps):
        if not steps:
            raise ValueError("steps must contain at least one (name, estimator) pair")
        names = [name for name, _ in steps]
        if len(set(names)) != len(names):
            raise ValueError(f"Step names must be unique, got {names}")
        for name, est in steps[:-1]:
            for method in ("partial_fit", "transform"):
                if not callable(getattr(est, method, None)):
                    raise TypeError(f"Step {name!r} must implement {method}()")
        final_name, final = steps[-1]
        if not callable(getattr(final, "partial_fit", None)):
            raise TypeError(f"Final step {final_name!r} must implement partial_fit()")
        self.steps = list(steps)

    @property
    def named_steps(self):
        """Mapping from step name to estimator (dict)."""
        return dict(self.steps)

    @property
    def final_estimator(self):
        """The last step of the pipeline."""
        return self.steps[-1][1]

    def _transformers(self):
        return [est for _, est in self.steps[:-1]]

    def _transform_only(self, X):
        for est in self._transformers():
            X = est.transform(X)
        return X

    def fit(self, X, y=None):
        """Fit every step from scratch on (X, y).

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
        y : array_like, shape (n_samples,) or None

        Returns
        -------
        self

        Complexity
        ----------
        Sum of the steps' ``fit`` and ``transform`` costs.
        """
        for est in self._transformers():
            X = _call(est, "fit", X, y).transform(X)
        _call(self.final_estimator, "fit", X, y)
        return self

    def partial_fit(self, X_chunk, y_chunk=None):
        """Update every step with one chunk.

        Each transformer is updated with the chunk, then transforms it for the
        next step; the final step receives the transformed chunk.

        Parameters
        ----------
        X_chunk : array_like, shape (n_samples, n_features)
        y_chunk : array_like, shape (n_samples,) or None

        Returns
        -------
        self

        Complexity
        ----------
        Sum of the steps' ``partial_fit`` and ``transform`` costs on the chunk.
        """
        X = X_chunk
        for est in self._transformers():
            X = _call(est, "partial_fit", X, y_chunk).transform(X)
        _call(self.final_estimator, "partial_fit", X, y_chunk)
        return self

    def transform(self, X):
        """Transform X through every step (the final step must be a transformer).

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        X_out : ndarray, shape (n_samples, n_output_features)

        Raises
        ------
        TypeError
            If the final step has no ``transform``.

        Complexity
        ----------
        Sum of the steps' ``transform`` costs.
        """
        if not callable(getattr(self.final_estimator, "transform", None)):
            raise TypeError("The final step has no transform(); use predict()")
        return self.final_estimator.transform(self._transform_only(X))

    def predict(self, X):
        """Transform X, then predict with the final model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)

        Raises
        ------
        NotFittedError
            If the model has not been trained yet.

        Complexity
        ----------
        Transform costs plus the model's ``predict`` cost.
        """
        return self.final_estimator.predict(self._transform_only(X))

    def predict_proba(self, X):
        """Transform X, then return the model's class probabilities.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray, shape (n_samples, n_classes)

        Complexity
        ----------
        Transform costs plus the model's ``predict_proba`` cost.
        """
        return self.final_estimator.predict_proba(self._transform_only(X))

    def score(self, X, y):
        """Accuracy of :meth:`predict` on (X, y).

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
        y : array_like, shape (n_samples,)

        Returns
        -------
        accuracy : float
            NaN for empty input.

        Raises
        ------
        ValueError
            If X and y have different lengths.

        Complexity
        ----------
        Cost of :meth:`predict` plus O(n).
        """
        y = check_y(y, np.asarray(X).shape[0])
        return float(np.mean(self.predict(X) == y)) if y.size else float("nan")

    def __repr__(self):
        inner = ", ".join(f"({n!r}, {type(e).__name__})" for n, e in self.steps)
        return f"Pipeline([{inner}])"
