"""Fingerprint-matching baselines for the neural-network regressor.

Fingerprinting predates learned regressors: the classic approach stores a
radio map of fingerprints recorded at known reference points and locates a
new measurement by its nearest neighbors in signal space (Bahl &
Padmanabhan, 2000). Two forms are available as models:

* ``knn`` / ``weighted_knn`` -- the :math:`k` nearest *packets* of the radio
  map, their positions averaged uniformly or weighted by inverse signal-space
  distance (scikit-learn's :class:`~sklearn.neighbors.KNeighborsRegressor`,
  configured in :mod:`~wifi_fingerprint_locator.models`).
* ``nearest_reference`` -- :class:`NearestReferencePoint`: every reference
  point is summarized by its mean fingerprint, and a measurement is assigned
  the position of the closest mean. It can only answer with a surveyed
  position, so it is a classifier of reference points in regression clothing.

A learned regressor is only worth its complexity if it beats these.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class NearestReferencePoint(RegressorMixin, BaseEstimator):
    """Assign the position of the reference point with the closest mean fingerprint.

    Parameters
    ----------
    decimals : int, optional
        Positions are rounded to this many decimals to decide which training
        items share a reference point.

    Attributes
    ----------
    references_ : numpy.ndarray, shape (n_references, n_outputs)
        Distinct training positions.
    centroids_ : numpy.ndarray, shape (n_references, n_features)
        Mean fingerprint of each reference point.

    References
    ----------
    .. [1] P. Bahl and V. N. Padmanabhan, "RADAR: An in-building RF-based user
       location and tracking system," in Proc. IEEE INFOCOM, 2000,
       pp. 775-784.

    Examples
    --------
    >>> X = np.array([[0.0, 0.0], [0.2, 0.0], [5.0, 5.0], [5.2, 5.0]])
    >>> y = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 1.0], [3.0, 1.0]])
    >>> model = NearestReferencePoint().fit(X, y)
    >>> model.predict([[4.0, 4.5], [0.5, 0.1]]).tolist()
    [[3.0, 1.0], [0.0, 0.0]]
    """

    def __init__(self, decimals: int = 6) -> None:
        self.decimals = decimals

    def fit(self, X: ArrayLike, y: ArrayLike) -> NearestReferencePoint:
        """Compute the mean fingerprint of every reference point.

        Parameters
        ----------
        X : array_like, shape (n_items, n_features)
            Fingerprints.
        y : array_like, shape (n_items, n_outputs)
            Positions.

        Returns
        -------
        NearestReferencePoint
            The fitted estimator.
        """
        inputs = np.asarray(X, dtype=np.float64)
        targets = np.asarray(y, dtype=np.float64)
        self.target_ndim_ = targets.ndim
        targets = targets.reshape(inputs.shape[0], -1)
        references, inverse = np.unique(
            np.round(targets, self.decimals), axis=0, return_inverse=True
        )
        inverse = inverse.ravel()
        counts = np.bincount(inverse, minlength=references.shape[0]).astype(np.float64)
        sums = np.zeros((references.shape[0], inputs.shape[1]))
        np.add.at(sums, inverse, inputs)
        self.references_ = references
        self.centroids_ = sums / counts[:, None]
        self.n_features_in_ = inputs.shape[1]
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Position of the nearest mean fingerprint for every row of ``X``."""
        check_is_fitted(self, "centroids_")
        inputs = np.asarray(X, dtype=np.float64)
        if inputs.ndim != 2 or inputs.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must be shaped (n_items, {self.n_features_in_}), got {inputs.shape}"
            )
        distances = (
            np.sum(inputs**2, axis=1)[:, None]
            - 2.0 * inputs @ self.centroids_.T
            + np.sum(self.centroids_**2, axis=1)[None, :]
        )
        estimate = self.references_[np.argmin(distances, axis=1)]
        return estimate.ravel() if self.target_ndim_ == 1 else estimate
