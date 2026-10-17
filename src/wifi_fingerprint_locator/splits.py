"""How a fingerprint survey is divided into training and test data.

A survey is recorded location by location, and consecutive packets of one
location are about a second apart and nearly identical. How the packets are
divided decides what an error figure means:

``packet_random``
    Packets are shuffled and divided regardless of where they were recorded.
    Every test location is also a training location, and the neighbors of a
    test packet in time are in the training set: this measures *recognition*
    of known locations and is optimistic.
``location_block``
    Whole reference points are kept together and the locations are divided in
    the order they were recorded, so each test fold is a contiguous region of
    the survey (often one edge of the grid): spatial *extrapolation*.
``location_holdout``
    Whole reference points are kept together and assigned to folds at random
    (a shuffled group k-fold): the model must *interpolate* to locations it
    has never seen. No location appears in both training and test data.
``temporal``
    Every location is in every fold, but each location's packets are cut into
    contiguous blocks in recording order, and each fold tests on one block of
    every location: robustness to the channel drifting over time.

:func:`fold_indices` gives k-fold cross-validation for all four;
:func:`holdout_indices` gives a single training / validation / test division
with chosen fractions. Both are reproducible from ``rng``. Items are assumed to
be ordered as ``prepare`` writes them: capture file by capture file, packets in
recording order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .errors import ConfigError

#: Strategies accepted by :func:`fold_indices`, :func:`holdout_indices` and
#: ``evaluation.split_strategy``.
SPLIT_STRATEGIES: tuple[str, ...] = (
    "packet_random",
    "location_block",
    "location_holdout",
    "temporal",
)
#: Strategies whose test folds are whole reference points never seen in training.
LOCATION_STRATEGIES: tuple[str, ...] = ("location_block", "location_holdout")


@dataclass(frozen=True, eq=False)
class SplitIndices:
    """A single training / validation / test division.

    Attributes
    ----------
    train, validation, test : numpy.ndarray of int64
        Sorted item indices; together they cover every item exactly once.

    Examples
    --------
    >>> split = SplitIndices(np.array([0, 1]), np.array([2]), np.array([3]))
    >>> split.sizes()
    {'train': 2, 'validation': 1, 'test': 1}
    """

    train: NDArray[np.int64]
    validation: NDArray[np.int64]
    test: NDArray[np.int64]

    def sizes(self) -> dict[str, int]:
        """Number of items in each part."""
        return {
            "train": int(self.train.size),
            "validation": int(self.validation.size),
            "test": int(self.test.size),
        }


def group_codes(groups: ArrayLike) -> NDArray[np.int64]:
    """Integer code of every item's group, numbered in order of first appearance.

    Examples
    --------
    >>> group_codes(["b", "b", "a", "c", "a"]).tolist()
    [0, 0, 1, 2, 1]
    """
    labels = np.asarray(groups).astype(str)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError(
            f"groups must be a non-empty one-dimensional array, got shape {labels.shape}"
        )
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    rank = np.empty(first.size, dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(first.size)
    return rank[inverse.ravel()]


def _rank_within_group(codes: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Position of each item within its group (in item order) and each group's size."""
    order = np.argsort(codes, kind="stable")
    sizes = np.bincount(codes)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    rank = np.empty(codes.size, dtype=np.int64)
    rank[order] = np.arange(codes.size) - np.repeat(starts, sizes)
    return rank, sizes


def _check_strategy(strategy: str) -> None:
    if strategy not in SPLIT_STRATEGIES:
        raise ConfigError(
            f"evaluation.split_strategy={strategy!r} is not supported; choose one of "
            f"{list(SPLIT_STRATEGIES)}"
        )


def _complement(n_items: int, test: NDArray[np.int64]) -> NDArray[np.int64]:
    mask = np.ones(n_items, dtype=bool)
    mask[test] = False
    return np.flatnonzero(mask).astype(np.int64)


def fold_indices(
    strategy: str,
    groups: ArrayLike,
    *,
    n_splits: int,
    rng: np.random.Generator | int | None = None,
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """k-fold cross-validation indices for one of :data:`SPLIT_STRATEGIES`.

    Parameters
    ----------
    strategy : str
        ``packet_random``, ``location_block``, ``location_holdout`` or ``temporal``.
    groups : array_like of str, shape (n_items,)
        Reference point of every item, items in recording order.
    n_splits : int
        Number of folds.
    rng : numpy.random.Generator or int, optional
        Randomness for the shuffled strategies.

    Returns
    -------
    list of (train, test)
        Sorted index arrays; every item is tested exactly once.

    Raises
    ------
    ConfigError
        For an unknown strategy or more folds than the data allow.

    Examples
    --------
    >>> groups = np.repeat(["p0", "p1", "p2", "p3"], 3)
    >>> folds = fold_indices("location_block", groups, n_splits=2)
    >>> [test.tolist() for _, test in folds]
    [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]
    >>> [test.tolist() for _, test in fold_indices("temporal", groups, n_splits=3)][0]
    [0, 3, 6, 9]
    """
    _check_strategy(strategy)
    rng = np.random.default_rng(rng)
    codes = group_codes(groups)
    n_items = codes.size
    n_groups = int(codes.max()) + 1
    if n_splits < 2:
        raise ConfigError(f"evaluation.n_splits must be at least 2, got {n_splits!r}")

    if strategy == "packet_random":
        if n_splits > n_items:
            raise ConfigError(
                f"evaluation.n_splits={n_splits} exceeds the {n_items} packets available"
            )
        chunks = np.array_split(rng.permutation(n_items), n_splits)
        tests = [np.sort(chunk).astype(np.int64) for chunk in chunks]
    elif strategy in LOCATION_STRATEGIES:
        if n_splits > n_groups:
            raise ConfigError(
                f"evaluation.n_splits={n_splits} exceeds the {n_groups} reference locations "
                f"available to the {strategy!r} split"
            )
        order = np.arange(n_groups) if strategy == "location_block" else rng.permutation(n_groups)
        tests = [
            np.flatnonzero(np.isin(codes, chunk)).astype(np.int64)
            for chunk in np.array_split(order, n_splits)
        ]
    else:  # temporal
        rank, sizes = _rank_within_group(codes)
        if n_splits > int(sizes.min()):
            raise ConfigError(
                f"evaluation.n_splits={n_splits} exceeds the {int(sizes.min())} packets of the "
                "smallest location, which the 'temporal' split divides into blocks"
            )
        block = (rank * n_splits) // sizes[codes]
        tests = [np.flatnonzero(block == fold).astype(np.int64) for fold in range(n_splits)]
    return [(_complement(n_items, test), test) for test in tests]


def _unit_counts(n_units: int, fractions: Sequence[float]) -> tuple[int, int, int]:
    n_train = round(fractions[0] * n_units)
    n_validation = round(fractions[1] * n_units)
    n_test = n_units - n_train - n_validation
    return n_train, n_validation, n_test


def holdout_indices(
    strategy: str,
    groups: ArrayLike,
    *,
    fractions: Sequence[float] = (0.7, 0.15, 0.15),
    rng: np.random.Generator | int | None = None,
) -> SplitIndices:
    """A single training / validation / test division.

    The fractions apply to packets for ``packet_random``, to reference
    locations for ``location_block`` and ``location_holdout`` (each location
    goes whole into one part), and to each location's packets for ``temporal``
    (the earliest packets train, the latest test).

    Parameters
    ----------
    strategy : str
        One of :data:`SPLIT_STRATEGIES`.
    groups : array_like of str, shape (n_items,)
        Reference point of every item, items in recording order.
    fractions : sequence of three floats, optional
        Training, validation and test shares; they must add up to 1.
    rng : numpy.random.Generator or int, optional
        Randomness for the shuffled strategies.

    Returns
    -------
    SplitIndices
        The division.

    Raises
    ------
    ConfigError
        For an unknown strategy, invalid fractions, or too little data for
        non-empty training and test parts.

    Examples
    --------
    56 locations with 200 packets each, divided into contiguous location
    blocks of 40, 8 and 8 locations:

    >>> groups = np.repeat(np.arange(56), 200)
    >>> split = holdout_indices("location_block", groups, fractions=(5 / 7, 1 / 7, 1 / 7))
    >>> split.sizes(), int(split.train[-1]), int(split.test[0])
    ({'train': 8000, 'validation': 1600, 'test': 1600}, 7999, 9600)
    """
    _check_strategy(strategy)
    shares = np.asarray(fractions, dtype=np.float64)
    if shares.shape != (3,) or np.any(shares < 0) or not np.isclose(shares.sum(), 1.0):
        raise ConfigError(
            f"split fractions must be three non-negative numbers adding up to 1, got {fractions!r}"
        )
    rng = np.random.default_rng(rng)
    codes = group_codes(groups)
    n_items = codes.size
    n_groups = int(codes.max()) + 1
    part = np.empty(n_items, dtype=np.int64)  # 0 train, 1 validation, 2 test

    if strategy == "temporal":
        rank, sizes = _rank_within_group(codes)
        n_train = np.rint(shares[0] * sizes).astype(np.int64)[codes]
        n_validation = np.rint(shares[1] * sizes).astype(np.int64)[codes]
        part[:] = 2
        part[rank < n_train + n_validation] = 1
        part[rank < n_train] = 0
    else:
        n_units = n_items if strategy == "packet_random" else n_groups
        counts = _unit_counts(n_units, shares)
        if counts[0] < 1 or counts[2] < 1:
            raise ConfigError(
                f"fractions {tuple(float(s) for s in shares)} leave no training or test data for "
                f"{n_units} unit(s) under the {strategy!r} split"
            )
        order = np.arange(n_units) if strategy == "location_block" else rng.permutation(n_units)
        unit_part = np.empty(n_units, dtype=np.int64)
        unit_part[order[: counts[0]]] = 0
        unit_part[order[counts[0] : counts[0] + counts[1]]] = 1
        unit_part[order[counts[0] + counts[1] :]] = 2
        part[:] = unit_part if strategy == "packet_random" else unit_part[codes]

    split = SplitIndices(
        train=np.flatnonzero(part == 0).astype(np.int64),
        validation=np.flatnonzero(part == 1).astype(np.int64),
        test=np.flatnonzero(part == 2).astype(np.int64),
    )
    if split.train.size == 0 or split.test.size == 0:
        raise ConfigError(f"the {strategy!r} split left no training or test data")
    return split
