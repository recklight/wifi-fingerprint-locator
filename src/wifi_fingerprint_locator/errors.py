"""Exceptions raised by the package.

Every expected failure is a ``ValueError`` (or ``FileNotFoundError`` for a
missing input), so the command line catches exactly those and turns them into
a one-line message.
"""

from __future__ import annotations


class ConfigError(ValueError):
    """Raised when a setting is unknown, missing or out of range."""


class DatasetError(ValueError):
    """Raised when a dataset index or one of its files cannot be used."""


class ModelFileError(ValueError):
    """Raised when a model file is incomplete or was written in another format."""


class BeamformFormatError(ValueError):
    """Raised when a beamforming report is malformed or internally inconsistent."""
