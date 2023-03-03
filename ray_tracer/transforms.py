from __future__ import annotations

import typing as t
from dataclasses import dataclass

import numpy as np

from ray_tracer.rayple import Rayple

NUMERIC_T: t.TypeAlias = int | float


@dataclass(slots=True)
class Matrix:
    """Thin wrapper around `np.ndarray` to support `Rayple` multiplication."""

    matrix: np.ndarray

    def __mul__(self, other: object) -> Rayple:
        if not isinstance(other, Rayple):
            return NotImplemented

        transformed = self.matrix.dot(other.as_array())
        return Rayple.from_np(transformed)

    def inv(self) -> Matrix:
        """Return an inverted `Matrix` instance."""
        return Matrix(np.linalg.inv(self.matrix))


def translation(x: NUMERIC_T, y: NUMERIC_T, z: NUMERIC_T) -> Matrix:
    """Generate a `4x4` translation matrix for the provided shift components."""
    matrix = np.identity(4)
    matrix[0:3, 3] = (x, y, z)

    return Matrix(matrix)


def scaling(x: NUMERIC_T, y: NUMERIC_T, z: NUMERIC_T) -> Matrix:
    """Generate a `4x4` scaling matrix for the provided scaling components."""
    matrix = np.identity(4)
    np.fill_diagonal(matrix, (x, y, z, 1))

    return Matrix(matrix)
