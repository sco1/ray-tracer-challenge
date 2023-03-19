from __future__ import annotations

import typing as t
from dataclasses import dataclass
from math import cos, sin

import numpy as np

from ray_tracer import NUMERIC_T
from ray_tracer.rayple import Rayple, cross

ZERO_TOL = 1e-16


@dataclass(slots=True)
class Matrix:
    """Thin wrapper around `np.ndarray` to support `Rayple` multiplication."""

    matrix: np.ndarray

    @t.overload
    def __mul__(self, other: Rayple) -> Rayple:
        ...

    @t.overload
    def __mul__(self, other: Matrix) -> Matrix:
        ...

    def __mul__(self, other: object) -> Rayple | Matrix:
        if isinstance(other, Rayple):
            transformed = self.matrix.dot(other.as_array())
            transformed[np.abs(transformed) < ZERO_TOL] = 0

            return Rayple.from_np(transformed)
        elif isinstance(other, Matrix):
            chained = self.matrix.dot(other.matrix)
            chained[np.abs(chained) < ZERO_TOL] = 0

            return Matrix(chained)
        else:
            return NotImplemented

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):  # pragma: no cover
            return NotImplemented

        return np.allclose(self.matrix, other.matrix, rtol=1e-4)

    def inv(self) -> Matrix:
        """Return an inverted `Matrix` instance."""
        return Matrix(np.linalg.inv(self.matrix))

    def transpose(self) -> Matrix:
        """Return a transposed `Matrix` instance."""
        return Matrix(self.matrix.T)

    @classmethod
    def identity(cls) -> Matrix:
        """Initialize an identity matrix."""
        return cls(np.identity(4))


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


def rot_x(mag: float) -> Matrix:
    """
    Generate a `4x4` 3D rotation matrix (left-hand rule) around the x-axis.

    NOTE: Rotation magnitude is assumed to be given in radians.
    """
    matrix = np.identity(4)
    np.fill_diagonal(matrix, (1, cos(mag), cos(mag), 1))
    np.fill_diagonal(np.flipud(matrix), (0, sin(mag), -sin(mag), 0))  # anti-diagonal

    return Matrix(matrix)


def rot_y(mag: float) -> Matrix:
    """
    Generate a `4x4` 3D rotation matrix (left-hand rule) around the y-axis.

    NOTE: Rotation magnitude is assumed to be given in radians.
    """
    matrix = np.identity(4)
    matrix[0, :] = (cos(mag), 0, sin(mag), 0)
    matrix[2, :] = (-sin(mag), 0, cos(mag), 0)

    return Matrix(matrix)


def rot_z(mag: float) -> Matrix:
    """
    Generate a `4x4` 3D rotation matrix (left-hand rule) around the z-axis.

    NOTE: Rotation magnitude is assumed to be given in radians.
    """
    matrix = np.identity(4)
    matrix[0, :] = (cos(mag), -sin(mag), 0, 0)
    matrix[1, :] = (sin(mag), cos(mag), 0, 0)

    return Matrix(matrix)


def rot(x: float = 0, y: float = 0, z: float = 0) -> Matrix:
    """
    Generate a `4x4` 3D rotation matrix (left-hand rule) around the origin.

    NOTE: Rotation magnitudes are assumed to be given in radians.
    """
    # Reverse the dot product order so we do x-plane, y-plane, then z-plane rotation
    matrix = rot_z(z) * rot_y(y) * rot_x(x)
    return matrix


def shearing(
    x_y: NUMERIC_T = 0,
    x_z: NUMERIC_T = 0,
    y_x: NUMERIC_T = 0,
    y_z: NUMERIC_T = 0,
    z_x: NUMERIC_T = 0,
    z_y: NUMERIC_T = 0,
) -> Matrix:
    """Generate a shearing (skew) transformation matrix for the provided component proportions."""
    matrix = np.array(
        [
            [1, x_y, x_z, 0],
            [y_x, 1, y_z, 0],
            [z_x, z_y, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    return Matrix(matrix)


def view_transform(from_p: Rayple, to_p: Rayple, up_v: Rayple) -> Matrix:
    """Create a transformation matrix from a given point to the provided point & orientation."""
    forward = (to_p - from_p).normalize()
    up_norm = up_v.normalize()
    left = cross(forward, up_norm)
    true_up = cross(left, forward)

    orientation = np.identity(4)
    orientation[0, 0:3] = [*left]
    orientation[1, 0:3] = [*true_up]
    orientation[2, 0:3] = [*-forward]

    return Matrix(orientation) * translation(*-from_p)
