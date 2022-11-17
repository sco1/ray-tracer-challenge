from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

NUMERIC_T = int | float


class RaypleType(IntEnum):  # noqa: D101
    VECTOR = 0
    POINT = 1


@dataclass(frozen=True, slots=True)
class Rayple:
    """
    The Ray Tracer's generic ordered list of (x,y,z) things, classified by `Rayple.w`.

    The following generic operations are supported:
        * Vector/Point addition
            * Adding two Points is undefined
        * Vector/Point subtraction
            * Subtracting a Point from a Vector is undefined
        * Negation

    The following Vector-only helpers are supported:
        * Scalar multiplication
        * Scalar division
        * Magnitude (implemented by `__abs__`)
        * Normalization
        * Dot product
        * Cross product
    """

    x: NUMERIC_T
    y: NUMERIC_T
    z: NUMERIC_T
    w: RaypleType

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rayple):
            return NotImplemented

        return all(
            (
                math.isclose(self.x, other.x),
                math.isclose(self.y, other.y),
                math.isclose(self.z, other.z),
                self.w == other.w,
            )
        )

    def __add__(self, other: object) -> Rayple:
        if not isinstance(other, Rayple):
            return NotImplemented

        if self.w == RaypleType.POINT and other.w == RaypleType.POINT:
            raise ValueError("Cannot add two points.")

        return Rayple(
            x=self.x + other.x,
            y=self.y + other.y,
            z=self.z + other.z,
            w=RaypleType(self.w + other.w),
        )

    def __sub__(self, other: object) -> Rayple:
        if not isinstance(other, Rayple):
            return NotImplemented

        if self.w == RaypleType.VECTOR and other.w == RaypleType.POINT:
            raise ValueError("Cannot subtract a point from a vector.")

        return Rayple(
            x=self.x - other.x,
            y=self.y - other.y,
            z=self.z - other.z,
            w=RaypleType(self.w - other.w),
        )

    def __neg__(self) -> Rayple:
        return Rayple(
            x=-self.x,
            y=-self.y,
            z=-self.z,
            w=self.w,
        )

    def __mul__(self, other: object) -> Rayple:
        if not isinstance(other, (int, float)):
            return NotImplemented

        return Rayple(
            x=self.x * other,
            y=self.y * other,
            z=self.z * other,
            w=self.w,
        )

    def __rmul__(self, other: object) -> Rayple:
        return self * other

    def __truediv__(self, other: object) -> Rayple:
        if not isinstance(other, (int, float)):
            return NotImplemented

        return Rayple(
            x=self.x / other,
            y=self.y / other,
            z=self.z / other,
            w=self.w,
        )

    def __abs__(self) -> float:
        if self.w == RaypleType.POINT:
            raise ValueError("Points have no magnitude.")

        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> Rayple:
        """
        Normalize into a unit vector.

        NOTE: This method is undefined for Points.
        """
        if self.w == RaypleType.POINT:
            raise ValueError("Canont normalize a point.")

        return Rayple(
            x=self.x / abs(self),
            y=self.y / abs(self),
            z=self.z / abs(self),
            w=self.w,
        )


def dot(left: Rayple, right: Rayple) -> NUMERIC_T:
    """
    Calculate the dot product of two vectors.

    NOTE: This method is undefined for Points.
    See: https://en.wikipedia.org/wiki/Dot_product
    """
    if left.w == RaypleType.POINT or right.w == RaypleType.POINT:
        raise ValueError(f"Both operands must be vectors. Received: {left.w} and {right.w}.")

    return (left.x * right.x) + (left.y * right.y) + (left.z * right.z)


def cross(left: Rayple, right: Rayple) -> Rayple:
    """
    Calculate the cross product of two vectors.

    NOTE: This method is undefined for Points.
    See: https://en.wikipedia.org/wiki/Cross_product
    """
    if left.w == RaypleType.POINT or right.w == RaypleType.POINT:
        raise ValueError(f"Both operands must be vectors. Received: {left.w} and {right.w}.")

    return Rayple(
        x=(left.y * right.z - left.z * right.y),
        y=(left.z * right.x - left.x * right.z),
        z=(left.x * right.y - left.y * right.x),
        w=RaypleType.VECTOR,
    )


def point(x: NUMERIC_T, y: NUMERIC_T, z: NUMERIC_T) -> Rayple:
    """Shortcut for a point `Rayple` (`w = 1`)."""
    return Rayple(x, y, z, RaypleType.POINT)


def vector(x: NUMERIC_T, y: NUMERIC_T, z: NUMERIC_T) -> Rayple:
    """Shortcut for a vector `Rayple` (`w = 0`)."""
    return Rayple(x, y, z, RaypleType.VECTOR)


def is_point(inp: Rayple) -> bool:  # noqa: D103
    return inp.w == RaypleType.POINT


def is_vector(inp: Rayple) -> bool:  # noqa: D103
    return inp.w == RaypleType.VECTOR
