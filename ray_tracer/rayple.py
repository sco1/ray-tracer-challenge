from __future__ import annotations

import math
import typing as t
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

NUMERIC_T: t.TypeAlias = int | float


class RaypleType(IntEnum):  # noqa: D101
    VECTOR = 0
    POINT = 1
    COLOR = 2


@dataclass(frozen=True, slots=True)
class Rayple:
    """
    The Ray Tracer's generic ordered list of (x,y,z) things, classified by `Rayple.w`.

    For simplicity, Colors are included under this class, where (x,y,z)<->(r,g,b).

    Iterating over a `Rayple` yields its (x,y,z).

    The following generic operations are supported:
        * Addition
            * Adding two Points is undefined
            * Colors may only be added to Colors
        * Subtraction
            * Subtracting a Point from a Vector is undefined
            * Colors may only be subtracted from Colors
        * Negation

    The following operations are supported for Vectors and Colors:
        * Scalar multiplication
        * Scalar division

    The following Color-only operations are supported for Colors:
        * Multipliciation (aka Hadamard or Schur product)

    The following Vector-only helpers are supported:
        * Magnitude (implemented by `__abs__`)
        * Normalization
        * Dot product
        * Cross product
    """

    x: NUMERIC_T
    y: NUMERIC_T
    z: NUMERIC_T
    w: RaypleType | int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rayple):
            return NotImplemented

        return all(
            (
                self.w == other.w,
                math.isclose(self.x, other.x),
                math.isclose(self.y, other.y),
                math.isclose(self.z, other.z),
            )
        )

    def __add__(self, other: object) -> Rayple:
        if not isinstance(other, Rayple):
            return NotImplemented

        if self.w == RaypleType.POINT and other.w == RaypleType.POINT:
            raise TypeError("Cannot add two Points.")

        if self.w == RaypleType.COLOR and other.w != RaypleType.COLOR:
            raise TypeError("Cannot add non-Color to Color.")

        # Colors break our clever enum addition logic so we have to calc separately
        if self.w == RaypleType.COLOR:
            out_type = RaypleType.COLOR
        else:
            out_type = RaypleType(self.w + other.w)

        return Rayple(
            x=self.x + other.x,
            y=self.y + other.y,
            z=self.z + other.z,
            w=out_type,
        )

    def __sub__(self, other: object) -> Rayple:
        if not isinstance(other, Rayple):
            return NotImplemented

        if self.w == RaypleType.VECTOR and other.w == RaypleType.POINT:
            raise TypeError("Cannot subtract a Point from a Vector.")

        if self.w == RaypleType.COLOR and other.w != RaypleType.COLOR:
            raise TypeError("Cannot subtract non-Color from Color.")

        # Colors break our clever enum subtraction logic so we have to calc separately
        if self.w == RaypleType.COLOR:
            out_type = RaypleType.COLOR
        else:
            out_type = RaypleType(self.w - other.w)

        return Rayple(
            x=self.x - other.x,
            y=self.y - other.y,
            z=self.z - other.z,
            w=out_type,
        )

    def __neg__(self) -> Rayple:
        return Rayple(
            x=-self.x,
            y=-self.y,
            z=-self.z,
            w=self.w,
        )

    def __mul__(self, other: object) -> Rayple:
        if isinstance(other, Rayple):
            if self.w != RaypleType.COLOR or other.w != RaypleType.COLOR:
                raise TypeError(
                    f"Nonscalar multiplication only supported between Colors. Received: {self.w} and {other.w}."  # noqa: E501
                )
        elif not isinstance(other, (int, float)):
            return NotImplemented

        if isinstance(other, Rayple):
            return Rayple(
                x=self.x * other.x,
                y=self.y * other.y,
                z=self.z * other.z,
                w=self.w,
            )
        else:
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
        if self.w != RaypleType.VECTOR:
            raise TypeError("Cannot calculate the magnitude of a non-Vector.")

        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __iter__(self) -> t.Generator[NUMERIC_T, None, None]:
        yield self.x
        yield self.y
        yield self.z

    def normalize(self) -> Rayple:
        """Normalize into a unit Vector."""
        if self.w != RaypleType.VECTOR:
            raise TypeError("Cannot normalize a non-Vector.")

        return Rayple(
            x=self.x / abs(self),
            y=self.y / abs(self),
            z=self.z / abs(self),
            w=self.w,
        )

    def as_array(self) -> np.ndarray:
        """Provide the `Rayple` as a `1x4` array."""
        return np.array((self.x, self.y, self.z, self.w))

    @classmethod
    def from_np(cls, in_vec: np.ndarray) -> Rayple:
        """Create an instance from a NumPy vector, ensuring that `w` is an integer."""
        if in_vec.size != 4:
            raise ValueError(f"Input vector must have 4 elements, received: {in_vec.size}")

        # Unpack manually since mypy can't reason with the splat
        x, y, z = in_vec[:3]
        w = int(in_vec[3])  # Cast may be fragile in some cases, but is workable for now
        return Rayple(x, y, z, w)


def dot(left: Rayple, right: Rayple) -> NUMERIC_T:
    """
    Calculate the dot product of two Vectors.

    See: https://en.wikipedia.org/wiki/Dot_product
    """
    if not (left.w == RaypleType.VECTOR and right.w == RaypleType.VECTOR):
        raise ValueError(f"Both operands must be vectors. Received: {left.w} and {right.w}.")

    return (left.x * right.x) + (left.y * right.y) + (left.z * right.z)


def cross(left: Rayple, right: Rayple) -> Rayple:
    """
    Calculate the cross product of two Vectors.

    See: https://en.wikipedia.org/wiki/Cross_product
    """
    if not (left.w == RaypleType.VECTOR and right.w == RaypleType.VECTOR):
        raise ValueError(f"Both operands must be vectors. Received: {left.w} and {right.w}.")

    return Rayple(
        x=(left.y * right.z - left.z * right.y),
        y=(left.z * right.x - left.x * right.z),
        z=(left.x * right.y - left.y * right.x),
        w=RaypleType.VECTOR,
    )


def point(x: NUMERIC_T, y: NUMERIC_T, z: NUMERIC_T) -> Rayple:
    """Shortcut for a Point `Rayple` (`w = 1`)."""
    return Rayple(x, y, z, RaypleType.POINT)


def vector(x: NUMERIC_T, y: NUMERIC_T, z: NUMERIC_T) -> Rayple:
    """Shortcut for a Vector `Rayple` (`w = 0`)."""
    return Rayple(x, y, z, RaypleType.VECTOR)


def color(red: NUMERIC_T, green: NUMERIC_T, blue: NUMERIC_T) -> Rayple:
    """Shortcut for a Color `Rayple` (`w = 2`)."""
    return Rayple(red, green, blue, RaypleType.COLOR)


def is_point(inp: Rayple) -> bool:  # noqa: D103
    return inp.w == RaypleType.POINT


def is_vector(inp: Rayple) -> bool:  # noqa: D103
    return inp.w == RaypleType.VECTOR


def is_color(inp: Rayple) -> bool:  # noqa: D103
    return inp.w == RaypleType.COLOR
