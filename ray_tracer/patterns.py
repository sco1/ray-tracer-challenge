from __future__ import annotations

import math
import typing as t
from dataclasses import dataclass, field

from ray_tracer.colors import BLACK, WHITE
from ray_tracer.rayple import Rayple
from ray_tracer.transforms import Matrix

if t.TYPE_CHECKING:
    from ray_tracer.shapes import Shape


@dataclass(frozen=True, slots=True)
class Pattern:
    """
    Base class for creating pattern objects; this is not intended to be instantiated.

    Child classes must define `at_point` to calculate the pattern's color at the given point. This
    method assumes that the point has already been passed the appropriate transormations and is
    presented in pattern space.
    """

    a: Rayple = WHITE
    b: Rayple = BLACK
    transform: Matrix = field(default_factory=Matrix.identity)

    def at_point(self, pt: Rayple) -> Rayple:  # pragma: no cover  # noqa: D102
        raise NotImplementedError

    def at_object(self, obj: Shape, world_pt: Rayple) -> Rayple:
        """Apply the appropriate transformations to shift the query point into pattern space."""
        object_pt = obj.transform.inv() * world_pt
        pattern_pt = self.transform.inv() * object_pt

        return self.at_point(pattern_pt)


@dataclass(frozen=True, slots=True)
class Stripe(Pattern):
    """Alternating color pattern along the x-axis."""

    def at_point(self, pt: Rayple) -> Rayple:  # noqa: D102
        if math.floor(pt.x) % 2 == 0:
            return self.a
        else:
            return self.b


@dataclass(frozen=True, slots=True)
class Gradient(Pattern):
    """Linear gradient along the x-axis."""

    def at_point(self, pt: Rayple) -> Rayple:  # noqa: D102
        # Linear blending function along x
        distance = self.b - self.a
        fractional = pt.x - math.floor(pt.x)

        return self.a + distance * fractional


@dataclass(frozen=True, slots=True)
class Ring(Pattern):
    """Alternating color pattern in concentric rings around the y-axis."""

    def at_point(self, pt: Rayple) -> Rayple:  # noqa: D102
        if math.floor(math.sqrt(pt.x**2 + pt.z**2)) % 2 == 0:
            return self.a
        else:
            return self.b


@dataclass(frozen=True, slots=True)
class Checker(Pattern):
    """Repeating pattern of squares in 3 dimensions."""

    def at_point(self, pt: Rayple) -> Rayple:  # noqa: D102
        if sum(math.floor(n) for n in pt) % 2 == 0:
            return self.a
        else:
            return self.b
