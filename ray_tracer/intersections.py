from __future__ import annotations

import typing as t
from collections import UserList
from dataclasses import dataclass, field
from functools import cached_property

from ray_tracer import EPSILON, NUMERIC_T
from ray_tracer.rayple import Rayple, dot
from ray_tracer.rays import Ray

if t.TYPE_CHECKING:
    from ray_tracer.shapes import ShapeBase


@dataclass(frozen=True, slots=True)
class Intersection:  # noqa: D101
    t: NUMERIC_T
    obj: ShapeBase


class Intersections(UserList):
    """
    Helper container for shape intersections.

    Intersections, if present, will be sorted by `Intersection.t` on instantiation.
    """

    def __init__(self, in_data: t.Iterable[Intersection]) -> None:
        self.data: list[Intersection] = list(in_data)
        self.sort()

    def sort(self, reverse: bool = False) -> None:  # type: ignore[override]  # noqa: D102
        self.data.sort(key=lambda x: x.t, reverse=reverse)

    @cached_property
    def hit(self) -> Intersection | None:
        """Determine the lowest non-negative intersection, otherwise return `None`."""
        for intersect in self.data:
            if intersect.t > 0:
                return intersect
        else:
            return None


@dataclass(slots=True)
class Comps:  # noqa: D101
    t: NUMERIC_T
    obj: ShapeBase
    point: Rayple
    eye_v: Rayple
    normal: Rayple
    inside: bool
    over_point: Rayple = field(init=False)

    def __post_init__(self) -> None:
        # Create a point shifted slightly in the direction of the normal to help prevent
        # self-shadowing due to floating point issues; if a point is on the surface it may be
        # accidentally considered inside
        self.over_point = self.point + self.normal * EPSILON


def prepare_computations(inter: Intersection, ray: Ray) -> Comps:
    """Precompute helper information relating to the provided `Ray` intersection."""
    pt = ray.position(inter.t)

    # Check if the hit occurs on the inside of the shape; if it does, the normal needs to be
    # inverted so the surface is illuminated properly
    # We can roughly check this by seeing if the normal points away from the eye vector
    eye_v = -ray.direction
    normal = inter.obj.normal_at(pt)
    if dot(normal, eye_v) < 0:
        inside = True
        normal = -normal
    else:
        inside = False

    return Comps(t=inter.t, obj=inter.obj, point=pt, eye_v=eye_v, normal=normal, inside=inside)
