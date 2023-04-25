from __future__ import annotations

import typing as t
from collections import UserList
from dataclasses import dataclass, field
from functools import cached_property

from ray_tracer import EPSILON, NUMERIC_T
from ray_tracer.rayple import Rayple, dot
from ray_tracer.rays import Ray

if t.TYPE_CHECKING:
    from ray_tracer.shapes import Shape


@dataclass(frozen=True, slots=True)
class Intersection:  # noqa: D101
    t: NUMERIC_T
    obj: Shape


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
    obj: Shape
    point: Rayple
    eye_v: Rayple
    normal: Rayple
    inside: bool
    reflect_v: Rayple
    n1: NUMERIC_T  # Material being exited
    n2: NUMERIC_T  # Material being entered
    over_point: Rayple = field(init=False)
    under_point: Rayple = field(init=False)

    def __post_init__(self) -> None:
        # Create points shifted slightly in each normal direction to help prevent self-shadowing due
        # to floating point issues; if a point is on the surface it may be accidentally considered
        # inside/outside, depending on the error direction
        self.over_point = self.point + self.normal * EPSILON
        self.under_point = self.point - self.normal * EPSILON


def _calc_refractive_indices(
    inter: Intersection, all_inters: Intersections
) -> tuple[NUMERIC_T, NUMERIC_T]:
    """
    Calculate the refractive indices of the materials on either side of a ray-object intersection.

    Indices are return as a (<exited material>, <entered material>) tuple pair.
    """
    # Record the shapes that have been encountered but not yet exited
    # May need to use a different container if all the membership checks end up being onerous
    containers: list[Shape] = []
    # We can assume that all_inters is never going to be empty
    for i in all_inters:  # pragma: no branch
        # Material being exited
        if i == inter:
            if not containers:
                # No containing object
                n1: NUMERIC_T = 1
            else:
                n1 = containers[-1].material.refractive_index

        # If the intersections object is already in the containers list, then this intersection is
        # assumed to be exiting the object. Otherwise, the intersection is entering the object and
        # it should be added to the list of container shapes
        try:
            containers.remove(i.obj)
        except ValueError:
            containers.append(i.obj)

        # Material being entered
        if i == inter:
            if not containers:
                # No containing object
                n2: NUMERIC_T = 1
            else:
                n2 = containers[-1].material.refractive_index

            break

    return n1, n2


def prepare_computations(
    inter: Intersection, ray: Ray, all_inters: Intersections | None = None
) -> Comps:
    """
    Precompute helper information relating to the provided `Ray` intersection.

    To assist with refraction calculations, `all_inters` may be passed to determine where the hit is
    relative to the rest of the intersections. If not provided, it is seeded with `inter`.
    """
    if all_inters is None:
        all_inters = Intersections([inter])

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

    reflect_v = ray.direction.reflect(normal)
    n1, n2 = _calc_refractive_indices(inter, all_inters)

    return Comps(
        t=inter.t,
        obj=inter.obj,
        point=pt,
        eye_v=eye_v,
        normal=normal,
        inside=inside,
        reflect_v=reflect_v,
        n1=n1,
        n2=n2,
    )
