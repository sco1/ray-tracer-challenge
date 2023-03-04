from __future__ import annotations

import typing as t
from collections import UserList
from dataclasses import dataclass
from functools import cached_property

from ray_tracer import NUMERIC_T

if t.TYPE_CHECKING:
    from ray_tracer.shapes import ShapeBase


@dataclass(frozen=True, slots=True)
class Intersection:  # noqa: D101
    t: NUMERIC_T
    obj: ShapeBase


class Intersections(UserList):
    """
    Helper container for shape intersections.

    Intersections, if present, will always be sorted by `Intersection.t`
    """

    def __init__(self, in_data: t.Iterable[Intersection]) -> None:
        self.data: list[Intersection] = list(in_data)
        if self.data:
            self.data.sort(key=lambda x: x.t)

    @cached_property
    def hit(self) -> Intersection | None:
        """Determine the lowest non-negative intersection, otherwise return `None`."""
        for intersect in self.data:
            if intersect.t > 0:
                return intersect
        else:
            return None
