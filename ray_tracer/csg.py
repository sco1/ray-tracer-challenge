from dataclasses import dataclass
from enum import Enum, auto

from ray_tracer.intersections import Intersections
from ray_tracer.rays import Ray
from ray_tracer.shapes import Group, Shape


class Operation(Enum):  # noqa: D101
    DIFFERENCE = auto()
    INTERSECTION = auto()
    UNION = auto()


@dataclass(kw_only=True, slots=True, eq=False)
class CSG(Group):
    """
    Constructive Solid Geometry (CSG) implementation.

    CSGs combine geometry primitives into more complex shapes via set operations: union,
    intersection, and difference. For simplicity, this implementation treats CSG operations as
    strictly binary, but more complex operations can be accomplished by combining into a hierarchy.
    """

    operation: Operation
    left_shape: Shape
    right_shape: Shape

    def __post_init__(self) -> None:
        self.add_child(self.left_shape)
        self.add_child(self.right_shape)

    def _local_intersect(self, transformed_ray: Ray) -> Intersections:
        all_inters = self.left_shape.intersect(transformed_ray)
        all_inters.extend(self.right_shape.intersect(transformed_ray))
        all_inters.sort()

        return self._filter_intersections(all_inters)

    def _is_inter_allowed(self, left_hit: bool, in_left: bool, in_right: bool) -> bool:
        if self.operation == Operation.UNION:
            return (left_hit and not in_right) or (not left_hit and not in_left)

        if self.operation == Operation.INTERSECTION:
            return (left_hit and in_right) or (not left_hit and in_left)

        if self.operation == Operation.DIFFERENCE:
            return (left_hit and not in_right) or (not left_hit and in_left)

        return False

    def _filter_intersections(self, inters: Intersections) -> Intersections:
        in_left = False
        in_right = False
        filtered_inters = Intersections([])

        for inter in inters:
            left_hit = check_includes(self.left_shape, inter.obj)

            if self._is_inter_allowed(left_hit, in_left, in_right):
                filtered_inters.append(inter)

            if left_hit:
                in_left = not in_left
            else:
                in_right = not in_right

        return filtered_inters


def check_includes(a: Shape, b: Shape) -> bool:
    """
    Check to see if `a` includes `b`.

    If `a` is a `Group` (includes `CSG`), return `True` if any child of `a` includes `b`, otherwise
    if `a` is any other shape, compare them directly.
    """
    obj_queue = [a]
    while obj_queue:
        obj = obj_queue.pop()

        if isinstance(obj, Group):
            if b in obj.children:
                return True
            else:
                obj_queue.extend(obj.children)
        else:
            return obj == b

    return False
