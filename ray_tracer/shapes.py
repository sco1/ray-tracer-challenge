import math
from dataclasses import dataclass

from ray_tracer import NUMERIC_T
from ray_tracer.intersections import Intersection, Intersections
from ray_tracer.rayple import Rayple, RaypleType, dot, point
from ray_tracer.rays import Ray


class ShapeBase:  # noqa: D101
    ...


@dataclass(frozen=True, slots=True)
class Sphere(ShapeBase):  # noqa: D101
    origin: Rayple = point(0, 0, 0)
    radius: NUMERIC_T = 1

    def __post_init__(self) -> None:
        if self.origin.w != RaypleType.POINT:
            raise ValueError("Sphere origin must be a point")

        if self.radius < 1:
            raise ValueError("Sphere radius must be non-negative.")

    def intersect(self, ray: Ray) -> Intersections:
        """
        Calculate the time position(s) where the provided Ray intersects the `Sphere`.

        NOTE: To assist with object overlaps, if the ray is tangent to the sphere, the time position
        is returned twice.

        NOTE: Negative timesteps are considered, so if the ray originates inside or beyond the
        sphere then negative value(s) can be returned.
        """
        # Calculate the discriminant to determine if there are any intersections
        sphere_to_ray = ray.origin - self.origin
        a = dot(ray.direction, ray.direction)
        b = 2 * dot(ray.direction, sphere_to_ray)
        c = dot(sphere_to_ray, sphere_to_ray) - 1
        discriminant = b**2 - (4 * a * c)

        if discriminant < 0:
            intersections = []
        else:
            intersections = [
                Intersection(((-b - math.sqrt(discriminant))) / (2 * a), self),
                Intersection(((-b + math.sqrt(discriminant))) / (2 * a), self),
            ]

        return Intersections(intersections)
