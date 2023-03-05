import math
from dataclasses import dataclass, field

from ray_tracer.intersections import Intersection, Intersections
from ray_tracer.rayple import dot, point
from ray_tracer.rays import Ray
from ray_tracer.transforms import Matrix


class ShapeBase:  # noqa: D101
    ...


@dataclass(frozen=True, slots=True, eq=False)
class Sphere(ShapeBase):
    """
    Unit sphere representation.

    NOTE: Spheres are assumed to be unit spheres, i.e. centered at `(0, 0, 0)` with a radius of `1`.
    Sphere position, size, and rotation can be adjusted by changing the value of `transform`.

    NOTE: Spheres are compared by object ID only, so no 2 instances will compare `True`
    """

    transform: Matrix = field(default_factory=Matrix.identity)

    def intersect(self, ray: Ray) -> Intersections:
        """
        Calculate the time position(s) where the provided Ray intersects the `Sphere`.

        NOTE: To assist with object overlaps, if the ray is tangent to the sphere, the time position
        is returned twice.

        NOTE: Negative timesteps are considered, so if the ray originates inside or beyond the
        sphere then negative value(s) can be returned.
        """
        # Calculate the discriminant to determine if there are any intersections
        # Apply the inverse of the sphere's transformation to the ray to account for the desired
        # sphere transformation
        transformed_ray = ray.transform(self.transform.inv())
        sphere_to_ray = transformed_ray.origin - point(0, 0, 0)
        a = dot(transformed_ray.direction, transformed_ray.direction)
        b = 2 * dot(transformed_ray.direction, sphere_to_ray)
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
