import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ray_tracer.intersections import Intersection, Intersections
from ray_tracer.materials import Material
from ray_tracer.rayple import Rayple, RaypleType, dot, point, vector
from ray_tracer.rays import Ray
from ray_tracer.transforms import Matrix


class ShapeBase(ABC):  # noqa: D101  # pragma: no cover
    transform: Matrix
    material: Material

    @abstractmethod
    def intersect(self, ray: Ray) -> Intersections:  # noqa: D102
        ...

    @abstractmethod
    def normal_at(self, query: Rayple) -> Rayple:  # noqa: D102
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
    material: Material = Material()

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

    def normal_at(self, query: Rayple) -> Rayple:
        """Calculate the normal vector from the `Sphere` at the provided surface point."""
        if query.w != RaypleType.POINT:
            raise ValueError("Query location must be a point.")

        # To account for transformations, we have to shift the query point from world space to the
        # object space, otherwise our initial normal is not going to be accurate
        object_point = self.transform.inv() * query
        object_normal = object_point - point(0, 0, 0)

        # Once we've shifted to object space to get the object normal, we need to shift this back to
        # the world space by transforming it with the inverse transpose of the sphere's
        # transformation matrix
        # Technically, we should be finding the 3x3 submatrix of the transform in order to prevent
        # any translation from messing with the rayple type; to get around this, we can just make a
        # new vector from the components
        world_normal = self.transform.inv().transpose() * object_normal
        world_normal = vector(*world_normal)

        return world_normal.normalize()
