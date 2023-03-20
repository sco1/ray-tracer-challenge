import math
from dataclasses import dataclass, field

from ray_tracer.intersections import Intersection, Intersections
from ray_tracer.materials import Material
from ray_tracer.rayple import Rayple, RaypleType, dot, point, vector
from ray_tracer.rays import Ray
from ray_tracer.transforms import Matrix


@dataclass(frozen=True, slots=True, eq=False)
class ShapeBase:
    """
    Base class for creating shape objects; this is not intended to be instantiated.

    Child instances must define `_local_intersect` and `_local_normal_at` to calculate their
    respective local values, which are then transformed into world coordinates by the base methods.
    """

    transform: Matrix = field(default_factory=Matrix.identity)
    material: Material = Material()

    def _local_intersect(self, local_ray: Ray) -> Intersections:  # pragma: no cover
        raise NotImplementedError

    def intersect(self, ray: Ray) -> Intersections:
        """
        Calculate the time position(s) where the provided Ray intersects the shape.

        NOTE: To assist with object overlaps, if the ray is tangent to the shape, the time position
        is returned twice.

        NOTE: Negative timesteps are considered, so if the ray originates inside or beyond the
        shape then negative value(s) can be returned.
        """
        # Apply the inverse of the shape's transformation to the ray to account for the desired
        # shape transformation
        transformed_ray = ray.transform(self.transform.inv())
        return self._local_intersect(transformed_ray)

    def _local_normal_at(self, local_point: Rayple) -> Rayple:  # pragma: no cover
        raise NotImplementedError

    def normal_at(self, query: Rayple) -> Rayple:
        """Calculate the normal vector from the shape at the provided surface point."""
        if query.w != RaypleType.POINT:
            raise ValueError("Query location must be a point.")

        # To account for transformations, we have to shift the query point from world space to the
        # object space, otherwise our initial normal is not going to be accurate
        local_point = self.transform.inv() * query
        local_normal = self._local_normal_at(local_point)

        # Once we've shifted to object space to get the object normal, we need to shift this back to
        # the world space by transforming it with the inverse transpose of the sphere's
        # transformation matrix
        # Technically, we should be finding the 3x3 submatrix of the transform in order to prevent
        # any translation from messing with the rayple type; to get around this, we can just make a
        # new vector from the components
        world_normal = self.transform.inv().transpose() * local_normal
        world_normal = vector(*world_normal)

        return world_normal.normalize()


@dataclass(frozen=True, slots=True, eq=False)
class Sphere(ShapeBase):
    """
    Unit sphere representation.

    NOTE: Spheres are assumed to be unit spheres, i.e. centered at `(0, 0, 0)` with a radius of `1`.
    Sphere position, size, and rotation can be adjusted by changing the value of `transform`.

    NOTE: Spheres are compared by object ID only, so no 2 instances will compare `True`
    """

    def _local_intersect(self, transformed_ray: Ray) -> Intersections:
        # Calculate the discriminant to determine if there are any intersections
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

    def _local_normal_at(self, local_point: Rayple) -> Rayple:
        return local_point - point(0, 0, 0)
