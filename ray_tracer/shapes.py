import math
from dataclasses import dataclass, field

from ray_tracer import EPSILON, NUMERIC_T
from ray_tracer.intersections import Intersection, Intersections
from ray_tracer.materials import Material
from ray_tracer.rayple import Rayple, RaypleType, dot, point, vector
from ray_tracer.rays import Ray
from ray_tracer.transforms import Matrix


@dataclass(frozen=True, slots=True, eq=False)
class Shape:
    """
    Base class for creating shape objects; this is not intended to be instantiated.

    Child classes must define `_local_intersect` and `_local_normal_at` to calculate their
    respective local values, which are then transformed into world coordinates by the base methods.

    NOTE: Shapes are compared by object ID only, so no 2 instances will compare `True`.
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
class Sphere(Shape):
    """
    Unit sphere representation.

    Spheres are assumed to be unit spheres, i.e. centered at `(0, 0, 0)` with a radius of `1`.
    Sphere position, size, and rotation can be adjusted by passing in a non-identity `transform`
    matrix.

    NOTE: For simplicity, the ntersection between a sphere and a tangent ray will result in two
    identical intersections.
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


@dataclass(frozen=True, slots=True, eq=False)
class Plane(Shape):
    """
    Plane representation.

    Planes are assumed to be infinite and oriented in the XZ plane. Plane position and rotation can
    be adjusted by passing in a non-identity `transform` matrix.

    NOTE: Rays that are coplanar with a given plane are treated as missing the plane rather than
    trying to consider an infinite number of intersections.
    """

    def _local_intersect(self, transformed_ray: Ray) -> Intersections:
        # If the slope is zero then we're parallel/coplanar
        if abs(transformed_ray.direction.y) < EPSILON:
            return Intersections([])

        t = -transformed_ray.origin.y / transformed_ray.direction.y
        return Intersections([Intersection(t, self)])

    def _local_normal_at(self, local_point: Rayple) -> Rayple:
        # The normal of a plane is constant everywhere
        return vector(0, 1, 0)


@dataclass(frozen=True, slots=True, eq=False)
class Cube(Shape):
    """
    Cube Representation.

    Cubes are modeled as axis-aligned bounding boxes; i.e. its sides are all aligned with the
    scene's axes. Cubes will begin centered at the origin and extend from `-1` to `+1` along each
    axis; transformation matrices can be used to manipulate them further within the scene.
    """

    def _local_intersect(self, transformed_ray: Ray) -> Intersections:
        # Treat the cube as being composed of six planes, one for each face. If the ray intersects
        # these planes in just the right way, it means it intersects the cube as well. Use a helper
        # function to consider these planes in parallel planes; if the cube is intersected then
        # there will be 4 points of intersection, and the intersection with the cube itself will be
        # the largest of the two closest points and the smallest of the two largest points.
        xt_min, xt_max = _check_axis(transformed_ray.origin.x, transformed_ray.direction.x)
        yt_min, yt_max = _check_axis(transformed_ray.origin.y, transformed_ray.direction.y)
        zt_min, zt_max = _check_axis(transformed_ray.origin.z, transformed_ray.direction.z)

        t_min = max(xt_min, yt_min, zt_min)
        t_max = min(xt_max, yt_max, zt_max)

        if t_min > t_max:
            return Intersections([])
        else:
            return Intersections([Intersection(t_min, self), Intersection(t_max, self)])

    def _local_normal_at(self, local_point: Rayple) -> Rayple:
        # We know which plane we're on because it has the component with the largest absolute value
        # In the ideal case it will be the component that equals 1, but we have floats so we can't
        # get by that easy
        # Use a custom max loop so we can enforce that corners are on either the +x or -x faces
        for i, comp in enumerate(local_point):
            if i == 0:
                max_c = abs(comp)
                idx = 0
                continue

            if abs(comp) > max_c:
                max_c = abs(comp)
                idx = i

        print(local_point, max_c, idx)

        if idx == 0:
            return vector(local_point.x, 0, 0)
        elif idx == 1:
            return vector(0, local_point.y, 0)
        else:
            return vector(0, 0, local_point.z)


def _check_axis(origin: NUMERIC_T, direction: NUMERIC_T) -> tuple[NUMERIC_T, NUMERIC_T]:
    """
    Locate the plane-ray intersection times, if present.

    Each pair of parallel lines will have a minimum t closest to the ray origin, and a maximum t
    farther away; we want to consider the largest minimum t value and the smallest maximum t value
    as the plane's intersection points.
    """
    t_min_numerator = -1 - origin
    t_max_numerator = 1 - origin

    if abs(direction) >= EPSILON:
        t_min = t_min_numerator / direction
        t_max = t_max_numerator / direction
    else:
        # If the denominator is 0, multiply by infinity rather than dividing by 0 so we retain the
        # correct sign
        t_min = t_min_numerator * math.inf
        t_max = t_max_numerator * math.inf

    if t_min > t_max:
        t_min, t_max = t_max, t_min

    return t_min, t_max
