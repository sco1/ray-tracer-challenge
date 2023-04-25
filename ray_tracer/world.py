from __future__ import annotations

import math
from dataclasses import dataclass

from ray_tracer.colors import BLACK, WHITE
from ray_tracer.intersections import Comps, Intersections, prepare_computations
from ray_tracer.lights import PointLight, lighting
from ray_tracer.materials import Material
from ray_tracer.rayple import Rayple, color, dot, point
from ray_tracer.rays import Ray
from ray_tracer.shapes import Shape, Sphere
from ray_tracer.transforms import scaling

DEFAULT_LIGHT = PointLight(point(-10, 10, -10), WHITE)

REF_LIMIT = 5


@dataclass(slots=True)
class World:  # noqa: D101
    light: PointLight
    objects: list[Shape]

    def intersect_world(self, ray: Ray) -> Intersections:
        """
        Calculate the `Ray`'s intersections with all objects in the current world.

        NOTE: Intersections are aggregated & sorted by their `t` values.
        """
        all_intersections = Intersections([])
        for obj in self.objects:
            all_intersections.extend(obj.intersect(ray))

        all_intersections.sort()
        return all_intersections

    def _shade_hit(self, comps: Comps, remaining: int = REF_LIMIT) -> Rayple:
        """
        Calculate the color at the provided pre-computed intersection point in the world.

        The `remaining` arg is included to prevent infinite reflections from blowing up the stack.
        This parameter is decremented in the `reflected_color` method.
        """
        # Use comps.over_point to account for floating point issues around object surfaces; this
        # value bumps the query point slightly towards the normal so it's not accidentally
        # considered inside
        shadowed = self.is_shadowed(comps.over_point)
        surface = lighting(
            material=comps.obj.material,
            obj=comps.obj,
            light=self.light,
            surf_pos=comps.point,
            eye_v=comps.eye_v,
            normal=comps.normal,
            in_shadow=shadowed,
        )
        reflected = self.reflected_color(comps, remaining=remaining)
        refracted = self.refracted_color(comps, remaining=remaining)
        return surface + reflected + refracted

    def color_at(self, r: Ray, remaining: int = REF_LIMIT) -> Rayple:
        """
        Calculate the color at the `Ray`'s first intersection point in the world.

        The `remaining` arg is included to prevent infinite reflections from blowing up the stack.
        This parameter is decremented in the `reflected_color` method.

        NOTE: If the `Ray` has no intersection point(s), the returned color will be black.
        """
        hit = self.intersect_world(r).hit
        if not hit:
            return BLACK

        comps = prepare_computations(hit, r)
        return self._shade_hit(comps, remaining=remaining)

    def is_shadowed(self, pt: Rayple) -> bool:
        """Determine if the query point is shadowed by a world object."""
        # Cast a ray from the point towards the light source & see if it hits anything along the way
        pt_v = self.light.position - pt
        pt_dist = abs(pt_v)
        pt_dir = pt_v.normalize()
        r = Ray(pt, pt_dir)

        intersections = self.intersect_world(r)
        h = intersections.hit
        if h and h.t < pt_dist:  # Make sure hit isn't past the light source
            return True
        else:
            return False

    def reflected_color(self, comps: Comps, remaining: int = REF_LIMIT) -> Rayple:
        """
        Determine the reflected color for the provided precomputed intersection.

        The `remaining` arg is included to prevent infinite reflections from blowing up the stack by
        returning Black if the reflection limit has been reached.
        """
        if comps.obj.material.reflective == 0:
            return BLACK
        if remaining <= 0:
            return BLACK

        # Use over_point to help prevent rays from originating just below the surface, causing them
        # to intersect the surface they're supposed to be reflecting from
        reflect_ray = Ray(comps.over_point, comps.reflect_v)
        col = self.color_at(reflect_ray, remaining=remaining - 1)
        return col * comps.obj.material.reflective

    def refracted_color(self, comps: Comps, remaining: int = REF_LIMIT) -> Rayple:
        """
        Determine the reflected color for the provided precomputed intersection.

        The `remaining` arg is included to prevent infinite reflections from blowing up the stack by
        returning Black if the reflection limit has been reached.
        """
        if comps.obj.material.transparency == 0:
            return BLACK
        if remaining <= 0:
            return BLACK

        # Check for total internal reflection
        # If light enters a material with a sufficiently acute angle, and the new medium has a lower
        # refractive index than the old, then the light will reflect off the interface rather than
        # pass through it
        n_ratio = comps.n1 / comps.n2
        cos_i = dot(comps.eye_v, comps.normal)
        sin2_t = n_ratio**2 * (1 - cos_i**2)

        if sin2_t > 1:
            return BLACK

        cos_t = math.sqrt(1.0 - sin2_t)
        direction = comps.normal * (n_ratio * cos_i - cos_t) - comps.eye_v * n_ratio
        refract_ray = Ray(comps.under_point, direction)

        color = self.color_at(refract_ray, remaining - 1) * comps.obj.material.transparency
        return color

    @classmethod
    def default_world(cls) -> World:  # pragma: no cover
        """Create a basic world containing two concentric spheres of varying material properties."""
        s1 = Sphere(material=Material(color=color(0.8, 1.0, 0.6), diffuse=0.7, specular=0.2))
        s2 = Sphere(transform=scaling(0.5, 0.5, 0.5))

        return cls(light=DEFAULT_LIGHT, objects=[s1, s2])
