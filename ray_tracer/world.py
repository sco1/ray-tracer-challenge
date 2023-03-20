from __future__ import annotations

from dataclasses import dataclass

from ray_tracer.colors import BLACK, WHITE
from ray_tracer.intersections import Comps, Intersections, prepare_computations
from ray_tracer.lights import PointLight, lighting
from ray_tracer.materials import Material
from ray_tracer.rayple import Rayple, color, point
from ray_tracer.rays import Ray
from ray_tracer.shapes import Shape, Sphere
from ray_tracer.transforms import scaling

DEFAULT_LIGHT = PointLight(point(-10, 10, -10), WHITE)


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

    def _shade_hit(self, comps: Comps) -> Rayple:
        """Calculate the color at the provided pre-computed intersection point in the world."""
        # Use comps.over_point to account for floating point issues around object surfaces; this
        # value bumps the query point slightly towards the normal so it's not accidentally
        # considered inside
        shadowed = self.is_shadowed(comps.over_point)
        return lighting(
            material=comps.obj.material,
            obj=comps.obj,
            light=self.light,
            surf_pos=comps.point,
            eye_v=comps.eye_v,
            normal=comps.normal,
            in_shadow=shadowed,
        )

    def color_at(self, r: Ray) -> Rayple:
        """
        Calculate the color at the `Ray`'s first intersection point in the world.

        NOTE: If the `Ray` has no intersection point(s), the returned color will be black.
        """
        hit = self.intersect_world(r).hit
        if not hit:
            return BLACK

        comps = prepare_computations(hit, r)
        return self._shade_hit(comps)

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

    @classmethod
    def default_world(cls) -> World:  # pragma: no cover
        """Create a basic world containing two concentric spheres of varying material properties."""
        s1 = Sphere(material=Material(color=color(0.8, 1.0, 0.6), diffuse=0.7, specular=0.2))
        s2 = Sphere(transform=scaling(0.5, 0.5, 0.5))

        return cls(light=DEFAULT_LIGHT, objects=[s1, s2])
