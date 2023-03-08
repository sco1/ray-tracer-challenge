from pathlib import Path

from ray_tracer.canvas import Canvas
from ray_tracer.lights import PointLight, lighting
from ray_tracer.materials import Material
from ray_tracer.rayple import color, point
from ray_tracer.rays import Ray
from ray_tracer.shapes import Sphere


def shaded_ray_sphere() -> None:
    canvas_size = 100
    c = Canvas(canvas_size, canvas_size)

    wall_z = 10
    wall_size = 7
    half_wall = wall_size / 2
    pixel_size = wall_size / canvas_size

    ray_origin = point(0, 0, -5)
    s = Sphere(material=Material(color=color(1, 0.2, 1)))

    light_position = point(-10, 10, -10)
    light_color = color(1, 1, 1)
    light = PointLight(light_position, light_color)

    for y in range(canvas_size + 1):
        world_y = half_wall - pixel_size * y
        for x in range(canvas_size + 1):
            world_x = -half_wall + pixel_size * x
            wall_coord = point(world_x, world_y, wall_z)

            ray_to_wall = wall_coord - ray_origin  # vector
            r = Ray(ray_origin, ray_to_wall.normalize())
            eye = -r.direction

            intersections = s.intersect(r)
            if intersections:
                if hit := intersections.hit:
                    hit = intersections.hit
                    surface_point = r.position(hit.t)
                    normal = hit.obj.normal_at(surface_point)
                    lit_color = lighting(hit.obj.material, light, surface_point, eye, normal)

                c.write_pixel(x, y, lit_color)

    c.to_ppm(Path("./demos/out/chapter_6.ppm"))


if __name__ == "__main__":
    shaded_ray_sphere()
