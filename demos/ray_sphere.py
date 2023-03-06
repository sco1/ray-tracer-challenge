from pathlib import Path

from ray_tracer.canvas import Canvas
from ray_tracer.rayple import color, point
from ray_tracer.rays import Ray
from ray_tracer.shapes import Sphere


def ray_sphere() -> None:
    canvas_size = 100
    c = Canvas(canvas_size, canvas_size)

    wall_z = 10
    wall_size = 7
    half_wall = wall_size / 2
    pixel_size = wall_size / canvas_size

    ray_origin = point(0, 0, -5)
    ray_color = color(0, 1, 0)
    s = Sphere()

    for y in range(canvas_size + 1):
        world_y = half_wall - pixel_size * y
        for x in range(canvas_size + 1):
            world_x = -half_wall + pixel_size * x
            wall_coord = point(world_x, world_y, wall_z)

            ray_to_wall = wall_coord - ray_origin  # vector
            r = Ray(ray_origin, ray_to_wall.normalize())

            intersections = s.intersect(r)
            if intersections:
                c.write_pixel(x, y, ray_color)

    c.to_ppm(Path("./demos/out/chapter_5.ppm"))


if __name__ == "__main__":
    ray_sphere()
