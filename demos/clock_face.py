from math import pi
from pathlib import Path

from ray_tracer.canvas import Canvas
from ray_tracer.rayple import color, point
from ray_tracer.transforms import rot_z, scaling, translation


def clock_face() -> None:
    canvas_size = 80
    border = 10
    c = Canvas(canvas_size, canvas_size)

    clock_rad = (canvas_size - (2 * border)) // 2
    trans = translation(canvas_size // 2, canvas_size // 2, 0)
    scale = scaling(clock_rad, clock_rad, 0)

    start_p = point(0, 1, 0)
    hours = [start_p]
    for i in range(1, 12):
        hour_rot = rot_z(i * (-pi / 6))
        hours.append(hour_rot * hours[0])

    for hour in hours:
        tmp = trans * scale * hour
        c.write_pixel(round(tmp.x), round(tmp.y), color(0, 1, 0))

    c.to_ppm(Path("./demos/out/chapter_4.ppm"))


if __name__ == "__main__":
    clock_face()
