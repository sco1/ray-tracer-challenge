import typing as t
from pathlib import Path

from ray_tracer.canvas import Canvas
from ray_tracer.rayple import Rayple, color, point, vector


class Environment(t.NamedTuple):  # noqa: D101
    gravity: Rayple
    wind: Rayple


def cannon() -> None:
    # Initial conditions from the textbook
    pos = point(0, 1, 0)
    vel = vector(1, 1.8, 0).normalize() * 11.25
    env = Environment(vector(0, -0.1, 0), vector(-0.01, 0, 0))

    c = Canvas(900, 550)
    c.write_pixel(round(pos.x), c.height - round(pos.y), color(1, 0, 0))
    while pos.y > 0:
        pos += vel
        vel += env.gravity + env.wind

        y_loc = c.height - round(pos.y)
        if 0 <= y_loc <= c.height:
            c.write_pixel(round(pos.x), c.height - round(pos.y), color(1, 0, 0))

    c.to_ppm(Path("./chapter_2.ppm"))


if __name__ == "__main__":
    cannon()
