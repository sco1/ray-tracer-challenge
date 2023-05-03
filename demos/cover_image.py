import math
from functools import partial
from pathlib import Path

from ray_tracer.camera import Camera
from ray_tracer.colors import BLUE, PURPLE, RED, WHITE
from ray_tracer.lights import PointLight
from ray_tracer.materials import Material
from ray_tracer.rayple import color, point, vector
from ray_tracer.shapes import Cube, Plane, Sphere
from ray_tracer.transforms import rot, scaling, translation, view_transform
from ray_tracer.world import World

MAT_P = partial(Material, diffuse=0.7, ambient=0.1, specular=0.0, reflective=0.1)
WHITE_MAT = MAT_P(color=WHITE)
BLUE_MAT = MAT_P(color=BLUE)
RED_MAT = MAT_P(color=RED)
PURPLE_MAT = MAT_P(color=PURPLE)

STD_OBJ = translation(1, -1, 1) * scaling(0.5, 0.5, 0.5)
LARGE_OBJ = STD_OBJ * scaling(3.5, 3.5, 3.5)
MED_OBJ = STD_OBJ * scaling(3, 3, 3)
SMALL_OBJ = STD_OBJ * scaling(2, 2, 2)


def render_cover() -> None:
    cam = Camera(
        h_size=800,
        v_size=800,
        fov=0.785,
        transform=view_transform(
            from_p=point(-6, 6, -10), to_p=point(6, 0, 6), up_v=vector(-0.45, 1, 0)
        ),
    )

    main_light = PointLight(position=point(50, 100, -50), intensity=WHITE)
    # second_light = PointLight(position=point(-400, 50, -10), intensity=color(0.2, 0.2, 0.2))

    backdrop = Plane(
        material=Material(color=color(1, 1, 1), ambient=1, diffuse=0, specular=0),
        transform=(rot(x=math.pi / 2) * translation(0, 0, 500)),
    )

    s = Sphere(
        material=Material(
            color=color(0.373, 0.404, 0.550),
            diffuse=0.2,
            ambient=0.0,
            specular=1.0,
            shininess=200,
            reflective=0.7,
            transparency=0.7,
            refractive_index=1.5,
        ),
        transform=LARGE_OBJ,
    )

    all_shapes = [backdrop, s]
    cube_specs = (
        (WHITE_MAT, MED_OBJ * translation(4, 0, 0)),
        (BLUE_MAT, LARGE_OBJ * translation(8.5, 1.5, -0.5)),
        (RED_MAT, LARGE_OBJ * translation(0, 0, 4)),
        (WHITE_MAT, SMALL_OBJ * translation(4, 0, 4)),
        (PURPLE_MAT, MED_OBJ * translation(7.5, 0.5, 4)),
        (WHITE_MAT, MED_OBJ * translation(-0.25, 0.25, 8)),
        (BLUE_MAT, LARGE_OBJ * translation(4, 1, 7.5)),
        (RED_MAT, MED_OBJ * translation(10, 2, 7.5)),
        (WHITE_MAT, SMALL_OBJ * translation(8, 2, 12)),
        (WHITE_MAT, SMALL_OBJ * translation(20, 1, 9)),
        (BLUE_MAT, LARGE_OBJ * translation(-0.5, -5, 0.25)),
        (RED_MAT, LARGE_OBJ * translation(4, -4, 0)),
        (WHITE_MAT, LARGE_OBJ * translation(8.5, -4, 0)),
        (WHITE_MAT, LARGE_OBJ * translation(0, -4, 4)),
        (PURPLE_MAT, LARGE_OBJ * translation(-0.5, -4.5, 8)),
        (WHITE_MAT, LARGE_OBJ * translation(0, -8, 4)),
        (WHITE_MAT, LARGE_OBJ * translation(-0.5, -8.5, 8)),
    )
    for mat, trans in cube_specs:
        all_shapes.append(Cube(material=mat, transform=trans))

    w = World(main_light, all_shapes)
    canvas = cam.render(world=w)
    canvas.to_ppm(Path("./demos/out/book_cover.ppm"))


if __name__ == "__main__":
    render_cover()
