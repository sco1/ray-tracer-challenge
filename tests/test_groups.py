from ray_tracer.rayple import Rayple, point, vector
from ray_tracer.rays import Ray
from ray_tracer.shapes import Group, Sphere
from ray_tracer.transforms import translation


def test_adding_child() -> None:
    s = Sphere()
    g = Group()

    g.add_child(s)

    assert s in g.children
    assert s.parent == g


def test_empty_group_ray_intersect() -> None:
    g = Group()
    r = Ray(point(0, 0, 0), vector(0, 0, 1))

    inters = g._local_intersect(r)
    assert len(inters) == 0


def test_filled_group_ray_intersect() -> None:
    g = Group()
    s1 = Sphere()
    s2 = Sphere(transform=translation(0, 0, -3))
    s3 = Sphere(transform=translation(5, 0, 0))

    for s in (s1, s2, s3):
        g.add_child(s)

    r = Ray(point(0, 0, -5), vector(0, 0, 1))
    inters = g._local_intersect(r)

    assert len(inters) == 4
    assert inters[0].obj == s2
    assert inters[1].obj == s2
    assert inters[2].obj == s1
    assert inters[3].obj == s1
