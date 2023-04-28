import math

from ray_tracer.rayple import point, vector
from ray_tracer.rays import Ray
from ray_tracer.shapes import Group, Sphere
from ray_tracer.transforms import rot, scaling, translation


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


def test_group_array_transform() -> None:
    g = Group(transform=scaling(2, 2, 2))
    s = Sphere(transform=translation(5, 0, 0))
    g.add_child(s)

    r = Ray(point(10, 0, -10), vector(0, 0, 1))
    inters = g.intersect(r)
    assert len(inters) == 2


def test_world_to_object_space() -> None:
    g1 = Group(transform=rot(y=math.pi / 2))
    g2 = Group(transform=scaling(2, 2, 2))
    g1.add_child(g2)

    s = Sphere(transform=translation(5, 0, 0))
    g2.add_child(s)

    p = s.world_to_object(point(-2, 0, -10))
    assert p == point(0, 0, -1)


RT_3 = math.sqrt(3)


def test_normal_to_world() -> None:
    g1 = Group(transform=rot(y=math.pi / 2))
    g2 = Group(transform=scaling(1, 2, 3))
    g1.add_child(g2)

    s = Sphere(transform=translation(5, 0, 0))
    g2.add_child(s)

    n = s.normal_to_world(vector(RT_3 / 3, RT_3 / 3, RT_3 / 3))
    assert n == vector(0.28571, 0.42857, -0.85714)


def test_child_normal_at() -> None:
    g1 = Group(transform=rot(y=math.pi / 2))
    g2 = Group(transform=scaling(1, 2, 3))
    g1.add_child(g2)

    s = Sphere(transform=translation(5, 0, 0))
    g2.add_child(s)

    norm = s.normal_at(point(1.7321, 1.1547, -5.5774))
    assert norm == vector(0.28570, 0.42854, -0.85716)
