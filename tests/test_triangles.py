import pytest

from ray_tracer.rayple import point, vector
from ray_tracer.rays import Ray
from ray_tracer.shapes import Triangle


def test_triangle_compute_on_init() -> None:
    t = Triangle(p1=point(0, 1, 0), p2=point(-1, 0, 0), p3=point(1, 0, 0))

    assert t.e1 == vector(-1, -1, 0)
    assert t.e2 == vector(1, -1, 0)
    assert t.norm == vector(0, 0, -1)


def test_triangle_normal() -> None:
    t = Triangle(p1=point(0, 1, 0), p2=point(-1, 0, 0), p3=point(1, 0, 0))

    n1 = t._local_normal_at(point(0, 0.5, 0))
    n2 = t._local_normal_at(point(-0.5, 0.75, 0))
    n3 = t._local_normal_at(point(0.5, 0.25, 0))

    assert n1 == t.norm
    assert n2 == t.norm
    assert n3 == t.norm


def test_triangle_parallel_ray() -> None:
    t = Triangle(p1=point(0, 1, 0), p2=point(-1, 0, 0), p3=point(1, 0, 0))
    r = Ray(point(0, -1, -2), vector(0, 1, 0))

    inters = t._local_intersect(r)
    assert len(inters) == 0


def test_triangle_ray_p1_p3_miss() -> None:
    t = Triangle(p1=point(0, 1, 0), p2=point(-1, 0, 0), p3=point(1, 0, 0))
    r = Ray(point(1, 1, -2), vector(0, 0, 1))

    inters = t._local_intersect(r)
    assert len(inters) == 0


def test_triangle_ray_p1_p2_miss() -> None:
    t = Triangle(p1=point(0, 1, 0), p2=point(-1, 0, 0), p3=point(1, 0, 0))
    r = Ray(point(-1, 1, -2), vector(0, 0, 1))

    inters = t._local_intersect(r)
    assert len(inters) == 0


def test_triangle_ray_p2_p3_miss() -> None:
    t = Triangle(p1=point(0, 1, 0), p2=point(-1, 0, 0), p3=point(1, 0, 0))
    r = Ray(point(0, -1, -2), vector(0, 0, 1))

    inters = t._local_intersect(r)
    assert len(inters) == 0


def test_triangle_ray_hit() -> None:
    t = Triangle(p1=point(0, 1, 0), p2=point(-1, 0, 0), p3=point(1, 0, 0))
    r = Ray(point(0, 0.5, -2), vector(0, 0, 1))

    inters = t._local_intersect(r)
    assert len(inters) == 1
    assert inters[0].t == pytest.approx(2)
