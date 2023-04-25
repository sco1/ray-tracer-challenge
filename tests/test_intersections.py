import math
from functools import partial

import pytest

from ray_tracer import EPSILON
from ray_tracer.intersections import Comps, Intersection, Intersections, prepare_computations
from ray_tracer.materials import Material
from ray_tracer.rayple import point, vector
from ray_tracer.rays import Ray
from ray_tracer.shapes import Plane, Sphere
from ray_tracer.transforms import scaling, translation


def test_intersections_container() -> None:
    s = Sphere()
    intersections = Intersections([Intersection(1, s), Intersection(2, s)])

    assert len(intersections) == 2
    assert intersections[0].t == 1
    assert intersections[1].t == 2


def test_intersections_sorted() -> None:
    s = Sphere()
    intersections = Intersections([Intersection(2, s), Intersection(1, s)])

    assert intersections[0].t == 1


P_INT = partial(Intersection, obj=Sphere())
HIT_TEST_CASES = (
    (Intersections([P_INT(1), P_INT(2)]), P_INT(1)),
    (Intersections([P_INT(-1), P_INT(1)]), P_INT(1)),
    (Intersections([P_INT(-2), P_INT(-1)]), None),
    (Intersections([P_INT(5), P_INT(7), P_INT(-3), P_INT(2)]), P_INT(2)),
)


@pytest.mark.parametrize(("intersections", "truth_hit"), HIT_TEST_CASES)
def test_hit(intersections: Intersections, truth_hit: Intersection) -> None:
    assert intersections.hit == truth_hit


BASE_SHAPE = Sphere()
COMPUTATIONS_CASES = (
    (
        Ray(point(0, 0, -5), vector(0, 0, 1)),
        Intersection(4, BASE_SHAPE),
        Comps(
            t=4,
            obj=BASE_SHAPE,
            point=point(0, 0, -1),
            eye_v=vector(0, 0, -1),
            normal=vector(0, 0, -1),
            inside=False,
            n1=1,
            n2=1,
            reflect_v=vector(0, 0, -1),
        ),
    ),
    (
        Ray(point(0, 0, 0), vector(0, 0, 1)),
        Intersection(1, BASE_SHAPE),
        Comps(
            t=1,
            obj=BASE_SHAPE,
            point=point(0, 0, 1),
            eye_v=vector(0, 0, -1),
            normal=vector(0, 0, -1),
            inside=True,
            n1=1,
            n2=1,
            reflect_v=vector(0, 0, -1),
        ),
    ),
)


@pytest.mark.parametrize(("r", "inter", "truth_comp"), COMPUTATIONS_CASES)
def test_prepare_computations(r: Ray, inter: Intersection, truth_comp: Comps) -> None:
    comps = prepare_computations(inter, r)

    assert comps == truth_comp


def test_prepare_computations_over_point() -> None:
    r = Ray(point(0, 0, -5), vector(0, 0, 1))
    shape = Sphere(transform=translation(0, 0, 1))
    i = Intersection(5, shape)

    comps = prepare_computations(i, r)
    assert comps.over_point.z < -EPSILON / 2  # Ensure correct direction
    assert comps.point.z > comps.over_point.z


def test_prepare_computations_under_point() -> None:
    r = Ray(point(0, 0, -5), vector(0, 0, 1))
    shape = Sphere(transform=translation(0, 0, 1))
    i = Intersection(5, shape)

    comps = prepare_computations(i, r)
    assert comps.under_point.z > -EPSILON / 2  # Ensure correct direction
    assert comps.point.z < comps.under_point.z


RT_2 = math.sqrt(2)


def test_reflection_vector() -> None:
    shape = Plane()
    r = Ray(point(0, 1, -1), vector(0, -RT_2 / 2, RT_2 / 2))
    i = Intersection(RT_2, shape)

    comps = prepare_computations(i, r)
    assert comps.reflect_v == vector(0, RT_2 / 2, RT_2 / 2)


@pytest.fixture
def refraction_scenario() -> Intersections:
    # 3 glass spheres: B & C overlap slightly and contained by A
    a = Sphere(scaling(2, 2, 2), Material(transparency=1, refractive_index=1.5))
    b = Sphere(translation(0, 0, -0.25), Material(transparency=1, refractive_index=2.0))
    c = Sphere(translation(0, 0, 0.25), Material(transparency=1, refractive_index=2.5))

    xs = Intersections(
        [
            Intersection(2, a),
            Intersection(2.75, b),
            Intersection(3.25, c),
            Intersection(4.75, b),
            Intersection(5.25, c),
            Intersection(6, a),
        ]
    )

    return xs


REFRACTION_CASES = (
    (0, 1.0, 1.5),
    (1, 1.5, 2.0),
    (2, 2.0, 2.5),
    (3, 2.5, 2.5),
    (4, 2.5, 1.5),
    (5, 1.5, 1.0),
)


@pytest.mark.parametrize(("idx", "n1", "n2"), REFRACTION_CASES)
def test_refraction_indices(
    idx: int, n1: float, n2: float, refraction_scenario: Intersections
) -> None:
    r = Ray(point(0, 0, -4), vector(0, 0, 1))
    comps = prepare_computations(
        inter=refraction_scenario[idx], ray=r, all_inters=refraction_scenario
    )

    assert comps.n1 == pytest.approx(n1)
    assert comps.n2 == pytest.approx(n2)
