from ray_tracer.intersections import Intersection
from ray_tracer.rayple import point, vector
from ray_tracer.shapes import SmoothTriangle


def test_interpolated_normal() -> None:
    st = SmoothTriangle(
        p1=point(0, 1, 0),
        p2=point(-1, 0, 0),
        p3=point(1, 0, 0),
        n1=vector(0, 1, 0),
        n2=vector(-1, 0, 0),
        n3=vector(1, 0, 0),
    )
    inter = Intersection(t=1, obj=st, u=0.45, v=0.25)

    norm = st.normal_at(point(0, 0, 0), inter)
    assert norm == vector(-0.5547, 0.83205, 0)
