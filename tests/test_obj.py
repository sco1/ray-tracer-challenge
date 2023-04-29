from pathlib import Path
from textwrap import dedent

from ray_tracer.obj import parse_obj_file, parse_obj_src
from ray_tracer.rayple import point


def test_skip_unsupported_commands() -> None:
    src = dedent(
        """\
        There was a young lady named Bright
        who traveled much faster than light.
        She set out one day
        in a relative way,
        and came back the previous night.
        """
    )

    vertices, tris, groups = parse_obj_src(src)
    assert len(vertices) == 0
    assert len(tris) == 0
    assert len(groups) == 1


def test_parse_vertex_records() -> None:
    src = dedent(
        """\
        v -1 1 0
        v -1.0000 0.5000 0.0000
        v 1 0 0
        v 1 1 0
        """
    )

    vertices, tris, groups = parse_obj_src(src)
    assert len(vertices) == 4
    assert len(tris) == 0
    assert len(groups) == 1

    assert vertices[0] == point(-1, 1, 0)
    assert vertices[1] == point(-1, 0.5, 0)
    assert vertices[2] == point(1, 0, 0)
    assert vertices[3] == point(1, 1, 0)


def test_parse_triangles() -> None:
    src = dedent(
        """\
        v -1 1 0
        v -1 0 0
        v 1 0 0
        v 1 1 0

        f 1 2 3
        f 1 3 4
        """
    )

    vertices, tris, groups = parse_obj_src(src)
    assert len(vertices) == 4
    assert len(tris) == 2
    assert len(groups) == 1

    # Face indices in the OBJ file are 1-indexed
    assert (tris[0].p1, tris[0].p2, tris[0].p3) == (vertices[0], vertices[1], vertices[2])
    assert (tris[1].p1, tris[1].p2, tris[1].p3) == (vertices[0], vertices[2], vertices[3])


def test_parse_triangulate_polygon() -> None:
    src = dedent(
        """\
        v -1 1 0
        v -1 0 0
        v 1 0 0
        v 1 1 0
        v 0 2 0

        f 1 2 3 4 5
        """
    )

    vertices, tris, groups = parse_obj_src(src)
    assert len(vertices) == 5
    assert len(tris) == 3
    assert len(groups) == 1

    # Face indices in the OBJ file are 1-indexed
    assert (tris[0].p1, tris[0].p2, tris[0].p3) == (vertices[0], vertices[1], vertices[2])
    assert (tris[1].p1, tris[1].p2, tris[1].p3) == (vertices[0], vertices[2], vertices[3])
    assert (tris[2].p1, tris[2].p2, tris[2].p3) == (vertices[0], vertices[3], vertices[4])


def test_default_group() -> None:
    src = dedent(
        """\
        v -1 1 0
        v -1 0 0
        v 1 0 0

        f 1 2 3
        """
    )

    _, tris, groups = parse_obj_src(src)
    assert len(groups) == 1
    assert len(groups[0].children) == 1

    assert tris[0].parent == groups[0]
    assert tris[0] in groups[0].children


def test_named_groups() -> None:
    src = dedent(
        """\
        v -1 1 0
        v -1 0 0
        v 1 0 0
        v 1 1 0

        g FirstGroup
        f 1 2 3
        g SecondGroup
        f 1 3 4
        """
    )

    _, tris, groups = parse_obj_src(src)
    assert len(tris) == 2
    assert len(groups) == 3

    assert len(groups[0].children) == 2
    assert groups[1] in groups[0].children
    assert groups[2] in groups[0].children

    assert len(groups[1].children) == 1
    assert tris[0].parent == groups[1]
    assert tris[0] in groups[1].children

    assert len(groups[2].children) == 1
    assert tris[1].parent == groups[2]
    assert tris[1] in groups[2].children


def test_obj_file_parse() -> None:
    sample_file = Path(__file__).parent / Path("triangles.obj")
    group = parse_obj_file(sample_file)

    assert len(group.children) == 2
