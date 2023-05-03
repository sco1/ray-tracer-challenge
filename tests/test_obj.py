from pathlib import Path
from textwrap import dedent

from ray_tracer.obj import parse_obj_file, parse_obj_src
from ray_tracer.rayple import point, vector
from ray_tracer.shapes import SmoothTriangle


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

    vertices, tris, groups, normals = parse_obj_src(src)
    assert len(vertices) == 0
    assert len(tris) == 0
    assert len(groups) == 1
    assert len(normals) == 0


def test_parse_vertex_records() -> None:
    src = dedent(
        """\
        v -1 1 0
        v -1.0000 0.5000 0.0000
        v 1 0 0
        v 1 1 0
        """
    )

    vertices, tris, groups, normals = parse_obj_src(src)
    assert len(vertices) == 4
    assert len(tris) == 0
    assert len(groups) == 1
    assert len(normals) == 0

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

    vertices, tris, groups, normals = parse_obj_src(src)
    assert len(vertices) == 4
    assert len(tris) == 2
    assert len(groups) == 1
    assert len(normals) == 0

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

    vertices, tris, groups, normals = parse_obj_src(src)
    assert len(vertices) == 5
    assert len(tris) == 3
    assert len(groups) == 1
    assert len(normals) == 0

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

    _, tris, groups, _ = parse_obj_src(src)
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

    _, tris, groups, _ = parse_obj_src(src)
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


def test_vertex_normals() -> None:
    src = dedent(
        """\
        vn 0 0 1
        vn 0.707 0 -0.707
        vn 1 2 3
        """
    )

    *_, normals = parse_obj_src(src)
    assert len(normals) == 3
    assert normals[0] == vector(0, 0, 1)
    assert normals[1] == vector(0.707, 0, -0.707)
    assert normals[2] == vector(1, 2, 3)


def test_faces_with_normals() -> None:
    src = dedent(
        """\
        v 0 1 0
        v -1 0 0
        v 1 0 0

        vn -1 0 0
        vn 1 0 0
        vn 0 1 0

        f 1//3 2//1 3//2
        f 1/0/3 2/102/1 3/14/2
        """
    )

    verts, tris, _, norms = parse_obj_src(src)
    assert len(tris) == 2
    assert all(isinstance(tri, SmoothTriangle) for tri in tris)

    assert (tris[0].p1, tris[0].p2, tris[0].p3) == (verts[0], verts[1], verts[2])
    assert (tris[0].n1, tris[0].n2, tris[0].n3) == (norms[2], norms[0], norms[1])

    # Both triangles should be equivalent, but we don't use an __eq__ method
    assert (tris[0].p1, tris[0].p2, tris[0].p3) == (tris[1].p1, tris[1].p2, tris[1].p3)
    assert (tris[0].n1, tris[0].n2, tris[0].n3) == (tris[1].n1, tris[1].n2, tris[1].n3)
