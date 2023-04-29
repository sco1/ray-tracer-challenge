from pathlib import Path

import more_itertools as miter

from ray_tracer.rayple import Rayple, point
from ray_tracer.shapes import Group, Triangle


def parse_obj_src(src: str) -> tuple[list[Rayple], list[Triangle], list[Group]]:
    """
    Parse a Wavefront OBJ file into collections of points, `Triangle`s, and `Groups`.

    OBJ formats are assumed to consist of statements, each of which occupies a single line. Each
    statement is prefaced with a command, followed by a space-delimited list of arguments.

    Supported commands are:
        * `v` - Vertex
        * `f` - Face; these reference previously defined vertices (NOTE: Vertices are 1-indexed).
        Faces may either be triangles or convex polygons; the latter is triangulated into its
        component triangles.
        * `g` - Named group

    All other lines are silently discarded.

    If Named Groups are present in the OBJ file, each named group is added as a child of the default
    group, and subsequent triangles are added to the most recently encountered `Group`. If no named
    group is present, all triangles are added to the default `Group`.

    NOTE: It is assumed that OBJ files and individual commands are well-formed, no validation is
    performed.
    """
    all_vertices = []
    all_triangles = []
    groups = [Group()]
    for line in src.splitlines():
        if not line or line[0] not in {"v", "f", "g"}:
            continue

        if line.startswith("v"):
            x, y, z = (float(val) for val in line.split()[1:])
            all_vertices.append(point(x, y, z))
        elif line.startswith("f"):
            vertices = [all_vertices[int(val) - 1] for val in line.split()[1:]]
            triangles = _fan_triangulation(vertices)
            all_triangles.extend(triangles)

            for t in triangles:
                groups[-1].add_child(t)
        elif line.startswith("g"):  # pragma: no branch
            new_group = Group()
            groups[0].add_child(new_group)
            groups.append(new_group)

    return all_vertices, all_triangles, groups


def _fan_triangulation(vertices: list[Rayple]) -> list[Triangle]:
    """Split the provided convex polygon into its component triangles."""
    triangles = []
    for p2, p3 in miter.sliding_window(vertices[1:], 2):
        triangles.append(Triangle(p1=vertices[0], p2=p2, p3=p3))

    return triangles


def parse_obj_file(filepath: Path) -> Group:
    """
    Parse a Wavefront OBJ file into a `Group` instance.

    OBJ formats are assumed to consist of statements, each of which occupies a single line. Each
    statement is prefaced with a command, followed by a space-delimited list of arguments.

    Supported commands are:
        * `v` - Vertex
        * `f` - Face; these reference previously defined vertices (NOTE: Vertices are 1-indexed).
        Faces may either be triangles or convex polygons; the latter is triangulated into its
        component triangles.
        * `g` - Named group

    All other lines are silently discarded.

    If Named Groups are present in the OBJ file, each named group is added as a child of the default
    group, and subsequent triangles are added to the most recently encountered `Group`. If no named
    group is present, all triangles are added to the default `Group`.

    NOTE: It is assumed that OBJ files and individual commands are well-formed, no validation is
    performed.
    """
    src = filepath.read_text()
    *_, groups = parse_obj_src(src)

    return groups[0]
