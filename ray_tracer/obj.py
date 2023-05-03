from pathlib import Path

import more_itertools as miter

from ray_tracer.rayple import Rayple, point, vector
from ray_tracer.shapes import Group, SmoothTriangle, Triangle


def parse_obj_src(
    src: str,
) -> tuple[list[Rayple], list[Triangle | SmoothTriangle], list[Group], list[Rayple]]:
    """
    Parse a Wavefront OBJ file into collections of geometric constructs.

    Currently output are:
        * Vertices
        * `Triangle`s
        * `Group`s
        * Vertex Normals

    OBJ formats are assumed to consist of statements, each of which occupies a single line. Each
    statement is prefaced with a command, followed by a space-delimited list of arguments.

    Supported commands are:
        * `f` - Face; These reference previously defined vertex values (1-indexed) & construct
        either triangles or convex polygons. Convex polygons are triangulated into their component
        triangles.

        Face values are expected as one of three forms: `"f 1 2 3"`, `"f 1/2/3 2/3/4 3/4/5"`, or
        `"f 1//3 2//4 3//5"`. For the latter two forms, the middle value specifies a texture vertex
        (ignored), and the rightmost value specifies a vertex normal.
        * `g` - Named group
        * `v` - Vertex
        * `vn` - Vertex normal vector; if present, are associated with the vertex of the same index

    All other lines are silently discarded.

    If Named Groups are present in the OBJ file, each named group is added as a child of the default
    group, and subsequent triangles are added to the most recently encountered `Group`. If no named
    group is present, all triangles are added to the default `Group`.

    NOTE: It is assumed that OBJ files and individual commands are well-formed, no validation is
    performed.
    """
    all_vertices = []
    all_triangles: list[Triangle | SmoothTriangle] = []
    groups = [Group()]
    all_vertex_normals = []
    for line in src.splitlines():
        if not line or line[0] not in {"v", "f", "g"}:
            continue

        if line.startswith("vn"):
            x, y, z = (float(val) for val in line.split()[1:])
            all_vertex_normals.append(vector(x, y, z))
        elif line.startswith("v"):
            x, y, z = (float(val) for val in line.split()[1:])
            all_vertices.append(point(x, y, z))
        elif line.startswith("f"):
            triangles: list[Triangle] | list[SmoothTriangle]
            if "/" in line:
                # Smooth triangles
                vertices = []
                normals = []
                for comp in line.split()[1:]:
                    # Since we're assuming a properly formatted OBJ file we can fall back to 0 for
                    # an unspecified texture vertex (we're ignoring it anyway)
                    vertex_idx, _, normal_idx = (
                        int(val) - 1 if val else 0 for val in comp.split("/")
                    )
                    vertices.append(all_vertices[vertex_idx])
                    normals.append(all_vertex_normals[normal_idx])

                triangles = _fan_triangulation_smooth(vertices, normals)
            else:
                # Regular boring triangles
                vertices = [all_vertices[int(val) - 1] for val in line.split()[1:]]
                triangles = _fan_triangulation(vertices)
            all_triangles.extend(triangles)

            for t in triangles:
                groups[-1].add_child(t)
        elif line.startswith("g"):  # pragma: no branch
            new_group = Group()
            groups[0].add_child(new_group)
            groups.append(new_group)

    return all_vertices, all_triangles, groups, all_vertex_normals


def _fan_triangulation(vertices: list[Rayple]) -> list[Triangle]:
    """Split the provided convex polygon into its component triangles."""
    triangles = []
    for p2, p3 in miter.sliding_window(vertices[1:], 2):
        triangles.append(Triangle(p1=vertices[0], p2=p2, p3=p3))

    return triangles


def _fan_triangulation_smooth(
    vertices: list[Rayple], vertex_normals: list[Rayple]
) -> list[SmoothTriangle]:
    """Split the provided convex polygon into its component smooth triangles."""
    triangles = []
    for (p2, p3), (n2, n3) in zip(
        miter.sliding_window(vertices[1:], 2), miter.sliding_window(vertex_normals[1:], 2)
    ):
        triangles.append(
            SmoothTriangle(p1=vertices[0], p2=p2, p3=p3, n1=vertex_normals[0], n2=n2, n3=n3)
        )

    return triangles


def parse_obj_file(filepath: Path) -> Group:
    """
    Parse a Wavefront OBJ file into a `Group` instance.

    OBJ formats are assumed to consist of statements, each of which occupies a single line. Each
    statement is prefaced with a command, followed by a space-delimited list of arguments.

    Supported commands are:
        * `f` - Face; these reference previously defined vertices (NOTE: Vertices are 1-indexed).
        Faces may either be triangles or convex polygons; the latter is triangulated into its
        component triangles.
        * `g` - Named group
        * `v` - Vertex
        * `vn` - Vertex normal vector; if present, are associated with the vertex of the same index

    All other lines are silently discarded.

    If Named Groups are present in the OBJ file, each named group is added as a child of the default
    group, and subsequent triangles are added to the most recently encountered `Group`. If no named
    group is present, all triangles are added to the default `Group`.

    NOTE: It is assumed that OBJ files and individual commands are well-formed, no validation is
    performed.
    """
    src = filepath.read_text()
    *_, groups, _ = parse_obj_src(src)

    return groups[0]
