from dataclasses import dataclass

from ray_tracer.materials import Material
from ray_tracer.rayple import Rayple, RaypleType, color, dot

BLACK = color(0, 0, 0)


@dataclass(frozen=True, slots=True)
class PointLight:  # noqa: D101
    position: Rayple
    intensity: Rayple


def lighting(
    material: Material,
    light: PointLight,
    surf_pos: Rayple,
    eye_vec: Rayple,
    normal: Rayple,
) -> Rayple:
    """
    Calculate the shading from the given light source at the given point on an object.

    Shading is calculated using the Phong reflection model, which simulates the interaction between
    three different types of lighting:
        * Ambient reflection - background lighting; treated as a constant coloring of all points on
        the surface equally.
        * Diffuse reflection - light reflected from a matte surface; depends only on the angle
        between the light source and the surface normal.
        * Specular reflection - reflection of the light source itself, resulting in specular
        lighting (a bright spot on a curved surface); depends on the angle between the reflection
        vector and the eye vector. This is controlled by the object's shininess: the higher the
        shininess, the smaller and tighter the specular highlight.
    """
    if surf_pos.w != RaypleType.POINT:
        raise ValueError("Surface position must be a point.")
    if eye_vec.w != RaypleType.VECTOR:
        raise ValueError("Eye vector must be a vector.")
    if normal.w != RaypleType.VECTOR:
        raise ValueError("Normal vector must be a vector.")

    effective_color = material.color * light.intensity
    ambient = effective_color * material.ambient

    light_vec = (light.position - surf_pos).normalize()
    light_dot_normal = dot(light_vec, normal)
    if light_dot_normal < 0:
        # A negative number means the light is on the other side of the surface, so the diffuse and
        # specular components go to 0
        diffuse = BLACK
        specular = BLACK
    else:
        diffuse = effective_color * material.diffuse * light_dot_normal

        # For the specular contribution, determine the angle between the reflection and eye vectors
        reflect_vec = -light_vec.reflect(normal)
        reflect_dot_eye = dot(reflect_vec, eye_vec)
        if reflect_dot_eye <= 0:
            # Light is reflecting away from the eye
            specular = BLACK
        else:
            factor = reflect_dot_eye**material.shininess
            specular = light.intensity * material.specular * factor

    return ambient + diffuse + specular
