from dataclasses import dataclass

from ray_tracer.materials import Material
from ray_tracer.rayple import Rayple, RaypleType, color, dot

BLACK = color(0, 0, 0)


@dataclass(frozen=True, slots=True)
class PointLight:
    position: Rayple
    intensity: Rayple


def lighting(
    material: Material,
    light: PointLight,
    surf_pos: Rayple,
    eye_vec: Rayple,
    normal: Rayple,
) -> Rayple:
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

        reflect_vec = -light_vec.reflect(normal)
        reflect_dot_eye = dot(reflect_vec, eye_vec)
        if reflect_dot_eye <= 0:
            specular = BLACK
        else:
            factor = reflect_dot_eye**material.shininess
            specular = light.intensity * material.specular * factor

    return ambient + diffuse + specular
