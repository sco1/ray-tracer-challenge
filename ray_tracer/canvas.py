from pathlib import Path

import numpy as np

from ray_tracer.rayple import Rayple, RaypleType, color


class Canvas:  # noqa: D101
    _pixels: np.ndarray

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

        self._pixels = np.zeros(shape=(width, height, 3))

    def pixel_at(self, x: int, y: int) -> Rayple:
        return color(*self._pixels[x, y, :])

    def write_pixel(self, x: int, y: int, color: Rayple) -> None:
        if color.w != RaypleType.COLOR:
            raise ValueError(f"Expected Color Rayple. Received: {type(color.w)}")

        self._pixels[x, y, :] = (*color,)

    ...