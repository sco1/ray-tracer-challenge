import textwrap
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
        """Return the Color value for the queried pixel."""
        return color(*self._pixels[x, y, :])

    def write_pixel(self, x: int, y: int, color: Rayple) -> None:
        """Map the provided Color value to the specified pixel location."""
        if color.w != RaypleType.COLOR:
            raise ValueError(f"Expected Color Rayple. Received: {type(color.w)}")

        self._pixels[x, y, :] = (*color,)

    def to_ppm(self, out_filepath: Path) -> None:
        """
        Output the current canvas as a Portable Pixmap (PPM).

        See: https://en.wikipedia.org/wiki/Netpbm for more info on the file format.
        """
        full_text = (
            f"{_build_ppm_header(self.width, self.height)}\n{_pixels_to_ppm(self._pixels)}\n"
        )
        out_filepath.write_text(full_text)


def _build_ppm_header(width: int, height: int, identifier: str = "P3", maxval: int = 255) -> str:
    """
    Build a Portable Pixmap (PPM) header from the provided parameters.

    The PPM header is assumed to contain 3 lines:
        1. Identifier, or "magic" number
        2. Image width and height
        3. Maximum color value
    """
    header = f"{identifier}\n{width} {height}\n{maxval}"
    return header


def _pixels_to_ppm(pixels: np.ndarray, maxval: int = 255, maxlen: int | None = 70) -> str:
    """
    Convert the provided pixel array to PPM format.

    The pixel array is assumed to be a numeric `NxMx3` array representing pixel RGB values. Color
    values are assumed to be between `0` and `1`, inclusive, and are scaled to integers by `maxval`.
    Any scaled values less than `0` or greater than `maxval` are clamped. Pixel values are then
    unwrapped row-wise into an (N*3)xM array of pixel values.

    If `maxlen` is not `None`, an attempt is made to limit each data row to a maximum length of
    `maxlen` characters.
    """
    # Scale to maxval and clamp
    scaled = (pixels * maxval).astype(int)
    scaled[scaled > maxval] = maxval
    scaled[scaled < 0] = 0

    # Now unwrap into the pixel rows
    _, height, *_ = pixels.shape
    scaled = scaled.reshape([height, -1])

    # There's probably a way to get numpy to do what we want but this is fine
    # Temporarily remove numpy's print setting since we're manually delimiting
    with np.printoptions(linewidth=np.inf):  # type: ignore[arg-type]
        tmp = np.array2string(scaled)
        tmp = "\n".join(" ".join(row.strip("[] ").split()) for row in tmp.splitlines())

    if maxlen is not None:
        return textwrap.fill(tmp, width=maxlen)
    else:
        return tmp
