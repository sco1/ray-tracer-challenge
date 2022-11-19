from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from ray_tracer.canvas import Canvas, _build_ppm_header, _pixels_to_ppm
from ray_tracer.rayple import color, point


def test_canvas_create() -> None:
    c = Canvas(10, 20)

    assert c.width == 10
    assert c.height == 20
    assert c._pixels.shape == (10, 20, 3)


def test_write_pixel() -> None:
    c = Canvas(10, 20)
    red = color(1, 0, 0)

    c.write_pixel(2, 3, red)
    assert c._pixels[2, 3, :] == pytest.approx((1, 0, 0))


def test_get_pixel() -> None:
    c = Canvas(10, 20)
    c._pixels[2, 3, :] = (1, 0, 0)

    red = color(1, 0, 0)
    assert c.pixel_at(2, 3) == red


def test_non_color_write_raises() -> None:
    c = Canvas(10, 20)
    with pytest.raises(ValueError):
        c.write_pixel(2, 3, point(1, 2, 3))


def test_build_ppm_header() -> None:
    assert _build_ppm_header(5, 3) == "P3\n5 3\n255"
    assert _build_ppm_header(5, 3, "Hello", 20) == "Hello\n5 3\n20"


def test_array_to_ppm_no_wrap() -> None:
    a = np.zeros(shape=(5, 3, 3))
    a[0, 0, :] = (1.5, 0, 0)
    a[2, 1, :] = (0, 0.5, 0)
    a[4, 2, :] = (-0.5, 0, 1)

    truth = dedent(
        """\
        255 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 127 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 255"""
    )
    assert _pixels_to_ppm(a, maxlen=None) == truth


def test_array_to_ppm_wrapped() -> None:
    a = np.zeros(shape=(5, 3, 3))
    a[0, 0, :] = (1.5, 0, 0)
    a[2, 1, :] = (0, 0.5, 0)
    a[4, 2, :] = (-0.5, 0, 1)

    truth = dedent(
        """\
        255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 127 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 255"""
    )
    assert _pixels_to_ppm(a, maxlen=42) == truth


def test_ppm_write(tmp_path: Path) -> None:
    c = Canvas(5, 3)
    c.write_pixel(0, 0, color(1.5, 0, 0))
    c.write_pixel(2, 1, color(0, 0.5, 0))
    c.write_pixel(4, 2, color(-0.5, 0, 1))

    out_img = tmp_path / "my_img.ppm"
    c.to_ppm(out_img)

    # The trailing newline is intentional
    truth = dedent(
        """\
        P3
        5 3
        255
        255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 255
        """
    )
    assert out_img.read_text() == truth
