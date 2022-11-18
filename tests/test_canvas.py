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
