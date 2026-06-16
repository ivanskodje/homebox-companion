"""Unit tests for crop_image_by_normalized_bbox."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from homebox_companion.ai.images import _otsu_threshold, crop_image_by_normalized_bbox

pytestmark = pytest.mark.unit


def _make_quadrant_image(width: int, height: int) -> bytes:
    """Build an image with a distinct solid color in each quadrant."""
    img = Image.new("RGB", (width, height), (0, 0, 0))
    half_w, half_h = width // 2, height // 2
    img.paste((255, 0, 0), (0, 0, half_w, half_h))
    img.paste((0, 255, 0), (half_w, 0, width, half_h))
    img.paste((0, 0, 255), (0, half_h, half_w, height))
    img.paste((255, 255, 255), (half_w, half_h, width, height))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=95)
    return out.getvalue()


def _dimensions(jpeg_bytes: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(jpeg_bytes)).size


def _center_pixel(jpeg_bytes: bytes) -> tuple[int, ...]:
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    pixel = img.getpixel((img.width // 2, img.height // 2))
    assert isinstance(pixel, tuple)
    return pixel


def _is_close(actual: tuple[int, ...], expected: tuple[int, int, int], tol: int = 70) -> bool:
    return all(abs(a - e) <= tol for a, e in zip(actual, expected, strict=True))


class TestCropImageByNormalizedBbox:
    """crop_image_by_normalized_bbox should crop, pad, clamp, and never raise."""

    def test_crops_requested_quadrant(self) -> None:
        """Cropping the top-right quadrant yields a green, half-by-half image."""
        image = _make_quadrant_image(1000, 800)

        crop, mime = crop_image_by_normalized_bbox(
            image, x=0.5, y=0.0, width=0.5, height=0.5, padding=0.0, refine=False
        )

        assert mime == "image/jpeg"
        assert _dimensions(crop) == (500, 400)
        assert _is_close(_center_pixel(crop), (0, 255, 0))

    def test_padding_expands_and_clamps_at_edges(self) -> None:
        """A box at the origin expands by padding on the inner sides and clamps at 0."""
        image = _make_quadrant_image(1000, 800)

        crop, _ = crop_image_by_normalized_bbox(
            image, x=0.0, y=0.0, width=0.25, height=0.25, padding=0.1, refine=False
        )

        assert _dimensions(crop) == (350, 280)

    def test_degenerate_box_returns_full_image(self) -> None:
        """A zero-area box falls back to the full image instead of raising."""
        image = _make_quadrant_image(1000, 800)

        crop, mime = crop_image_by_normalized_bbox(
            image, x=0.5, y=0.5, width=0.0, height=0.0, padding=0.0, refine=False
        )

        assert mime == "image/jpeg"
        assert _dimensions(crop) == (1000, 800)

    def test_resizes_crop_to_max_dimension(self) -> None:
        """A large crop is downscaled to max_dimension."""
        image = _make_quadrant_image(2000, 2000)

        crop, _ = crop_image_by_normalized_bbox(
            image, x=0.0, y=0.0, width=1.0, height=1.0, padding=0.0, max_dimension=512, refine=False
        )

        assert max(_dimensions(crop)) == 512

    def test_exif_orientation_normalized_before_crop(self) -> None:
        """The box maps to the upright image, so a rotated EXIF source crops to upright dimensions."""
        stored = Image.new("RGB", (800, 400), (10, 20, 30))
        exif = stored.getexif()
        exif[274] = 6  # Orientation: rotate on view, upright is 400x800
        out = io.BytesIO()
        stored.save(out, format="JPEG", exif=exif, quality=95)

        crop, _ = crop_image_by_normalized_bbox(
            out.getvalue(), x=0.0, y=0.0, width=1.0, height=1.0, padding=0.0, refine=False
        )

        assert _dimensions(crop) == (400, 800)


def _make_object_in_corner(
    width: int,
    height: int,
    background: tuple[int, int, int],
    obj_color: tuple[int, int, int],
    obj_box: tuple[int, int, int, int],
) -> bytes:
    """Build a uniform-background image with a solid block pasted in."""
    img = Image.new("RGB", (width, height), background)
    img.paste(obj_color, obj_box)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=95)
    return out.getvalue()


class TestCropRefinement:
    """Refinement tightens a loose box to the object, or falls back, but never enlarges."""

    def test_tightens_loose_box_high_contrast(self) -> None:
        """A small dark object in a loose light box should crop down to the object."""
        image = _make_object_in_corner(1000, 800, (200, 200, 200), (20, 20, 20), (100, 80, 360, 320))

        loose, _ = crop_image_by_normalized_bbox(image, x=0.0, y=0.0, width=0.9, height=0.9, padding=0.0, refine=False)
        refined, _ = crop_image_by_normalized_bbox(image, x=0.0, y=0.0, width=0.9, height=0.9, padding=0.0, refine=True)

        loose_w, loose_h = _dimensions(loose)
        refined_w, refined_h = _dimensions(refined)
        assert refined_w * refined_h < 0.5 * loose_w * loose_h
        assert _is_close(_center_pixel(refined), (20, 20, 20))

    def test_low_contrast_tightens_or_falls_back(self) -> None:
        """A dark object on a dark background must never produce a larger crop than the loose box."""
        image = _make_object_in_corner(1000, 800, (40, 35, 30), (70, 62, 55), (100, 80, 360, 320))

        loose, _ = crop_image_by_normalized_bbox(image, x=0.0, y=0.0, width=0.9, height=0.9, padding=0.0, refine=False)
        refined, _ = crop_image_by_normalized_bbox(image, x=0.0, y=0.0, width=0.9, height=0.9, padding=0.0, refine=True)

        loose_w, loose_h = _dimensions(loose)
        refined_w, refined_h = _dimensions(refined)
        assert refined_w <= loose_w and refined_h <= loose_h

    def test_uniform_region_falls_back(self) -> None:
        """A featureless region has no foreground, so the crop equals the un-refined box."""
        image = _make_object_in_corner(1000, 800, (128, 128, 128), (128, 128, 128), (0, 0, 1, 1))

        loose, _ = crop_image_by_normalized_bbox(image, x=0.1, y=0.1, width=0.6, height=0.6, padding=0.0, refine=False)
        refined, _ = crop_image_by_normalized_bbox(image, x=0.1, y=0.1, width=0.6, height=0.6, padding=0.0, refine=True)

        assert _dimensions(refined) == _dimensions(loose)

    def test_busy_region_never_enlarges(self) -> None:
        """Even on a cluttered/noisy region the refined crop is never larger than the loose box."""
        noise = Image.effect_noise((800, 640), 90).convert("RGB")
        buffer = io.BytesIO()
        noise.save(buffer, format="JPEG", quality=90)
        image = buffer.getvalue()

        loose, _ = crop_image_by_normalized_bbox(
            image, x=0.05, y=0.05, width=0.9, height=0.9, padding=0.0, refine=False
        )
        refined, _ = crop_image_by_normalized_bbox(
            image, x=0.05, y=0.05, width=0.9, height=0.9, padding=0.0, refine=True
        )

        loose_w, loose_h = _dimensions(loose)
        refined_w, refined_h = _dimensions(refined)
        assert refined_w <= loose_w and refined_h <= loose_h

    def test_low_contrast_colored_body_captured_whole(self) -> None:
        """A colored body with a bright band is captured whole, not clipped to the bright band."""
        img = Image.new("RGB", (600, 800), (45, 30, 18))
        img.paste((60, 80, 45), (60, 40, 540, 760))
        img.paste((235, 235, 235), (90, 70, 510, 200))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        image = buffer.getvalue()

        refined, _ = crop_image_by_normalized_bbox(
            image, x=0.05, y=0.03, width=0.9, height=0.94, padding=0.0, refine=True
        )

        rw, rh = _dimensions(refined)
        assert rh >= 400
        assert _is_close(_center_pixel(refined), (60, 80, 45))

    def test_isolates_centered_item_ignoring_corner_distractor(self) -> None:
        """A high-contrast distractor in a corner must not capture the crop; the centered item wins."""
        img = Image.new("RGB", (1000, 800), (90, 85, 80))
        img.paste((30, 28, 26), (380, 280, 620, 520))
        img.paste((255, 255, 255), (860, 40, 970, 150))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        image = buffer.getvalue()

        refined, _ = crop_image_by_normalized_bbox(image, x=0.0, y=0.0, width=1.0, height=1.0, padding=0.0, refine=True)

        rw, rh = _dimensions(refined)
        assert rw < 600 and rh < 600
        assert _is_close(_center_pixel(refined), (30, 28, 26))


class TestOtsuThreshold:
    """_otsu_threshold separates a bimodal histogram and handles degenerate input."""

    def test_lands_between_clusters(self) -> None:
        histogram = [0] * 256
        for value in range(20, 41):
            histogram[value] = 200
        for value in range(210, 231):
            histogram[value] = 50
        threshold = _otsu_threshold(histogram)
        assert 40 <= threshold < 210

    def test_empty_histogram_returns_midpoint(self) -> None:
        assert _otsu_threshold([0] * 256) == 128

    def test_single_cluster_does_not_split_it(self) -> None:
        histogram = [0] * 256
        histogram[128] = 1000
        assert _otsu_threshold(histogram) <= 128
