"""Image encoding and optimization utilities for AI/LLM APIs."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from loguru import logger
from PIL import Image, ImageChops, ImageFilter

# Default settings for image optimization (for AI vision)
DEFAULT_MAX_DIMENSION = 2048  # Most vision models work best with max 2048px images
DEFAULT_JPEG_QUALITY = 85

# PIL format to MIME type mapping
_FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
}

# Crop-refinement tunables (Pillow-only foreground tightening); heuristic, may need
# tuning on real photos.
_REFINE_MASK_MAX_SIDE = 448
_REFINE_BORDER_FRACTION = 0.08
_REFINE_EDGE_THRESHOLD = 32
_REFINE_DIFF_CAP = 110
_REFINE_OPEN_MIN_SIZE = 3
_REFINE_CONNECT_SIZE = 7
_REFINE_MIN_FG_FRACTION = 0.005
_REFINE_MAX_FG_FRACTION = 0.85
_REFINE_MIN_BBOX_FRACTION = 0.02
_REFINE_BREATHING_ROOM = 0.08


def _detect_mime_type(image_bytes: bytes) -> str:
    """Detect the MIME type of an image from its bytes.

    Args:
        image_bytes: Raw image data.

    Returns:
        MIME type string (e.g., "image/jpeg", "image/png").
        Falls back to "image/jpeg" if detection fails.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        fmt = img.format
        if fmt and fmt.upper() in _FORMAT_TO_MIME:
            return _FORMAT_TO_MIME[fmt.upper()]
    except Exception:
        pass
    return "image/jpeg"  # Safe fallback for unknown formats


def _normalize_image(img: Image.Image) -> Image.Image:
    """Normalize image: handle EXIF orientation and convert to RGB.

    This is a shared helper for both optimize_image_for_vision() and
    compress_image_for_upload() to avoid code duplication.

    Args:
        img: PIL Image object to normalize.

    Returns:
        Normalized PIL Image in RGB mode with correct orientation.
    """
    # Handle EXIF orientation
    try:
        from PIL import ExifTags

        orientation: int | None = None
        for tag_id in ExifTags.TAGS:
            if ExifTags.TAGS[tag_id] == "Orientation":
                orientation = tag_id
                break

        exif = img.getexif()
        if exif is not None and orientation is not None:
            orientation_value = exif.get(orientation)
            if orientation_value == 3:
                img = img.rotate(180, expand=True)
            elif orientation_value == 6:
                img = img.rotate(270, expand=True)
            elif orientation_value == 8:
                img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, TypeError):
        # No EXIF data or no orientation tag
        pass

    # Convert to RGB if necessary (handles RGBA, P mode, etc.)
    if img.mode in ("RGBA", "P", "LA"):
        # Create white background for transparent images
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    return img


def optimize_image_for_vision(
    image_bytes: bytes,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> tuple[bytes, str]:
    """Optimize an image for LLM vision processing.

    Resizes large images and compresses to JPEG for faster uploads and
    reduced token costs. Vision models work best with images
    up to 2048px on the longest side.

    Args:
        image_bytes: Raw image data.
        max_dimension: Maximum width or height in pixels.
        quality: JPEG compression quality (1-100).

    Returns:
        Tuple of (optimized_bytes, mime_type).
    """
    original_size = len(image_bytes)

    try:
        img = Image.open(io.BytesIO(image_bytes))
        original_dimensions = img.size

        # Normalize image (EXIF orientation + RGB conversion)
        img = _normalize_image(img)

        # Resize if larger than max dimension
        needs_resize = max(img.size) > max_dimension
        if needs_resize:
            img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            logger.debug(f"Resized image from {original_dimensions} to {img.size}")

        # Compress to JPEG
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=quality, optimize=True)
        optimized_bytes = output.getvalue()

        optimized_size = len(optimized_bytes)
        savings = ((original_size - optimized_size) / original_size) * 100

        if savings > 5:  # Only log if meaningful savings
            logger.debug(f"Image optimized: {original_size:,} -> {optimized_size:,} bytes ({savings:.1f}% reduction)")

        return optimized_bytes, "image/jpeg"

    except Exception as e:
        logger.warning(f"Image optimization failed, using original: {e}")
        # Return original bytes with their actual MIME type, not hardcoded JPEG
        return image_bytes, _detect_mime_type(image_bytes)


def encode_image_to_data_uri(image_path: Path | str, optimize: bool = True) -> str:
    """Read an image file and return a data URI for vision model APIs.

    Args:
        image_path: Path to the image file.
        optimize: Whether to optimize the image for vision processing.

    Returns:
        A data URI string (e.g., "data:image/jpeg;base64,...").
    """
    path = Path(image_path)
    image_bytes = path.read_bytes()

    if optimize:
        image_bytes, mime_type = optimize_image_for_vision(image_bytes)
    else:
        mime_type = _detect_mime_type(image_bytes)

    # Extract suffix from mime_type (e.g., "image/jpeg" -> "jpeg")
    suffix = mime_type.split("/")[-1] if "/" in mime_type else "jpeg"
    payload = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/{suffix};base64,{payload}"


def encode_image_bytes_to_data_uri(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    optimize: bool = True,
) -> str:
    """Encode raw image bytes to a data URI for vision model APIs.

    Args:
        image_bytes: Raw image data.
        mime_type: MIME type of the image.
        optimize: Whether to optimize the image for vision processing.

    Returns:
        A data URI string.
    """
    if optimize:
        image_bytes, mime_type = optimize_image_for_vision(image_bytes)

    # Extract suffix from mime_type (e.g., "image/jpeg" -> "jpeg")
    suffix = mime_type.split("/")[-1] if "/" in mime_type else "jpeg"
    payload = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/{suffix};base64,{payload}"


def compress_image_for_upload(
    image_bytes: bytes,
    max_dimension: int | None = None,
    quality: int = 75,
) -> tuple[bytes, str]:
    """Compress an image for Homebox upload based on quality settings.

    This function is separate from optimize_image_for_vision() to allow
    different compression strategies for AI (which needs good quality)
    vs Homebox storage (where users may prefer smaller files).

    Args:
        image_bytes: Raw image data.
        max_dimension: Maximum width or height in pixels. None = no resizing.
        quality: JPEG compression quality (1-100).

    Returns:
        Tuple of (compressed_bytes, mime_type).
    """
    # If raw quality requested (no max_dimension), return original with correct MIME type
    if max_dimension is None:
        return image_bytes, _detect_mime_type(image_bytes)

    original_size = len(image_bytes)

    try:
        img = Image.open(io.BytesIO(image_bytes))
        original_dimensions = img.size

        # Normalize image (EXIF orientation + RGB conversion)
        img = _normalize_image(img)

        # Resize if larger than max dimension
        needs_resize = max(img.size) > max_dimension
        if needs_resize:
            img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            logger.debug(f"Resized image for upload from {original_dimensions} to {img.size}")

        # Compress to JPEG
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=quality, optimize=True)
        compressed_bytes = output.getvalue()

        compressed_size = len(compressed_bytes)
        savings = ((original_size - compressed_size) / original_size) * 100

        if savings > 5:  # Only log if meaningful savings
            logger.debug(
                f"Image compressed for upload: {original_size:,} -> {compressed_size:,} bytes "
                f"({savings:.1f}% reduction)"
            )

        return compressed_bytes, "image/jpeg"

    except Exception as e:
        logger.warning(f"Image compression failed, using original: {e}")
        # Return original bytes with their actual MIME type, not hardcoded JPEG
        return image_bytes, _detect_mime_type(image_bytes)


def encode_compressed_image_to_base64(
    image_bytes: bytes,
    max_dimension: int | None = None,
    quality: int = 75,
) -> tuple[str, str]:
    """Compress an image and encode to base64 for API response.

    Args:
        image_bytes: Raw image data.
        max_dimension: Maximum width or height in pixels. None = no resizing.
        quality: JPEG compression quality (1-100).

    Returns:
        Tuple of (base64_string, mime_type).
    """
    compressed_bytes, mime_type = compress_image_for_upload(image_bytes, max_dimension, quality)
    base64_str = base64.b64encode(compressed_bytes).decode("ascii")
    return base64_str, mime_type


def _otsu_threshold(histogram: list[int]) -> int:
    """Otsu's threshold for a 256-bin grayscale histogram (pure Python; numpy unavailable).

    Returns the 0-255 value that maximizes between-class variance, which for a bimodal
    histogram lands in the valley between the two clusters. Returns 128 for an empty histogram.
    """
    total = sum(histogram)
    if total == 0:
        return 128
    sum_all = sum(value * count for value, count in enumerate(histogram))
    sum_background = 0.0
    weight_background = 0
    best_variance = -1.0
    threshold = 0
    for value in range(256):
        weight_background += histogram[value]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += value * histogram[value]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_all - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > best_variance:
            best_variance = variance
            threshold = value
    return threshold


def _estimate_border_background(region: Image.Image) -> tuple[int, int, int]:
    """Estimate the background color as the per-channel median of the region's border ring.

    The border of a loose bounding box is mostly background (the surface the item rests on),
    so a robust median gives a good background estimate even when the item is off-center.
    """
    width, height = region.size
    strip = max(1, int(round(min(width, height) * _REFINE_BORDER_FRACTION)))
    border_boxes = (
        (0, 0, width, strip),
        (0, height - strip, width, height),
        (0, 0, strip, height),
        (width - strip, 0, width, height),
    )
    medians: list[int] = []
    for channel in region.split():
        histogram = [0] * 256
        for box in border_boxes:
            for value, count in enumerate(channel.crop(box).histogram()):
                histogram[value] += count
        half = sum(histogram) / 2
        running = 0
        median = 0
        for value in range(256):
            running += histogram[value]
            if running >= half:
                median = value
                break
        medians.append(median)
    return medians[0], medians[1], medians[2]


def _select_item_component(
    mask: Image.Image, center_x: int, center_y: int
) -> tuple[tuple[int, int, int, int], int] | None:
    """Pick the foreground blob that corresponds to the item the model boxed, or None.

    The model centers its box on the item, so the item is the component spanning the box
    center, or failing that the one nearest the center. Either rule beats a higher-contrast
    distractor in a corner (e.g. a bag), which "largest blob" would wrongly pick. Returns the
    largest center-spanning component, else the nearest non-trivial component, else None. Pure
    Python on the downscaled mask; foreground is 255.
    """
    width, height = mask.size
    flat = list(mask.tobytes())
    visited = bytearray(width * height)
    min_area = _REFINE_MIN_FG_FRACTION * width * height
    spanning: tuple[int, tuple[int, int, int, int]] | None = None
    nearest: tuple[float, int, tuple[int, int, int, int]] | None = None
    for start in range(width * height):
        if flat[start] == 0 or visited[start]:
            continue
        stack = [start]
        visited[start] = 1
        area = 0
        sum_x = 0
        sum_y = 0
        min_x = max_x = start % width
        min_y = max_y = start // width
        while stack:
            index = stack.pop()
            area += 1
            cx = index % width
            cy = index // width
            sum_x += cx
            sum_y += cy
            if cx < min_x:
                min_x = cx
            elif cx > max_x:
                max_x = cx
            if cy < min_y:
                min_y = cy
            elif cy > max_y:
                max_y = cy
            if cx > 0 and flat[index - 1] and not visited[index - 1]:
                visited[index - 1] = 1
                stack.append(index - 1)
            if cx < width - 1 and flat[index + 1] and not visited[index + 1]:
                visited[index + 1] = 1
                stack.append(index + 1)
            if cy > 0 and flat[index - width] and not visited[index - width]:
                visited[index - width] = 1
                stack.append(index - width)
            if cy < height - 1 and flat[index + width] and not visited[index + width]:
                visited[index + width] = 1
                stack.append(index + width)
        bbox = (min_x, min_y, max_x + 1, max_y + 1)
        if min_x <= center_x <= max_x and min_y <= center_y <= max_y:
            if spanning is None or area > spanning[0]:
                spanning = (area, bbox)
        if area >= min_area:
            distance = (sum_x / area - center_x) ** 2 + (sum_y / area - center_y) ** 2
            if nearest is None or distance < nearest[0]:
                nearest = (distance, area, bbox)
    if spanning is not None:
        return spanning[1], spanning[0]
    if nearest is not None:
        return nearest[2], nearest[1]
    return None


def _refine_bbox_to_foreground(region: Image.Image) -> tuple[int, int, int, int] | None:
    """Find a tight bbox around the dominant foreground object inside a region, or None.

    Pillow-only. Combines a color-difference signal (region vs. a solid background image,
    Otsu-thresholded) with an edge-magnitude signal (FIND_EDGES, fixed threshold) so that
    low-contrast items are still found, applies a morphological opening to remove speckle,
    then takes getbbox(). Returns None (the caller keeps the looser box) when the result is
    untrustworthy: no foreground, too little (object not found / too low contrast), too much
    (background estimate failed or the region is cluttered), or an implausibly thin box.

    Returned coordinates are full-region pixels; the internal downscale is undone first.
    Morphological kernels must be odd (a Pillow requirement) and MaxFilter >= MinFilter so the
    opening nets out to consolidating the object after speckle removal.
    """
    full_width, full_height = region.size
    if full_width < 4 or full_height < 4:
        return None

    longest = max(full_width, full_height)
    if longest > _REFINE_MASK_MAX_SIDE:
        scale = _REFINE_MASK_MAX_SIDE / longest
        new_size = (max(1, round(full_width * scale)), max(1, round(full_height * scale)))
        work = region.resize(new_size, Image.Resampling.BILINEAR)
    else:
        scale = 1.0
        work = region

    background = _estimate_border_background(work)
    channel_diff = ImageChops.difference(work, Image.new("RGB", work.size, background))
    red, green, blue = channel_diff.split()
    difference = ImageChops.add(ImageChops.add(red, green), blue).point(lambda p: min(p, _REFINE_DIFF_CAP))
    difference_threshold = _otsu_threshold(difference.histogram())
    difference_mask = difference.point(lambda p: 255 if p > difference_threshold else 0)

    edges = work.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_mask = edges.point(lambda p: 255 if p > _REFINE_EDGE_THRESHOLD else 0)

    mask = ImageChops.lighter(difference_mask, edge_mask)
    mask = mask.filter(ImageFilter.MinFilter(_REFINE_OPEN_MIN_SIZE))
    mask = mask.filter(ImageFilter.MaxFilter(_REFINE_CONNECT_SIZE))

    work_width, work_height = work.size
    component = _select_item_component(mask, work_width // 2, work_height // 2)
    if component is None:
        return None
    bbox, area = component

    foreground_fraction = area / float(work_width * work_height)
    if foreground_fraction < _REFINE_MIN_FG_FRACTION or foreground_fraction > _REFINE_MAX_FG_FRACTION:
        return None

    left, top, right, bottom = bbox
    min_width = _REFINE_MIN_BBOX_FRACTION * work_width
    min_height = _REFINE_MIN_BBOX_FRACTION * work_height
    if (right - left) < min_width or (bottom - top) < min_height:
        return None

    if scale != 1.0:
        inverse = 1.0 / scale
        left = int(left * inverse)
        top = int(top * inverse)
        right = min(full_width, int(round(right * inverse)))
        bottom = min(full_height, int(round(bottom * inverse)))
    return left, top, right, bottom


def _refine_crop_box(
    img: Image.Image,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> tuple[int, int, int, int]:
    """Tighten a crop box to the object inside it, returning the box unchanged on any failure.

    Refinement only ever shrinks the box toward the object (plus breathing room) or leaves it
    as-is, so the resulting crop is never looser than the model's box. Never raises.
    """
    try:
        tight = _refine_bbox_to_foreground(img.crop((left, top, right, bottom)))
        if tight is None:
            return left, top, right, bottom

        t_left, t_top, t_right, t_bottom = tight
        pad_x = int(round((t_right - t_left) * _REFINE_BREATHING_ROOM))
        pad_y = int(round((t_bottom - t_top) * _REFINE_BREATHING_ROOM))

        new_left = max(left, min(left + t_left - pad_x, right))
        new_top = max(top, min(top + t_top - pad_y, bottom))
        new_right = max(left, min(left + t_right + pad_x, right))
        new_bottom = max(top, min(top + t_bottom + pad_y, bottom))

        if new_right - new_left < 1 or new_bottom - new_top < 1:
            return left, top, right, bottom
        return new_left, new_top, new_right, new_bottom
    except Exception as e:
        logger.debug(f"Crop refinement failed, using model box: {e}")
        return left, top, right, bottom


def crop_image_by_normalized_bbox(
    image_bytes: bytes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    padding: float = 0.08,
    max_dimension: int | None = None,
    quality: int = 75,
    refine: bool = True,
) -> tuple[bytes, str]:
    """Crop the region described by a normalized bounding box and compress it.

    The box (x, y, width, height) is expressed as fractions of the EXIF-normalized
    image, matching the coordinate space the vision model sees. The box is padded and
    clamped to the image bounds. When ``refine`` is set, the crop is then tightened to the
    actual object inside the box via foreground detection (vision-model boxes are often far
    looser than the item), falling back to the padded box when the result is uncertain. The
    crop is compressed with the same parameters as Homebox uploads. On a degenerate box or any
    failure the full image is compressed instead, so the caller always receives a usable image.

    Args:
        image_bytes: Raw image data.
        x: Left edge of the box as a fraction of image width.
        y: Top edge of the box as a fraction of image height.
        width: Box width as a fraction of image width.
        height: Box height as a fraction of image height.
        padding: Extra margin added on every side, as a fraction of each axis.
        max_dimension: Maximum width or height in pixels. None = no resizing.
        quality: JPEG compression quality (1-100).
        refine: If True, tighten the crop to the detected object within the box.

    Returns:
        Tuple of (image_bytes, mime_type).
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = _normalize_image(img)
        img_width, img_height = img.size

        left = max(0, min(int(round((x - padding) * img_width)), img_width))
        top = max(0, min(int(round((y - padding) * img_height)), img_height))
        right = max(0, min(int(round((x + width + padding) * img_width)), img_width))
        bottom = max(0, min(int(round((y + height + padding) * img_height)), img_height))

        if right - left < 1 or bottom - top < 1:
            logger.warning("Bounding box is degenerate after clamping; using full image")
            return compress_image_for_upload(image_bytes, max_dimension, quality)

        if refine:
            left, top, right, bottom = _refine_crop_box(img, left, top, right, bottom)

        output = io.BytesIO()
        img.crop((left, top, right, bottom)).save(output, format="JPEG", quality=quality, optimize=True)
        return compress_image_for_upload(output.getvalue(), max_dimension, quality)
    except Exception as e:
        logger.warning(f"Image crop failed, using full image: {e}")
        return compress_image_for_upload(image_bytes, max_dimension, quality)
