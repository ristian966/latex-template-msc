import base64
import io
import threading

from flask import Flask, jsonify, render_template, request
import numpy as np
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
from rembg import new_session, remove


app = Flask(__name__)

_session = None
_session_lock = threading.Lock()


def _find_exterior_background(background: np.ndarray) -> np.ndarray:
    """Mark all background pixels connected to the image border."""
    h, w = background.shape
    reachable = np.zeros((h, w), dtype=bool)
    stack = []

    for x in range(w):
        if background[0, x]:
            stack.append((0, x))
        if background[h - 1, x]:
            stack.append((h - 1, x))
    for y in range(h):
        if background[y, 0]:
            stack.append((y, 0))
        if background[y, w - 1]:
            stack.append((y, w - 1))

    while stack:
        y, x = stack.pop()
        if y < 0 or y >= h or x < 0 or x >= w:
            continue
        if reachable[y, x] or not background[y, x]:
            continue
        reachable[y, x] = True
        stack.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))

    return reachable


def count_enclosed_hole_pixels(alpha_channel: np.ndarray, threshold: int = 18) -> int:
    foreground = alpha_channel > threshold
    background = ~foreground
    if not np.any(background):
        return 0

    reachable = _find_exterior_background(background)
    holes = background & ~reachable
    return int(holes.sum())


def refine_alpha_mask_legacy(alpha_channel: np.ndarray) -> np.ndarray:
    alpha_img = Image.fromarray(alpha_channel, mode="L")
    alpha_img = alpha_img.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    alpha = np.array(alpha_img, dtype=np.uint8)

    foreground = alpha > 10
    background = ~foreground
    if not np.any(background):
        return alpha

    h, _ = alpha.shape
    reachable = _find_exterior_background(background)
    holes = background & ~reachable
    if not np.any(holes):
        return alpha

    visited = np.zeros((h, alpha.shape[1]), dtype=bool)
    max_hole_area = max(200, int(alpha.size * 0.045))
    min_hole_area = 60

    ys, xs = np.where(holes)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue

        component_pixels = []
        local_stack = [(start_y, start_x)]
        while local_stack:
            y, x = local_stack.pop()
            if y < 0 or y >= h or x < 0 or x >= alpha.shape[1]:
                continue
            if visited[y, x] or not holes[y, x]:
                continue
            visited[y, x] = True
            component_pixels.append((y, x))
            local_stack.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))

        area = len(component_pixels)
        if area < min_hole_area or area > max_hole_area:
            continue

        rows = [p[0] for p in component_pixels]
        if (sum(rows) / area) > h * 0.82:
            continue

        for y, x in component_pixels:
            alpha[y, x] = max(alpha[y, x], 235)

    return alpha


def refine_alpha_mask_modern(alpha_channel: np.ndarray) -> np.ndarray:
    alpha_img = Image.fromarray(alpha_channel, mode="L")
    smoothed_alpha = np.array(alpha_img.filter(ImageFilter.GaussianBlur(0.55)), dtype=np.uint8)

    # Close narrow cut-through gaps so transparent windows are treated as enclosed regions.
    alpha_img = Image.fromarray(smoothed_alpha, mode="L")
    alpha_img = alpha_img.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(7))
    alpha = np.array(alpha_img, dtype=np.uint8)

    foreground = alpha > 6
    background = ~foreground
    if not np.any(background):
        return smoothed_alpha

    reachable = _find_exterior_background(background)
    h, w = alpha.shape
    holes = background & ~reachable
    if not np.any(holes):
        return smoothed_alpha

    visited = np.zeros((h, w), dtype=bool)
    max_hole_area = max(200, int(alpha.size * 0.22))
    min_hole_area = 20

    ys, xs = np.where(holes)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue

        component_pixels = []
        local_stack = [(start_y, start_x)]
        while local_stack:
            y, x = local_stack.pop()
            if y < 0 or y >= h or x < 0 or x >= w:
                continue
            if visited[y, x] or not holes[y, x]:
                continue
            visited[y, x] = True
            component_pixels.append((y, x))
            local_stack.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))

        area = len(component_pixels)
        if area < min_hole_area or area > max_hole_area:
            continue

        for y, x in component_pixels:
            smoothed_alpha[y, x] = max(smoothed_alpha[y, x], 242)

    # Keep interior body details opaque while preserving anti-aliased edges.
    interior = Image.fromarray((foreground.astype(np.uint8) * 255), mode="L")
    interior = interior.filter(ImageFilter.MinFilter(9))
    interior_mask = np.array(interior, dtype=np.uint8) > 127
    smoothed_alpha[interior_mask] = np.maximum(smoothed_alpha[interior_mask], 225)

    return np.array(Image.fromarray(smoothed_alpha, mode="L").filter(ImageFilter.GaussianBlur(0.45)), dtype=np.uint8)


def refine_alpha_mask(alpha_channel: np.ndarray) -> np.ndarray:
    modern = refine_alpha_mask_modern(alpha_channel)
    legacy = refine_alpha_mask_legacy(alpha_channel)

    # Some cars/images behave better with stricter legacy hole handling.
    if count_enclosed_hole_pixels(modern) <= count_enclosed_hole_pixels(legacy):
        return modern
    return legacy


def get_rembg_session():
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                try:
                    _session = new_session("isnet-general-use")
                except Exception:
                    _session = new_session("u2net")
    return _session


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/remove-bg")
def remove_bg():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    uploaded = request.files["image"]
    if uploaded.filename == "":
        return jsonify({"error": "Image filename is empty."}), 400

    try:
        source_bytes = uploaded.read()
        source_img = Image.open(io.BytesIO(source_bytes))
        source_img = ImageOps.exif_transpose(source_img)
        source_img.load()

        normalized_buffer = io.BytesIO()
        source_img.convert("RGB").save(normalized_buffer, format="PNG")
        normalized_png = normalized_buffer.getvalue()

        result_bytes = remove(
            normalized_png,
            session=get_rembg_session(),
            alpha_matting=True,
            alpha_matting_foreground_threshold=235,
            alpha_matting_background_threshold=8,
            alpha_matting_erode_size=3,
            post_process_mask=True,
        )

        transparent_img = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
        alpha = np.array(transparent_img.getchannel("A"), dtype=np.uint8)
        refined_alpha = refine_alpha_mask(alpha)
        transparent_img.putalpha(Image.fromarray(refined_alpha, mode="L"))

        output_buffer = io.BytesIO()
        transparent_img.save(output_buffer, format="PNG", optimize=True)
        encoded = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "Unsupported or corrupted image."}), 400
    except Exception:
        return jsonify({"error": "Background removal failed."}), 500

    return jsonify({"image_base64": encoded})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
