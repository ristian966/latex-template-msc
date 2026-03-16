import base64
import io
import threading

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from rembg import new_session, remove


app = Flask(__name__)

_session = None
_session_lock = threading.Lock()


def get_rembg_session():
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
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
        source_img.load()

        normalized_buffer = io.BytesIO()
        source_img.save(normalized_buffer, format="PNG")
        normalized_png = normalized_buffer.getvalue()

        result_bytes = remove(normalized_png, session=get_rembg_session())

        transparent_img = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
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
