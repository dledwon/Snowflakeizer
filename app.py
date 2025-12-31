from flask import Flask, jsonify, request, send_from_directory
from model import generate_random, generate_img2img, images_to_base64

app = Flask(__name__, static_folder="static")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)


@app.route("/api/generate", methods=["GET"])
def generate():
    steps = request.args.get("steps", default=25, type=int)
    image = generate_random(steps)
    img_b64 = images_to_base64(image)
    return jsonify({"image": img_b64})


@app.route("/api/img2img", methods=["POST"])
def img2img():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    steps = int(request.form.get("steps", 25))
    strength = float(request.form.get("strength", 0.7))

    try:
        image = generate_img2img(file, steps, strength)
        img_b64 = images_to_base64(image)
        return jsonify({"image": img_b64})
    except Exception as e:
        print("Img2Img error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)
