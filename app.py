"""Flask app for the Image Classification Project (V2).

Provides a small web UI and a `/predict` endpoint that accepts an uploaded image,
preprocesses it, and returns a JSON object with the predicted label, confidence
and a low-confidence flag for borderline cases.
"""

from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image

from model_v2 import BetterCNN
from preprocess_v2 import get_loaders

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use get_loaders once to know class order (['FAKE','REAL'] etc)
_, _, _, classes = get_loaders()
print("Classes detected (for Flask):", classes)

# Build a label map from detected classes to friendly labels.
# If a class name contains "fake" (case-insensitive) we label it "AI-Generated".
label_map = {}
for i, cls_name in enumerate(classes):
    name_lower = cls_name.lower()
    if "fake" in name_lower or "ai" in name_lower:
        label_map[i] = "AI-Generated"
    elif "real" in name_lower or "human" in name_lower:
        label_map[i] = "Real"
    else:
        label_map[i] = cls_name

# Instantiate model and load weights (fail gracefully if missing)
model = BetterCNN(num_classes=len(classes)).to(device)
model_path = "model_v2/best_model_v2.pth"
try:
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
except Exception as e:
    # Helpful error message if model file is absent or incompatible
    raise RuntimeError(f"Failed to load model from '{model_path}': {e}")

# For RGB images we must normalize per-channel (3 values)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle an uploaded image file and return prediction JSON.

    The endpoint accepts multipart form data with the `image` field set to the
    uploaded image file. It returns JSON: `{label, confidence, uncertain, note}`.
    """

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    image = Image.open(file.stream).convert("RGB")

    # Downscale to 32x32 first (to simulate the compressed training data),
    # then let the existing transform Resize(128,128) upsample it back to 128x128.
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        # Pillow compatibility for older versions
        resample = Image.LANCZOS if hasattr(
            Image, "LANCZOS") else Image.ANTIALIAS

    orig_size = image.size
    image = image.resize((32, 32), resample)
    print(
        f"[predict] original image size={orig_size}, resized to={image.size} (32x32) before preprocessing")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)[0]
        conf_val, pred_idx = torch.max(probs, 0)
        idx = pred_idx.item()
        label = label_map.get(idx, "Unknown")
        # Confidence as percentage
        confidence = round(conf_val.item() * 100, 2)

        # If the top probability is low, consider the result uncertain/borderline
        uncertain = conf_val.item() < 0.7
        note = "Low confidence — borderline result" if uncertain else ""

    return jsonify({
        "label": label,
        "confidence": confidence,
        "uncertain": uncertain,
        "note": note
    })


if __name__ == "__main__":
    app.run(debug=True)
