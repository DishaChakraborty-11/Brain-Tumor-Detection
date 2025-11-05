# app/app.py
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import torch
from src.predict import predict_image, allowed_file

# Config
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = os.environ.get("FLASK_SECRET", "change-this-secret")

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "brain_tumor_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

def load_model():
    global model
    if model is None:
        from src.model import SimpleCNN  # model architecture
        net = SimpleCNN(num_classes=2)
        state = torch.load(MODEL_PATH, map_location=device)
        # handle if saved state_dict or whole model
        if isinstance(state, dict) and "state_dict" in state:
            net.load_state_dict(state["state_dict"])
        elif isinstance(state, dict) and next(iter(state)).startswith("conv"):
            net.load_state_dict(state)
        else:
            try:
                net = state
            except Exception as e:
                raise RuntimeError(f"Unable to load model: {e}")
        net.to(device)
        net.eval()
        model = net

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("No file part")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for("index"))

    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        try:
            load_model()
            result, prob = predict_image(model, save_path, device=device)
            return render_template("result.html", filename=f"static/uploads/{filename}", result=result, prob=round(prob*100,2))
        except Exception as e:
            app.logger.exception("Prediction error")
            flash(f"Prediction failed: {e}")
            return redirect(url_for("index"))
    else:
        flash("Allowed image types are png, jpg, jpeg")
        return redirect(url_for("index"))

if __name__ == "__main__":
    # When running locally for debugging:
    app.run(host="0.0.0.0", port=5000, debug=True)
