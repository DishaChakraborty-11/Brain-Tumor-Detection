# src/predict.py
from PIL import Image
import torch
from torchvision import transforms
import os

def allowed_file(filename, allowed_set):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set

def predict_image(model, image_path, device=torch.device("cpu")):
    # Preprocessing: grayscale, resize 64x64, normalize to [0,1]
    input_size = 64
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),  # scales to [0,1]
        # optionally normalize: transforms.Normalize((0.5,), (0.5,))
    ])

    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)  # shape [1,1,64,64] or [1,3..] depending

    model.eval()
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        prob, pred_idx = torch.max(probs, dim=1)
        pred_idx = pred_idx.item()
        prob = prob.item()
    label_map = {0: "Non-Tumor", 1: "Tumor"}  # adjust if your classes reversed
    return label_map.get(pred_idx, "Unknown"), prob
