from flask import Flask, render_template, request, redirect, url_for
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your trained model later
# model = torch.load('model/brain_tumor_model.pth')
# model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect('/')
    
    file = request.files['file']
    if file.filename == '':
        return redirect('/')
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    # Preprocess the image (placeholder for now)
    image = Image.open(filepath).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    
    # ---- Dummy prediction (since model not trained yet) ----
    prediction = "Brain Tumor Detected ✅" if "yes" in file.filename.lower() else "No Tumor Detected ❌"

    # When model is trained, replace with:
    # output = model(image)
    # prediction = "Brain Tumor Detected ✅" if torch.argmax(output) == 1 else "No Tumor Detected ❌"

    return render_template('result.html', prediction=prediction, image_name=file.filename)

if __name__ == '__main__':
    app.run(debug=True)

