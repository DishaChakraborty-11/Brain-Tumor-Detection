# Brain-Tumor-Detection
This project is a Flask web application for detecting brain tumors from MRI images using a deep learning model built with PyTorch.


Dataset:
The dataset contains MRI images, divided into two categories: tumor and non-tumor.
Preprocessing techniques are applied to the dataset to ensure optimal model performance.

Project Overview
This end-to-end project consists of:

Data Loading: Load MRI images for training, validation, and testing.
Data Preprocessing: Apply normalization, resizing, and augmentation techniques.
Model Building: Build a Convolutional Neural Network (CNN) using PyTorch to classify the MRI images.
Model Training: Train the model on GPU (if available) to detect brain tumors.
Flask Web Application: Develop a Flask app for user interaction, allowing image uploads for tumor detection.
Model Deployment: Deploy the trained model within the Flask app.
Prediction: Provide real-time predictions through the Flask web app.


Model Download and Directory Structure
Pretrained Model:
You can download the pretrained model from the following link: Brain Tumor Detection Model

Directory Structure:
Brain-Tumor-Detection/
│
├── app/
│   ├── static/                 # CSS, JS, and images for the Flask web app
│   ├── templates/              # HTML templates for the Flask app
│   └── app.py                  # Main Flask application
│
├── model/
│   └── brain_tumor_model.pth   # Pretrained PyTorch model
│
├── data/
│   ├── train/                  # Training MRI images
│   ├── test/                   # Testing MRI images
│
├── src/
│   ├── dataset.py              # Script to load and preprocess the dataset
│   ├── model.py                # CNN model architecture using PyTorch
│   └── train.py                # Script to train the model
│
├── README.md                   # Project documentation
└── requirements.txt            # List of required Python packages


Setup Instructions
Step 1: Create a Virtual Environment
Create a virtual environment to isolate the dependencies for this project.

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
Step 2: Install Required Libraries
Install the dependencies listed in requirements.txt:

pip install -r requirements.txt
Step 3: Download the Pretrained Model
Download the pretrained model from this link and place it in the model/ directory as brain_tumor_model.pth.

Step 4: Running the Flask App
To start the Flask web app, navigate to the app/ directory and run the app.py file:

cd app/
python app.py
The app will be hosted at http://127.0.0.1:5000/. You can open the URL in your browser and upload MRI images to receive predictions.

Flask Web Application Features
Image Upload: Users can upload MRI images through the web interface.
Tumor Detection: The uploaded image is fed into the model to predict whether a tumor is present.
Result Display: The result is displayed on the same page with either a "Tumor" or "Non-Tumor" label.

Model Architecture
The model used in this project is a Convolutional Neural Network (CNN) built using PyTorch. The architecture has been optimized for image classification tasks and consists of several layers:

Key Layers:
Convolutional Layers: For feature extraction from MRI images.
Max Pooling Layers: For downsampling and reducing spatial dimensions.
Fully Connected Layers: For classification.
Softmax Activation: To produce the output probability of each class (Tumor/Non-Tumor).

Data Preprocessing
To ensure the CNN model performs optimally, the following preprocessing steps are applied:

Grayscale Conversion: All MRI images are converted to grayscale.
Resizing: Images are resized to 64x64 pixels for uniformity.
Normalization: Each pixel value is normalized to a range of [0, 1].
Data Augmentation: Techniques like random rotation, flipping, and zooming are applied to expand the dataset and prevent overfitting.

Conclusion
This Flask web app provides an end-to-end solution for detecting brain tumors using MRI images. With a simple user interface and a powerful backend, it can serve as a diagnostic tool for medical professionals. The project can be further enhanced by incorporating additional data, improving model accuracy, or deploying the app to a cloud platform like Heroku.

Future Enhancements
Integration with Cloud Platforms: Deploy the app on Heroku or AWS for wider accessibility.
Mobile Application: Develop a mobile app to upload MRI images and get predictions on the go.
Transfer Learning: Incorporate pre-trained models like ResNet to further improve accuracy.
