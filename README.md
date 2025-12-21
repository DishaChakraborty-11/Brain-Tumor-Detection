# 🧠 Brain Tumor Detection using Convolutional Neural Networks

> A deep learning project that detects and classifies brain tumors from MRI images using a Convolutional Neural Network (CNN).  
> The model achieves high accuracy and helps in early diagnosis through automated image classification.

⚠️ Disclaimer: This project is intended for learning and demonstration purposes only and is not suitable for clinical or medical use.
---
Dataset

Source: Public Brain Tumor MRI datasets available on Kaggle

Data type: MRI brain scan images

Classes:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

Notes:

- A subset of the dataset was used for training and evaluation.
- Images were resized and normalized before training.
- Basic data augmentation was applied to improve generalization.

---

## 🚀 Project Overview
This project aims to automate brain tumor detection from MRI images using deep learning techniques.  
A Convolutional Neural Network (CNN) is trained to classify MRI images as either **tumor** or **non-tumor**.  

The dataset was preprocessed to ensure better performance through normalization, augmentation, and resizing.

---

## 🧩 Features
- 🧠 Automatic detection of tumors from MRI scans  
- 🧹 Data preprocessing and augmentation for accuracy improvement  
- 📊 Model evaluation using confusion matrix and accuracy metrics  
- 💾 Model saved for reuse and future deployment  

---

## 🛠️ Tech Stack
| Category | Tools Used |
|-----------|-------------|
| **Language** | Python |
| **Frameworks** | TensorFlow, Keras |
| **Libraries** | NumPy, OpenCV, Matplotlib, Scikit-learn |
| **IDE** | Jupyter Notebook / Google Colab |

---
## 📂 Project Structure
Brain-Tumor-Detection/
│
├── dataset/
│ ├── yes/ # MRI images with tumors
│ └── no/ # MRI images without tumors
├── brain_tumor_detection.ipynb
├── model/
│ └── brain_tumor_model.h5
├── static/
│ └── screenshots/
│ ├── sample_prediction.png
│ ├── accuracy_graph.png
│ └── confusion_matrix.png
├── README.md
└── requirements.txt



---

## 📈 Model Performance
| Metric | Value |
|---------|--------|
| Training Accuracy | 98% |
| Validation Accuracy | 96% |
| Loss | 0.12 |

📊 *Example Graphs:*
![Accuracy Graph](static/screenshots/accuracy_graph.png)
![Confusion Matrix](static/screenshots/confusion_matrix.png)

---

## 💻 How to Run
1. Clone the repository  
   ``bash
   git clone https://github.com/DishaChakraborty-11/Brain-Tumor-Detection.git
Install dependencies

pip install -r requirements.txt

Open the Jupyter Notebook

jupyter notebook brain_tumor_detection.ipynb

Run all cells to train and evaluate the model.

🧠 Future Improvements
Integrate the model into a Flask web app for real-time predictions

Use Grad-CAM for visualizing the tumor area in MRI images

Deploy the model using Streamlit or Render

---

Limitations

Trained on a limited public dataset

Dataset may be imbalanced

No cross-dataset or clinical validation performed

Not optimized for production or real-world medical deployment

---

Intended Use

Educational demonstrations

Portfolio showcase

Learning CNN-based medical image classification

Understanding ML deployment workflows

---

👩‍💻 Author
Disha Chakraborty
B.Tech CSE (AI & ML) 

