# 😊 Real-Time Facial Emotion Recognition System

This project is a deep learning-based real-time facial emotion recognition system that detects human emotions (e.g., Happy, Sad, Angry, Neutral) from video input. It uses trained CNN models and advanced architectures like VGG19, Xception, and Vision Transformers for accurate emotion classification.

## 📌 Features

- Real-time video emotion recognition
- Trained with custom CNN and pre-trained deep learning models
- Streamlit/Flask-based app interface (`app.py`)
- Visualization of training metrics and architecture
- Includes output demo videos

## 🧠 Models Used

- CNN (Custom Convolutional Neural Network)
- VGG19
- Xception
- Vision Transformers (ViT)
- Final model exported as `model_filter.h5`

## 🗂️ Project Structure

├── app.py # Main application for real-time prediction
├── model_filter.h5 # Final trained model
├── OUTPUT_VIDEO.mp4 # Demo of output
├── requirements.txt # Python dependencies
├── Model/
│ ├── cnn_train_model.ipynb # CNN training notebook
│ ├── vgg19_tarin_model.ipynb # VGG19 training notebook
│ ├── xception_model_file.ipynb # Xception model implementation
│ ├── vision_transformes.ipynb # Vision Transformer training
│ ├── Face_Emotion_Recognition.ipynb# Master training logic
├── Dataset/ # Dataset for training/testing
└── arc.png # Model architecture diagram

markdown
Copy code

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
Run the application:

bash
Copy code
python app.py
The webcam will open and display live emotion predictions on your face.

🎥 Demo
OUTPUT_VIDEO.mp4: shows the model working on a real-time video stream.

Screen Recording 2025-05-05.mp4: project walkthrough recording.

📊 Dataset
Custom dataset under /Dataset/ with labeled emotion images.

Classes include: Happy, Sad, Angry, Surprise, Neutral, etc.

💡 Highlights
Real-time performance using OpenCV + TensorFlow

Multiple model comparison and evaluation

Vision Transformer integration for modern deep learning insights

🙌 Acknowledgements
TensorFlow and Keras

FER2013 and other facial expression datasets

Pre-trained weights for VGG19, Xception

