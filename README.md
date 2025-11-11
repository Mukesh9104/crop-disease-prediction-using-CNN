🌱 Crop Disease Prediction System:

A fast and accurate AI-powered Crop Disease Prediction Web App using CNN + Streamlit.

Upload a leaf → Get prediction instantly ✅

🚀 Features

⚡ Deep Learning (CNN-based)

🌿 Built using the PlantVillage dataset

📸 Real-time image prediction

🖥️ Clean Streamlit UI

🔥 Supports GPU training

📁 Easy dataset integration

📥 Dataset

PlantVillage Dataset
🔗 Download: https://data.mendeley.com/datasets/tywbtsjrjv/1

Place the dataset inside your main project folder:

project/
   plantvillage/
   crop_disease_allinone.py

🧠 Technologies Used
Technology	Purpose
TensorFlow / Keras	Model training
OpenCV	Image preprocessing
NumPy	Array operations
Pillow	Image handling
Streamlit	Web UI
Scikit-Learn	Evaluation & preprocessing
⚙️ Installation & Setup
✅ Install dependencies
pip install -r requirements.txt


Or manually install:

pip install tensorflow opencv-python numpy pillow streamlit scikit-learn matplotlib

✅ Train the Model
python crop_disease_allinone.py --train --data_dir "plantvillage"

✅ Run the Streamlit App
streamlit run crop_disease_allinone.py

🔍 Project Summary

Uses a custom Convolutional Neural Network

Implements ImageDataGenerator for augmentation

Achieves high accuracy on validation images

Predicts crop health using leaf images

Optimized for performance

✅ Future Enhancements

Grad-CAM Heatmaps (Explainable AI)

Multi-model comparison

Database + Admin panel

Mobile app version

Cloud deployment (AWS/Render)

👨‍💻 Developer

Mukesh Kanna
