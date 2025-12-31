
import streamlit as st
import cv2
import numpy as np
import joblib
import json
import os
from skimage.feature import hog

# -------------------------------
# LOAD MODELS FROM GOOGLE DRIVE
# -------------------------------
MODEL_DIR = "/content/drive/MyDrive/plant-disease-project/"

svm_model = joblib.load(MODEL_DIR + "svm_model.joblib")
pca = joblib.load(MODEL_DIR + "pca.joblib")
scaler = joblib.load(MODEL_DIR + "scaler.joblib")

try:
    reg_model = joblib.load(MODEL_DIR + "reg_model.joblib")
except:
    reg_model = None

try:
    with open(MODEL_DIR + "precautions.json", "r") as f:
        precautions = json.load(f)
except:
    precautions = {}

# -------------------------------
# CLASS NAMES (same as training)
# -------------------------------
class_names = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot',
    'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
    'Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy',
    'Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy'
]

# -------------------------------
# PREDICT FUNCTION
# -------------------------------
def predict_image(path):
    img = cv2.imread(path)
    if img is None:
        return {"error": f"Image not found: {path}"}

    img = cv2.resize(img, (128, 128))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # HOG FEATURES
    features = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True
    )

    # Scale + PCA
    X_scaled = scaler.transform([features])
    X_pca = pca.transform(X_scaled)

    pred_class = svm_model.predict(X_pca)[0]
    class_name = class_names[pred_class]

    # severity optional
    severity = None
    if reg_model is not None:
        severity = float(np.clip(reg_model.predict(X_pca)[0], 0, 100))

    precaution = precautions.get(class_name, "No precaution found.")

    return {
        "class": class_name,
        "severity": severity,
        "precaution": precaution
    }

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("🌿 Plant Disease Detector (SVM + HOG + PCA)")
st.write("Upload a leaf image to predict disease, severity & precautions.")

uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img_path = "/content/temp.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(img_path, caption="Uploaded image", use_column_width=True)

    result = predict_image(img_path)

    st.subheader("Prediction:")
    st.write(f"**Disease:** {result['class']}")

    if result["severity"] is not None:
        st.write(f"**Severity (0-100):** {result['severity']:.2f}%")

    st.subheader("Precaution:")
    st.write(result["precaution"])
