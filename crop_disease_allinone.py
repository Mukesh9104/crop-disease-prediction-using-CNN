"""
crop_disease_allinone.py

CPU-friendly CNN + Streamlit:
- Custom CNN
- Fast training (15 epochs)
- Image augmentation, class weights
- Save/load model locally
- Streamlit app for prediction, Grad-CAM, PDF report
"""

import os
import argparse
from pathlib import Path
import tempfile
import io
from datetime import datetime

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import joblib
import cv2

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# Streamlit
import streamlit as st

# PDF export
from fpdf import FPDF

# -------------------------
# Settings
# -------------------------
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 15  # Reduced for faster training
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILE = MODEL_DIR / "crop_model.keras"
META_FILE = MODEL_DIR / "labels.joblib"
HISTORY_FILE = MODEL_DIR / "history.joblib"

# -------------------------
# Disease info
# -------------------------
DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "name": "Bell Pepper — Bacterial Spot",
        "description": "Small, water-soaked lesions on leaves and fruits that turn dark and scabby.",
        "management": "Use certified disease-free seeds, remove infected plants, rotate crops, apply copper-based bactericides."
    },
    "Pepper__bell___healthy": {
        "name": "Bell Pepper — Healthy",
        "description": "No visible symptoms. Leaves and fruits appear normal.",
        "management": "Maintain proper watering, good soil nutrition, and pest control."
    },
    "Potato___Early_blight": {
        "name": "Potato — Early Blight",
        "description": "Dark, concentric ring lesions on older leaves; may cause leaf drop.",
        "management": "Rotate crops, remove infected debris, use fungicides, avoid overhead watering."
    },
    "Potato___Late_blight": {
        "name": "Potato — Late Blight",
        "description": "Water-soaked spots that rapidly enlarge; stems and tubers can rot.",
        "management": "Plant resistant varieties, remove infected plants, apply fungicides, ensure good drainage."
    },
    "Potato___healthy": {
        "name": "Potato — Healthy",
        "description": "No visible symptoms on leaves or tubers.",
        "management": "Maintain good crop hygiene, proper fertilization, and irrigation."
    },
    "Tomato_Bacterial_spot": {
        "name": "Tomato — Bacterial Spot",
        "description": "Small, dark lesions on leaves and fruits; leaves may yellow and drop.",
        "management": "Use disease-free seeds, copper-based sprays, remove infected leaves, crop rotation."
    },
    "Tomato_Early_blight": {
        "name": "Tomato — Early Blight",
        "description": "Dark brown spots with concentric rings on older leaves, may defoliate plants.",
        "management": "Remove infected debris, apply fungicides, practice crop rotation."
    },
    "Tomato_Late_blight": {
        "name": "Tomato — Late Blight",
        "description": "Water-soaked lesions on leaves and fruits; rapid decay in wet conditions.",
        "management": "Plant resistant varieties, remove infected plants, apply protective fungicides."
    },
    "Tomato_Leaf_Mold": {
        "name": "Tomato — Leaf Mold",
        "description": "Yellow spots on upper leaf surface with grayish mold underneath.",
        "management": "Improve ventilation, remove affected leaves, use fungicides, avoid overhead irrigation."
    },
    "Tomato_Septoria_leaf_spot": {
        "name": "Tomato — Septoria Leaf Spot",
        "description": "Small, circular spots with dark borders on leaves, often with gray centers.",
        "management": "Remove infected leaves, apply fungicides, rotate crops, avoid wet foliage."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "name": "Tomato — Spider Mites (Two-spotted)",
        "description": "Tiny mites causing yellowing, stippling, and webbing on leaves.",
        "management": "Spray miticides, encourage natural predators, maintain humidity."
    },
    "Tomato__Target_Spot": {
        "name": "Tomato — Target Spot",
        "description": "Lesions with dark margins and light centers on leaves and stems.",
        "management": "Remove infected debris, apply fungicides, rotate crops."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "name": "Tomato — Yellow Leaf Curl Virus",
        "description": "Leaves curl upward, yellowing along veins, stunted growth.",
        "management": "Use virus-free seedlings, control whiteflies, remove infected plants."
    },
    "Tomato__Tomato_mosaic_virus": {
        "name": "Tomato — Mosaic Virus",
        "description": "Mottled leaves with light and dark green areas; stunted growth.",
        "management": "Use resistant varieties, disinfect tools, remove infected plants."
    },
    "Tomato_healthy": {
        "name": "Tomato — Healthy",
        "description": "No visible symptoms on leaves or fruits.",
        "management": "Maintain proper watering, fertilization, and pest control."
    }
}

# -------------------------
# Build CNN
# -------------------------
def build_custom_cnn(num_classes, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)):
    weight_decay = 1e-4
    inputs = layers.Input(shape=input_shape)

    def conv_block(x, filters, pool=True):
        x = layers.Conv2D(filters, 3, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, 3, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        if pool:
            x = layers.MaxPooling2D(2)(x)
        return x

    x = conv_block(inputs, 32)
    x = layers.Dropout(0.2)(x)
    x = conv_block(x, 64)
    x = layers.Dropout(0.25)(x)
    x = conv_block(x, 128)
    x = layers.Dropout(0.3)(x)
    x = conv_block(x, 256)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------------
# Training
# -------------------------
def train_model(data_dir, epochs=EPOCHS, batch_size=BATCH_SIZE):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found.")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.12,
        height_shift_range=0.12,
        shear_range=0.12,
        zoom_range=0.18,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=(0.7, 1.3),
        fill_mode='nearest',
        validation_split=0.15
    )
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)

    train_flow = train_datagen.flow_from_directory(
        str(data_dir), target_size=IMG_SIZE, batch_size=batch_size,
        class_mode='categorical', subset='training', shuffle=True
    )
    val_flow = val_datagen.flow_from_directory(
        str(data_dir), target_size=IMG_SIZE, batch_size=batch_size,
        class_mode='categorical', subset='validation', shuffle=False
    )

    num_classes = train_flow.num_classes
    print(f"Found {num_classes} classes.")
    print("Class indices:", train_flow.class_indices)

    y_train = train_flow.classes
    classes = np.unique(y_train)
    class_weights_list = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = {int(k): float(v) for k, v in zip(classes, class_weights_list)}
    print("Computed class weights:", class_weight)

    model = build_custom_cnn(num_classes)

    ckpt = ModelCheckpoint(str(MODEL_FILE), monitor='val_accuracy', save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

    history = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=[ckpt, early, reduce_lr]
    )

    label_map = train_flow.class_indices
    index_to_label = {v: k for k, v in label_map.items()}
    joblib.dump({"label_map": label_map, "index_to_label": index_to_label}, str(META_FILE))
    joblib.dump(history.history, str(HISTORY_FILE))
    print("✅ Saved model and metadata:", MODEL_FILE)

# -------------------------
# Utilities
# -------------------------
def load_model_and_meta():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_FILE}. Train first.")
    model = tf.keras.models.load_model(str(MODEL_FILE))
    meta = {}
    if META_FILE.exists():
        meta = joblib.load(str(META_FILE))
    return model, meta

def preprocess_pil(img_pil):
    img = img_pil.convert('RGB').resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32)/255.0
    return np.expand_dims(arr, axis=0)

def predict_single(model, meta, pil_img):
    x = preprocess_pil(pil_img)
    preds = model.predict(x)[0]
    top_idx = int(np.argmax(preds))
    label = meta.get("index_to_label", {}).get(top_idx, str(top_idx))
    conf = float(preds[top_idx])
    sorted_indices = preds.argsort()[::-1]
    topk = [(meta.get("index_to_label", {}).get(i, str(i)), float(preds[i])) for i in sorted_indices[:5]]
    return label, conf, topk, preds

# -------------------------
# Grad-CAM
# -------------------------
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model for Grad-CAM.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs],
                                       [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:,:,i] *= pooled_grads[i]
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap,0)
    heatmap /= (np.max(heatmap)+1e-10)
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    return heatmap

def overlay_heatmap_on_image(pil_img, heatmap, alpha=0.45):
    img = np.array(pil_img.resize(IMG_SIZE)).astype(np.uint8)
    heatmap_uint8 = np.uint8(255*heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(colored, alpha, img, 1-alpha, 0)
    return Image.fromarray(overlayed)

# -------------------------
# PDF report
# -------------------------
def make_simple_pdf_report(label, conf, topk, pil_img):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 8, "Crop Disease Prediction Report", ln=1, align='C')
    pdf.ln(4)
    pdf.set_font("Arial", size=10)
    pdf.cell(0,6,f"Prediction: {label}", ln=1)
    pdf.cell(0,6,f"Confidence: {conf*100:.2f}%", ln=1)
    pdf.ln(4)
    pdf.cell(0,6,"Top predictions:", ln=1)
    for n,p in topk:
        pdf.cell(0,6,f" - {n}: {p*100:.2f}%", ln=1)
    pdf.ln(6)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    pil_img.resize((300,300)).save(tmp.name)
    pdf.image(tmp.name, x=10, y=None, w=80)
    try:
        os.remove(tmp.name)
    except:
        pass
    return pdf.output(dest='S').encode('latin-1')

# -------------------------
# Streamlit UI
# -------------------------
def run_streamlit():
    st.set_page_config(page_title="Crop Disease Detector", layout="wide")
    st.title("🌾 Crop Disease Prediction — CNN (CPU friendly)")

    st.sidebar.header("About")
    st.sidebar.markdown("""
    - Custom CNN
    - Image size: 256x256, strong augmentation, class weights
    - Grad-CAM explainability included
    - Balloons appear after prediction 🎉
    """)

    model, meta = None, {}
    model_loaded = False
    try:
        model, meta = load_model_and_meta()
        model_loaded = True
    except Exception as e:
        st.sidebar.warning("Model not found. Train first via CLI --train.")
        st.sidebar.write(str(e))

    uploaded_file = st.sidebar.file_uploader("Upload leaf image", type=['jpg','jpeg','png'])
    topk_n = st.sidebar.slider("Top K predictions", 1,5,3)
    show_gradcam = st.sidebar.checkbox("Show Grad-CAM", value=True)
    show_info = st.sidebar.checkbox("Show disease info", value=True)

    if 'last_prediction' not in st.session_state:
        st.session_state['last_prediction'] = None
        st.session_state['last_image_bytes'] = None

    if uploaded_file is None:
        st.info("Upload an image to predict.")
        return

    pil_img = Image.open(uploaded_file).convert('RGB')
    st.image(pil_img, caption="Uploaded Image", width=320)

    if st.sidebar.button("Predict"):
        if not model_loaded:
            st.error("Model not loaded. Train first.")
            return
        with st.spinner("Predicting..."):
            label, conf, topk, raw = predict_single(model, meta, pil_img)
            st.session_state['last_prediction'] = {
                'label': label, 'conf': conf, 'topk': topk, 'raw': raw.tolist()
            }
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG')
            st.session_state['last_image_bytes'] = buf.getvalue()

    if st.session_state.get('last_prediction') is not None:
        data = st.session_state['last_prediction']
        label, conf, topk = data['label'], data['conf'], data['topk']

        st.success(f"Prediction: {label.replace('___',' → ')}")
        st.info(f"Confidence: {conf*100:.2f}%")
        st.balloons()

        names = [n.replace('___',' → ') for n,p in topk[:topk_n]]
        probs = [p*100 for n,p in topk[:topk_n]]
        fig,ax = plt.subplots(figsize=(7,max(1,0.6*len(names))))
        ax.barh(range(len(names))[::-1], probs)
        ax.set_yticks(range(len(names))[::-1])
        ax.set_yticklabels(names)
        ax.set_xlabel("Probability (%)")
        st.pyplot(fig)

        if show_info:
            info = DISEASE_INFO.get(label)
            if info:
                st.subheader("Disease Info & Management")
                st.write("**Name:**", info["name"])
                st.write("**Description:**", info["description"])
                st.write("**Management:**", info["management"])
            else:
                st.warning("No info available for this label.")

        if show_gradcam:
            try:
                img_bytes = st.session_state.get('last_image_bytes')
                pil_img_state = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                x = preprocess_pil(pil_img_state)
                last_conv = find_last_conv_layer(model)
                heatmap = make_gradcam_heatmap(x, model, last_conv)
                overlay = overlay_heatmap_on_image(pil_img_state, heatmap)
                st.subheader("Grad-CAM Explainability")
                col1,col2 = st.columns(2)
                col1.image(pil_img_state.resize(IMG_SIZE), caption="Original", use_column_width=True)
                col2.image(overlay, caption="Grad-CAM overlay", use_column_width=True)
            except Exception as e:
                st.warning("Grad-CAM failed: "+str(e))

        txt = f"Prediction: {label}\nConfidence: {conf*100:.2f}%\nTop predictions:\n"
        for n,p in topk:
            txt += f"{n}: {p*100:.2f}%\n"
        st.download_button("Download result (.txt)", data=txt, file_name="prediction_result.txt")

        if st.button("Download PDF report"):
            img_bytes = st.session_state.get('last_image_bytes')
            pil_img_state = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            pdf_bytes = make_simple_pdf_report(label, conf, topk, pil_img_state)
            st.download_button("Click to download PDF", data=pdf_bytes, file_name="prediction_report.pdf",
                               mime="application/pdf")

# -------------------------
# CLI entry
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model (provide --data_dir)")
    parser.add_argument("--data_dir", type=str, help="Path to dataset root (class subfolders inside)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args,_ = parser.parse_known_args()

    if args.train:
        if not args.data_dir:
            raise ValueError("Provide --data_dir when using --train")
        train_model(args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
        return

    run_streamlit()

if __name__=="__main__":
    main()
