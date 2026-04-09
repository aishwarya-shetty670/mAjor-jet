# app.py
import streamlit as st
from PIL import Image

from predict_module import load_image_any, predict_severity, CLASS_NAMES

# Page config
st.set_page_config(page_title="Pothole Severity Detector", layout="centered")

# Title
st.title("🛣️ Pothole Severity Classification")
st.write("Upload a road image or paste an image URL to classify it as **NORMAL**, **MODERATE**, or **SEVERE**.")

# Sidebar
st.sidebar.title("About")
st.sidebar.write("""
This app uses a **CNN (MobileNetV2 transfer learning)** trained on three classes:
- Normal road
- Moderate pothole
- Severe pothole

Model file used: `severity_final.keras`
""")

# Input mode
mode = st.radio("Select input type:", ["Upload Image", "Image URL"])

img = None

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Error reading uploaded image: {e}")

elif mode == "Image URL":
    url = st.text_input("Paste image URL here:")
    if url:
        with st.spinner("Downloading image..."):
            img = load_image_any(url)
        if img is None:
            st.error("Could not load image from this URL. Check the link.")

# If image is available, show and predict
if img is not None:
    st.image(img, caption="Input Image", use_column_width=True)

    if st.button("🔍 Predict Severity"):
        with st.spinner("Analyzing image..."):
            label, conf, probs = predict_severity(img)

        label_display = label.upper()
        if label == "normal":
            emoji = "✅"
            desc = "Road appears safe. No significant potholes detected."
        elif label == "moderate":
            emoji = "⚠️"
            desc = "Moderate pothole detected. Maintenance recommended."
        else:
            emoji = "🚨"
            desc = "Severe pothole detected. Immediate repair required!"

        st.markdown(f"## {emoji} Prediction: **{label_display}**")
        st.write(f"**Confidence:** {conf*100:.2f}%")
        st.write(desc)

        st.subheader("Class Probabilities")
        for cls_name, p in zip(CLASS_NAMES, probs):
            st.write(f"- **{cls_name.upper():8s}** : {p*100:.2f}%")
else:
    st.info("Please upload an image or paste a URL to begin.")