import streamlit as st
from PIL import Image
from predict import predict_image
from utils.tumor_info import tumor_details

st.set_page_config(page_title="Brain Tumor MRI Detection", layout="wide")

st.title("Brain Tumor MRI Detection with Grad-CAM")
st.write("Upload an MRI scan to predict tumor type and visualize the attention heatmap.")
st.warning("For educational use only. Not a medical diagnosis tool.")

uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    predicted_label, confidence, heatmap_overlay = predict_image(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original MRI")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Grad-CAM Heatmap")
        st.image(heatmap_overlay, use_column_width=True)

    st.subheader("Prediction")
    st.success(f"{predicted_label} ({confidence * 100:.2f}% confidence)")

    info = tumor_details[predicted_label]

    st.subheader("Description")
    st.write(info["description"])

    st.subheader("Common Symptoms")
    st.write(info["common_symptoms"])

    st.subheader("Treatment Overview")
    st.write(info["treatment"])