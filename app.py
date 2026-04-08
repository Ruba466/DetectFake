import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import gdown
import onnxruntime as ort

# ============================================================
# CONFIG — Update GDRIVE_FILE_ID with your new model's file ID
# ============================================================
GDRIVE_FILE_ID = "1VDtnUzSBCRiw4m_MXdYNIc1Wyjelpo_r"  # update with new model ID
MODEL_PATH = "models/best_model.onnx"
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🕵️",
    layout="centered"
)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        with st.spinner("Downloading model for first time (~100MB)..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    session = ort.InferenceSession(MODEL_PATH)
    return session

def preprocess_image(image):
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def main():
    st.title("🕵️ Deepfake & AI-Generated Face Detection")
    st.markdown("Upload a facial image to determine if it is **REAL** or **FAKE**.")

    st.sidebar.header("About")
    st.sidebar.info(
        "This system uses a fine-tuned Xception CNN model trained on "
        "~170,000 diverse facial images to detect deepfakes and "
        "AI-generated faces from multiple generation techniques."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model:** Xception (Transfer Learning)")
    st.sidebar.markdown("**Architecture:** Xception CNN + ONNX Runtime")
    st.sidebar.markdown("**Validation Accuracy:** 88.54%")
    st.sidebar.markdown("**AUC Score:** 0.9641")
    st.sidebar.markdown("**F1-Score:** 0.88")
    st.sidebar.markdown("**Fake Recall:** 97%")
    st.sidebar.markdown("**Datasets:** 140k Faces + Real vs AI Generated")
    st.sidebar.markdown("**Training Images:** ~170,000")
    st.sidebar.markdown("**Detects:** StyleGAN2 · Stable Diffusion · FaceSwap · SFHQ")
    st.sidebar.markdown("**Platform:** Kaggle (Tesla T4 x2)")

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)

        with st.spinner("Analyzing image..."):
            session = load_model()
            img_array = preprocess_image(image)
            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: img_array})
            prediction = float(output[0][0][0])

        # sigmoid: close to 1 = real, close to 0 = fake
        is_real = prediction > 0.5
        confidence = prediction if is_real else 1 - prediction
        label = "REAL" if is_real else "FAKE"
        color = "#00cc66" if is_real else "#ff3333"

        with col2:
            st.subheader("Result")
            st.markdown(
                f"<h2 style='text-align:center; color:{color}'>{label}</h2>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<h4 style='text-align:center'>Confidence: {confidence*100:.2f}%</h4>",
                unsafe_allow_html=True
            )
            if confidence < 0.70:
                st.warning("⚠️ Low confidence — try a clearer frontal face image.")
            elif confidence < 0.85:
                st.info("ℹ️ Moderate confidence prediction.")
            else:
                st.success("✅ High confidence prediction.")

        st.markdown("---")
        st.subheader("Prediction Scores")
        col3, col4 = st.columns(2)
        with col3:
            st.metric("REAL probability", f"{prediction*100:.2f}%")
        with col4:
            st.metric("FAKE probability", f"{(1-prediction)*100:.2f}%")

        st.progress(float(confidence))

        st.markdown("---")
        st.caption(
            "Note: This model is trained on StyleGAN2, Stable Diffusion, FaceSwap, and SFHQ images. "
            "Performance may vary on generation methods not seen during training."
        )

if __name__ == "__main__":
    main()
