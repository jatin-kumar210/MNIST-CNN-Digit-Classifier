import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(page_title="CNN Digit Recognizer", page_icon="🔢", layout="centered")

st.title("🔢 Handwritten Digit Recognizer")
st.caption("Powered by the CNN model trained in `cnn.ipynb` (MNIST, 28x28 grayscale)")


# ----------------------------------------------------------------------
# Load model (cached so it only loads once)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cnn_model.keras")
    return model


try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(
        "Could not load `cnn_model.keras`. Make sure the file is in the same "
        "folder as this app.\n\nError: " + str(e)
    )

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.header("Options")
input_mode = st.sidebar.radio("Input method", ["Draw a digit", "Upload an image"])
stroke_width = st.sidebar.slider("Brush size", 5, 25, 15)
st.sidebar.markdown("---")
st.sidebar.write(
    "The model expects a **28x28 grayscale** image, just like the MNIST "
    "dataset it was trained on. Draw a digit with a thick white/black stroke "
    "on a plain background for best results."
)

# ----------------------------------------------------------------------
# Helper: preprocess image -> match notebook preprocessing
# (astype float32 / 255.0, reshape to (1, 28, 28, 1))
# ----------------------------------------------------------------------
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    # Convert to grayscale
    img = pil_img.convert("L")
    # Resize to 28x28 (MNIST size)
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img).astype("float32")

    # MNIST digits are white-on-black. If the uploaded/drawn image looks like
    # black-on-white (mean pixel value is high/bright), invert it.
    if arr.mean() > 127:
        arr = 255.0 - arr

    arr = arr / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr


def predict(arr: np.ndarray):
    preds = model.predict(arr, verbose=0)[0]
    pred_class = int(np.argmax(preds))
    confidence = float(preds[pred_class])
    return pred_class, confidence, preds


# ----------------------------------------------------------------------
# Input: Draw a digit
# ----------------------------------------------------------------------
input_image = None

if input_mode == "Draw a digit":
    st.write("Draw a single digit (0-9) below:")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=stroke_width,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None and canvas_result.image_data.sum() > 0:
        input_image = Image.fromarray(
            canvas_result.image_data.astype("uint8"), mode="RGBA"
        ).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        predict_clicked = st.button("🔍 Predict", use_container_width=True, type="primary")
    with col2:
        st.write("")  # spacer; canvas has its own clear/undo controls

else:
    st.write("Upload an image containing a single handwritten digit:")
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    predict_clicked = False
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Uploaded image", width=200)
        predict_clicked = st.button("🔍 Predict", type="primary")

# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
if predict_clicked:
    if not model_loaded:
        st.warning("Model isn't loaded, so prediction can't run.")
    elif input_image is None:
        st.warning("Please draw or upload a digit first.")
    else:
        processed = preprocess_image(input_image)

        st.subheader("What the model sees (28x28)")
        st.image(
            (processed.reshape(28, 28) * 255).astype("uint8"),
            width=140,
            clamp=True,
        )

        pred_class, confidence, all_probs = predict(processed)

        st.subheader(f"Prediction: **{pred_class}**")
        st.metric("Confidence", f"{confidence * 100:.2f}%")

        st.write("Class probabilities:")
        st.bar_chart({"probability": all_probs}, height=250)

st.markdown("---")
st.caption(
    "Model architecture: Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → "
    "Flatten → Dense(128, relu) → Dropout(0.5) → Dense(10, softmax)"
)
