# CNN Digit Recognizer — Streamlit UI

A simple Streamlit UI for the MNIST CNN model trained in your notebook (`cnn.ipynb`), which saves the model as `cnn_model.keras`.

## Setup

1. Make sure `cnn_model.keras` (produced at the end of your notebook via `cnn.save("cnn_model.keras")`) is in the **same folder** as `app.py`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Features

- **Draw a digit** directly on an in-browser canvas, or **upload an image** of a handwritten digit.
- Automatically preprocesses input to match the notebook's pipeline: grayscale → resize to 28x28 → normalize to `[0, 1]` → reshape to `(1, 28, 28, 1)`.
- Auto-inverts colors if you upload a black-on-white image (MNIST digits are white-on-black).
- Shows the predicted digit, confidence score, and full probability bar chart for all 10 classes.

## Notes

- The canvas input uses `streamlit-drawable-canvas`, which needs an internet connection the first time `pip install` runs.
- If the model file has a different name/path, update this line in `app.py`:
  ```python
  model = tf.keras.models.load_model("cnn_model.keras")
  ```
