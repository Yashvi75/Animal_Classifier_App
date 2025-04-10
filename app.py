
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.markdown("""
<style>
body {
    background-color: #fdf6f0;
}
</style>
""", unsafe_allow_html=True)

logo_url = "https://cdn-icons-png.flaticon.com/512/616/616408.png"
st.image(logo_url, width=100)

# Load model
model = load_model("Animal_classifier_model.keras")

# Class labels
class_labels = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant',
                'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']

# Streamlit UI
st.title("🐾 Animal Image Classifier")
st.write("Upload an image of an animal and I will tell you which one it is!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_index]

    st.markdown(f"""
<div style='background-color: #cce5ff; padding: 10px; border-radius: 8px;'>
    <h4 style='color: #004085;'>🔍 Confidence: {confidence:.2f}%</h4>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div style='background-color: #d4edda; padding: 10px; border-radius: 8px;'>
    <h3 style='color: #155724;'>🐾 Prediction: {predicted_label}</h3>
</div>
""", unsafe_allow_html=True)

