import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image
import numpy as np
import json
import os

# Load model and class indices
@st.cache_resource
def load_model_and_classes():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = f"{working_dir}/trained_model/my_keras_model.keras"
    model = tf.keras.models.load_model(model_path)
    class_indices = json.load(open(f"{working_dir}/class_indices.json"))
    return model, class_indices

model, class_indices = load_model_and_classes()

# Define prediction function
def predict_image_class(model, uploaded_image, class_indices):
    # Preprocess the uploaded image
    image = Image.open(uploaded_image).convert('RGB')
    image = image.resize((224, 224))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict the class
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    # Get the class label
    predicted_label = class_indices[str(predicted_class)]
    return predicted_label, confidence

# Streamlit App Interface
st.title('Plant Disease Classifier')

# Upload an image
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img, caption="Uploaded Image", use_container_width=True)

    with col2:
        if st.button('Classify'):
            # Predict the class of the uploaded image
            label, confidence = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {label}')
            st.info(f'Confidence: {confidence:.2f}')
