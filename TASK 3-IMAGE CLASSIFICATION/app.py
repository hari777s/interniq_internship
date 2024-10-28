import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('cifar10_model.h5')

# Define the CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((32, 32))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict the class of an image
def predict(image):
    prediction = model.predict(image)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

# Streamlit app
def main():
    st.title("CIFAR-10 Image Classification")
    st.write("Upload an image and the model will predict its class.")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")

        # Preprocess and predict
        st.write("Classifying...")
        processed_image = preprocess_image(image)
        predicted_class, confidence = predict(processed_image)
        
        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()
