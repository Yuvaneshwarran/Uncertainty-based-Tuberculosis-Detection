import streamlit as st
import cv2 as cv
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# Function to preprocess the input image
def preprocess_image(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, (256, 256))
    image = image.astype('float32') / 255.0
    return image.reshape(-1, 256, 256, 1)

# Class for Monte Carlo Dropout
class MCDropout(Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

# Load your pre-trained Bayesian CNN model
def create_bayesian_cnn():
    model = keras.Sequential([
        keras.Input(shape=(256, 256, 1)),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        MCDropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy'],
    )
    return model

# Load the model
cnn_bayesian = create_bayesian_cnn()
cnn_bayesian.load_weights('./tb.weights.h5')  # Adjust this path

# Streamlit app layout
st.title("Tuberculosis Image Classifier")
st.write("Upload a chest X-ray image to classify as Normal or Tuberculosis.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = cv.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv.IMREAD_COLOR)
    
    st.image(image, caption='Uploaded Image', channels='BGR')
    st.write("")
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Make predictions
    T = 10  # Number of stochastic forward passes
    predictions = np.stack([cnn_bayesian.predict(preprocessed_image) for _ in range(T)], axis=0)
    
    # Calculate mean predictions and uncertainty
    mean_predictions = np.mean(predictions, axis=0)
    predicted_label = (mean_predictions > 0.5).astype('int32')[0][0]
    std_dev = np.std(predictions, axis=0)[0][0]
    
    # Display results
    if predicted_label == 0:
        st.success("The image is classified as Normal.")
    else:
        st.error("The image is classified as Tuberculosis.")
    
    st.write(f"Mean Prediction: {mean_predictions[0][0]:.4f}")
    st.write(f"Uncertainty (Standard Deviation): {std_dev:.4f}")

