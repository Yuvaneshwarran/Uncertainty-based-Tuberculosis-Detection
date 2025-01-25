#Uncertainty Assisted Robust Tuberculosis Identification With Bayesian Convolutional Neural Networks

# Importing the necessary libraries for dataset loading and preprocessing
import cv2 as cv
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import os

# Initializing the values needed for all the image files
normaldir = './Normal'
tbdir = './Tuberculosis'
images = []
labels = []
imagesize = 256

# Storing all the image directories in the 'images' array and corresponding them to either 1 for TB images or 0 for normal images.
for x in os.listdir(normaldir):
    imagedir = os.path.join(normaldir, x)
    image = cv.imread(imagedir, cv.IMREAD_GRAYSCALE)
    image = cv.resize(image, (imagesize, imagesize))
    images.append(image)
    labels.append(0)

for y in os.listdir(tbdir):
    imagedir = os.path.join(tbdir, y)
    image = cv.imread(imagedir, cv.IMREAD_GRAYSCALE)
    image = cv.resize(image, (imagesize, imagesize))
    images.append(image)
    labels.append(1)

# Converting to NumPy arrays since they have more features than regular lists
images = np.array(images)
labels = np.array(labels)

# Splitting the images and labels into training and testing sets, then normalizing the values within them for computational efficiency (from 0-255 scale to 0-1 scale)
imagetrain, imagetest, labeltrain, labeltest = train_test_split(images, labels, test_size=0.3, random_state=42)
imagetrain = (imagetrain.astype('float32')) / 255
imagetest = (imagetest.astype('float32')) / 255

# Flattening the image array into 2D to be suitable for SMOTE oversampling
imagetrain = imagetrain.reshape(2940, (imagesize*imagesize))

# Performing oversampling
smote = SMOTE(random_state=42)
imagetrain, labeltrain = smote.fit_resample(imagetrain, labeltrain)

# Unflattening the images back for use in the convolutional neural network
imagetrain = imagetrain.reshape(-1, imagesize, imagesize, 1)
print(imagetrain.shape)

# Classes balanced - equal counts of each label
print(np.unique(labeltrain, return_counts=True))

# Importing necessary libraries for the Bayesian CNN model
import tensorflow as tf
import keras
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Defining a function to add Monte Carlo Dropout (keeping dropout active during inference)
class MCDropout(Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)  # Forces dropout during inference as well

# The Bayesian CNN model with MC Dropout
cnn_bayesian = keras.Sequential(
    [
    # Input layer, same shape as all the images (256x256x1):
    keras.Input(shape=(imagesize, imagesize, 1)),
    
    # 1st convolutional layer:
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # 2nd convolutional layer:
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # 3rd convolutional layer:
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Flattening layer for the dense layers:
    Flatten(),
    
    # 1st dense layer following the convolutional layers:
    Dense(64, activation='relu'),
    
    # Bayesian Dropout layer - MC Dropout for stochastic predictions
    MCDropout(0.5),
    
    # Output layer that squeezes each image to either 0 or 1 with sigmoid activation
    Dense(1, activation='sigmoid')
    ]
)

# Compiling the model
cnn_bayesian.compile(
    loss='binary_crossentropy',  # Best for binary classification
    optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Good starting LR for dataset of this size
    metrics=['accuracy'],  # Looking for accuracy
)

# Fitting the model with a ReduceLROnPlateau callback
from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=1, min_lr=0.00001, verbose=1)

cnn_bayesian.fit(imagetrain, labeltrain, batch_size=16, epochs=10, verbose=2, callbacks=[reduce_lr])
cnn_bayesian.save_weights('./tb.weights.h5')
# Evaluating the data
print('TESTING DATA:')
cnn_bayesian.evaluate(imagetest, labeltest, batch_size=32, verbose=2)
# Advanced metrics, but now with Monte Carlo Dropout for multiple stochastic predictions
print('ADVANCED TESTING METRICS:')
from sklearn.metrics import classification_report, confusion_matrix

# Performing multiple forward passes to estimate uncertainty
T = 10  # Number of stochastic forward passes (can be increased for better uncertainty estimation)
predictions = np.stack([cnn_bayesian.predict(imagetest, batch_size=32) for _ in range(T)], axis=0)
print("predictions: ",predictions)

#print(cnn_bayesian.predict(imagetest,batch_size=32))

# Averaging the predictions across T runs to get the final prediction
mean_predictions = np.mean(predictions, axis=0)
predicted_labels = (mean_predictions > 0.5).astype('int32')

# Printing the classification report and confusion matrix
print(classification_report(labeltest, predicted_labels))
print(confusion_matrix(labeltest, predicted_labels))

# To measure uncertainty, you can also look at the standard deviation of predictions across the T runs:
std_dev = np.std(predictions, axis=0)
print("Uncertainty (standard deviation of predictions):")
print(std_dev)

'''
# Input
Testimage = cv.imread('./Normal/Normal-1.png', cv.IMREAD_GRAYSCALE)
Testimage = cv.resize(Testimage, (imagesize, imagesize))  # Resize if necessary
Testimage = np.array(Testimage)
Testimage = (Testimage.astype('float32')) / 255  # Normalize to [0, 1] range
Testimage = Testimage.reshape(1, imagesize, imagesize, 1)  # Reshape for model input
print(Testimage.shape)
# Perform prediction
result = cnn_bayesian.predict(Testimage, batch_size = 1)
print("Result:", "TB" if result[0][0] > 0.5 else "Normal")
print("Prediction Probability:", result[0][0])
'''
import tensorflow as tf

# Function for predicting on a single image and calculating uncertainty
def predict_with_uncertainty(model, image_path, T=10):
    # Load and preprocess the image
    Testimage = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    Testimage = cv.resize(Testimage, (imagesize, imagesize))  # Resize to match model input
    Testimage = (Testimage.astype('float32')) / 255  # Normalize
    Testimage = Testimage.reshape(1, imagesize, imagesize, 1)  # Reshape for model input
    
    # Force static shape
    Testimage = tf.ensure_shape(Testimage, [1, imagesize, imagesize, 1])
    
    print("Testimage shape:", Testimage.shape)  # Debug shape

    # Perform T stochastic forward passes to estimate uncertainty
    predictions = []
    for _ in range(T):
        pred = model.predict(Testimage, batch_size=1)
        predictions.append(pred)
    
    predictions = np.stack(predictions, axis=0)
    
    # Averaging predictions to get the final probability score
    mean_prediction = np.mean(predictions, axis=0)
    std_dev = np.std(predictions, axis=0)  # Standard deviation for uncertainty measure

    # Interpret final result based on mean prediction
    final_result = "TB" if mean_prediction[0][0] > 0.5 else "Normal"
    prediction_probability = mean_prediction[0][0]
    
    # Print results
    print("Prediction:", final_result)
    print("Prediction Probability:", prediction_probability)
    print("Uncertainty (Standard Deviation):", std_dev[0][0])


