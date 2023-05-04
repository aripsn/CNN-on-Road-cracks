# -*- coding: utf-8 -*-
"""
Created on Mon May 1 21:02:24 2023

@author: Anur
"""
#Import all necessary library
import os
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import seaborn as sns
from sklearn.metrics import confusion_matrix


dir = 'F:/Semester 1 Spring 2023/Machine Learning for CVEN/Project_6'
os.chdir(dir)


# Define the image size and number of channels
image_size = (300, 300)
image_channel = 3 #RGB

# Define the directories for the Cracks and noCracks images
Cracks_dir = "F:/Semester 1 Spring 2023/Machine Learning for CVEN/Project_6/Crack"
noCracks_dir = "F:/Semester 1 Spring 2023/Machine Learning for CVEN/Project_6/NoCrack"

# Preprocess each image
def preprocess_image(file_dir):
    # Load the image using PIL
    image = Image.open(file_dir)
    
    # Resize the image
    image = image.resize(image_size)
    
    # Convert the image to a numpy array
    image_array = np.array(image)
    
    # # Normalize pixel values to be between 0 and 1
    image_array = image_array / 255.0
    
    return image_array

# Load and preprocess the Cracks images
Cracks_images = []
for file_name in os.listdir(Cracks_dir):
    if file_name.endswith(".JPG"):
        file_dir = os.path.join(Cracks_dir, file_name)
        image_array = preprocess_image(file_dir)
        Cracks_images.append(image_array)
        
Cracks_images = np.array(Cracks_images)

# Load and preprocess the noCracks images
noCracks_images = []
for file_name in os.listdir(noCracks_dir):
    if file_name.endswith(".JPG"):
        file_dir = os.path.join(noCracks_dir, file_name)
        image_array = preprocess_image(file_dir)
        noCracks_images.append(image_array)
        
noCracks_images = np.array(noCracks_images)

# Combine all images into a single array
X = np.concatenate((Cracks_images, noCracks_images), axis=0)

# Create the target labels (1 for Cracks images, 0 for noCracks images)
y = np.concatenate((np.ones(len(Cracks_images)), np.zeros(len(noCracks_images))))

# Random the data
permutation = np.random.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Print the shapes of the data splits
print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)
print("Testing data shape:", X_test.shape)

# Create the convolutional base
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(300, 300, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

# Model evaluation using the test dataset
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

#Plot Loss with epochs
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.head(5))
print(hist.tail(5))

plt.figure(figsize=(6,4))

plt.plot(hist['epoch'], hist['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.show()        



# Make predictions on the testing set
y_pred = model.predict(X_test)
y_pred_classes = np.round(y_pred)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Define the class names
class_names = ['NoCrack', 'Crack']

# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(cm, annot=True, cmap='Greens', fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

