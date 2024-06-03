import os
import cv2
import numpy as np
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input

# Load the saved model
loaded_model = load_model("carton_box_model_vgg16_with_augmentation.h5")

# Function to load and preprocess test images
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# Set the path to the test image folder
test_folder = "Testing"

# Iterate over the images in the test folder
for filename in os.listdir(test_folder):
    image_path = os.path.join(test_folder, filename)
    
    # Preprocess the image
    img = preprocess_image(image_path)
    
    # Perform prediction using the loaded model
    prediction = loaded_model.predict(img)
    
    # Get the predicted label
    if prediction[0] > 0.5:
        label = "Defected"
    else:
        label = "Non-Defected"
    
    # Print the prediction result
    print(f"Image: {filename}, Prediction: {label}")