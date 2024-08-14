import streamlit as st
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image

page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background-image:  url("https://images.unsplash.com/photo-1655720828083-96a45b0a48b5?q=80&w=1932&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
background-size" cover;
}
</style>
"""
# Inject the CSS background style into Streamlit
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)  # Adjusting the final layer for 3 classes
model.load_state_dict(torch.load('model_checkpoint.pth', map_location=device))
model.to(device)
model.eval()

class_names = ['normal', 'viral', 'covid']

# Function to preprocess the image using OpenCV
def preprocess_image_cv2(image_cv2):
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image_cv2 = cv2.resize(image_cv2, (224, 224))  # Resize to 224x224
    image_cv2 = image_cv2.astype(np.float32) / 255.0  # Normalize
    image_cv2 = (image_cv2 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # Normalize
    image_cv2 = image_cv2.transpose((2, 0, 1))  # Transpose for PyTorch format
    image_tensor = torch.tensor(image_cv2, dtype=torch.float32).unsqueeze(0).to(device)  # Convert to tensor
    return image_tensor

# Function to make predictions
def predict(image_cv2):
    image_tensor = preprocess_image_cv2(image_cv2)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence_scores = torch.softmax(outputs, dim=1)[0]
    return class_names[predicted.item()], confidence_scores

# Streamlit UI

st.markdown(
    """
    <style>
    .title-text {
        color: #FFFFFF; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Displaying the title with the specified CSS class
st.markdown('<h1 class="title-text">COVID-19 - Viral Pneumonia Detection App</h1>', unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload a chest X-ray image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display the uploaded image
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button('Predict'):
        # Make prediction when button is clicked
        prediction, confidence_scores = predict(image)
        st.write(f'Prediction: {prediction}')
        st.write(f'Accuracy:')
        for i, class_name in enumerate(class_names):
            st.write(f'{class_name}: {confidence_scores[i]*100:.2f}%')

