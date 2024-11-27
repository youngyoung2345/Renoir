# Import model 

import torch
import torch.nn.functional as F

from model.Backbone import *
from model.Preprocessing import *

def import_model():
    Feature_Extractor = VGG19()
    state_dict = torch.load('model/VGG19_weight.pth', map_location=torch.device('cpu'))
    Feature_Extractor.load_state_dict(state_dict)

    Feature_Extractor.eval()

    return Feature_Extractor

def calculate_similarity(Feature_Extractor, image1, image2):
    image1 = preprocess(image1).unsqueeze(0)
    image2 = preprocess(image2).unsqueeze(0)

    feature_1 = Feature_Extractor(image1)
    feature_1 = feature_1.view(feature_1.size(0), -1)
    feature_2 = Feature_Extractor(image2)
    feature_2 = feature_2.view(feature_2.size(0), -1)

    similarity = F.cosine_similarity(feature_1, feature_2)

    return similarity


import base64
from io import BytesIO

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def upload_image(index):
    uploaded_file = st.file_uploader(f"Upload Image_{index}", type=["jpg", "png", "jpeg"], key=f"file_{index}", label_visibility="collapsed")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_str = encode_image(image)  
        st.markdown(f'<div class="image-uploader"><img src="data:image/png;base64,{img_str}"></div>', unsafe_allow_html=True)

        return image
    return None

# Make pront

import streamlit as st

from PIL import Image

def main():
    

    st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Dancing+Script&display=swap');

    html, body {
        margin: 0;
        height: 80vh;  
        background: linear-gradient(to bottom, #0D3383, #ffffff);  
        color: white;
        display: flex;
        justify-content: center;  
        align-items: center;  
    }

    .title-container {
        text-align: center;
        padding: 50px;
        background: linear-gradient(to bottom, #0D3383, #ffffff);  
        border-radius: 15px;
        width: 100%;  
        margin-bottom: 50px;
    }

    .centered-title {
        font-size: 5em;
        font-weight: bolder;
        font-family: 'Dancing Script', cursive;
        color: white;
        margin: 0;
    }

    .sub-title {
        font-size: 2em;
        font-weight:lighter;
        font-family: 'Dancing Script', cursive;
        color: white;
        margin-top: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.markdown('<div class="title-container"><h1 class="centered-title">Renoir</h1><h2 class="sub-title">We recognize the world you create</h2></div>', unsafe_allow_html=True)
    st.markdown("""
    <style>
    .image-uploader {
        width: 300px;
        height: 300px;
        background-color: #f0f0f0;
        border-radius: 20px;
        border: 2px dashed #0D3383;
        display: flex;
        justify-content: center;
        align-items: center;
        cursor: pointer;
    }

    .image-uploader img {
        object-fit: cover;
        width: 100%;
        height: 100%;
        border-radius: 15px;
    }

    .image-uploader:hover {
        background-color: #e0e0e0;
    }

    .uploaded-images-container {
        display: flex;
        gap: 20px;  
        border-radius: 15px;
    }
                
    .similarity-result {
        margin-top: 20px;
        font-size: 1.5em;
        text-align: center;
    }

    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)  

    with col1:
        image1 = upload_image(1)

    with col2:
        image2 = upload_image(2)

    if image1 is not None and image2 is not None:
        Feature_Extractor = import_model()
        similarity = calculate_similarity(Feature_Extractor, image1, image2).item()

        st.markdown(
            f"""
            <div class="similarity-result" style="
                font-size: 2em;
                font-weight: lighter;
                font-family: 'Dancing Script', cursive;
                color : #0D3383;
            ">
                <b>Similarity: {similarity * 100:.2f}%</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

if __name__ == "__main__":
    main()

