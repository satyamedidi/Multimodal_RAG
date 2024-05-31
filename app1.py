import streamlit as st
import torch
from PIL import Image
import os
import clip
from torchvision import transforms

@st.cache(allow_output_mutation=True)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

# Load images and descriptions
img_dir = './'
description_file = './descriptions.txt'
images = []
descriptions = []

@st.cache
def load_images_and_descriptions():
    with open(description_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 2:
                img_path, description = parts
                full_img_path = os.path.join(img_dir, img_path)
                if os.path.exists(full_img_path):
                    images.append(full_img_path)
                    descriptions.append(description)

    # Preprocess and encode all images
    processed_images = [preprocess(Image.open(img).convert('RGB')).unsqueeze(0).to(device) for img in images]
    image_features_list = [model.encode_image(img_tensor) for img_tensor in processed_images]
    return images, descriptions, image_features_list

images, descriptions, image_features_list = load_images_and_descriptions()

# Streamlit interface
st.title('Multimodal AI')

question = st.text_input('Ask a question:')
if question:
    text_tokens = clip.tokenize([question]).to(device)
    text_features = model.encode_text(text_tokens)

    # Calculate similarity with each image
    similarities = [(text_features @ img_features.T).squeeze().item() for img_features in image_features_list]

    # Find the best matching image
    best_idx = similarities.index(max(similarities))
    best_image_path = images[best_idx]

    # Display the image
    image = Image.open(best_image_path)
    st.image(image, caption=descriptions[best_idx])
