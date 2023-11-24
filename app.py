import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import style_transfer, cnn, cnn_normalization_mean, cnn_normalization_std

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("Neural Style Transfer App")

def load_image(image_file):
    img = Image.open(image_file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor.to(device)

content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

if content_file and style_file:
    content_img = load_image(content_file)
    style_img = load_image(style_file)

    st.write("Content Image:")
    st.image(content_file, use_column_width=True)
    st.write("Style Image:")
    st.image(style_file, use_column_width=True)

    input_img = content_img.clone()

    # Perform style transfer
    with torch.no_grad():
        output_img = style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img)

    # Convert the output tensor to PIL Image for display
    to_pil = transforms.ToPILImage()
    output_img = to_pil(output_img.squeeze(0).cpu())

    st.write("Stylized Image:")
    st.image(output_img, use_column_width=True)
