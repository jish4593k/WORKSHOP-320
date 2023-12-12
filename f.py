from PIL import Image, ImageTk
import os
import random
import argparse
import sys
import binascii
import tkinter as tk
from torchvision import models, transforms
import torch

image_formats = ['.jpg', '.jpeg', '.png', '.tif', '.bmp', 'gif', 'tiff']
OUTPUT_FORMAT = '.png'
SAVE_IMAGE = True


model = models.resnet50(pretrained=True)
model.eval()
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_image(image_path):
    input_image = Image.open(image_path)
    input_image = input_image.resize((224, 224))  # Resize to match model input size
    input_tensor = transform(input_image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    _, predicted_idx = torch.max(output, 1)
    return str(predicted_idx.item())



def get_all_images_from_the_input_dir(input_dir):
    images = []
    for file in os.listdir(input_dir):
        filepath = os.path.join(input_dir, file)
        if os.path.isfile(filepath):
            if os.path.splitext(filepath)[1].lower() in image_formats:
                img = Image.open(filepath)
                images.append(img)
    return images


def main():
    images = get_all_images_from_the_input_dir(INPUT_DIR)
    spliced_image = splice_images(images, MIN_STRIPES, MAX_STRIPES, orientation=ORIENTATION)
    spliced_image.show()

  
    classification_result = classify_image("path_to_spliced_image.png")

    if SAVE_IMAGE:
        save_image(spliced_image)
    
    
    update_gui("path_to_spliced_image.png", classification_result)

if __name__ == '__main__':
    
