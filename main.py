import numpy as np
import cv2
import torch
import open3d as o3d
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
midas.eval()

device = torch.device("cpu")
midas.to(device)

transform = Compose([
    Resize((384, 384)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return img


def estimate_depth(image):
    img_input = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = midas(img_input)

    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=(image.height, image.width),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    return depth
