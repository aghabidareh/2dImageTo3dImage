from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import numpy as np
import cv2
import torch
import open3d as o3d
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import os
import tempfile
import uuid

app = FastAPI(title="Image to 3D Point Cloud API", description="API for Image to 3D Point Cloud", version="1.0")

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


def estimate_depth(image: Image.Image):
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
