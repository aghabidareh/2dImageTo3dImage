import numpy as np
import cv2
import torch
import open3d as o3d
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
midas.eval()
