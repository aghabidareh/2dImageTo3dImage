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


def create_point_cloud(rgb_image, depth_map, focal_length=500.0):
    rgb = np.array(rgb_image)

    height, width = depth_map.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    depth_map = np.clip(depth_map, 1e-6, np.max(depth_map))

    z = depth_map
    x = (x - width / 2) * z / focal_length
    y = (y - height / 2) * z / focal_length

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def main():
    image_path = "image.jpg"
    image = load_image(image_path)

    depth_map = estimate_depth(image)

    point_cloud = create_point_cloud(image, depth_map)

    output_path = "output_point_cloud.ply"
    o3d.io.write_point_cloud(output_path, point_cloud)
    print(f"Point cloud saved to {output_path}")
