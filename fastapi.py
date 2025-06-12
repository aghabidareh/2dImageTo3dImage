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


def create_point_cloud(rgb_image: Image.Image, depth_map: np.ndarray, focal_length: float = 500.0):
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

app.get('/')
async def root():
    return {'message': 'Welcome to Image to 3D Point Cloud!'}


@app.post("/convert-to-3d/", response_class=FileResponse)
async def convert_to_3d(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image (e.g., JPG, PNG)")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        depth_map = estimate_depth(image)

        point_cloud = create_point_cloud(image, depth_map)

        temp_file = os.path.join(tempfile.gettempdir(), f"point_cloud_{uuid.uuid4()}.ply")
        o3d.io.write_point_cloud(temp_file, point_cloud)

        return FileResponse(
            temp_file,
            media_type="application/octet-stream",
            filename="point_cloud.ply",
            background=BackgroundTask(cleanup, temp_file)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


