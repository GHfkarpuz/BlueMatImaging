import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from deepinv.models import DRUNet

# === FIX 1: cuDNN-Fehler auf Clustern umgehen ===
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False  # Zwingt PyTorch auf stabile CUDA-Kernels

VIDEO_PATH = "../inverse_problems/main_test_file/media3.avi"
OUTPUT_PATH = "media3_denoised.mp4"
N_FRAMES = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

frames = []
for _ in range(N_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
cap.release()

os.makedirs("frames", exist_ok=True)

sigma = torch.tensor([0.5], device=device).float()
denoiser = DRUNet(in_channels=1, out_channels=1, pretrained="download", device=device)
denoiser.eval()

height, width = frames[0].shape
writer = cv2.VideoWriter(
    OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), isColor=False
)

with torch.no_grad():
    for idx, frame in enumerate(frames):
        x = torch.from_numpy(frame).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)
        
        # === FIX 2: Dynamic Padding für Vielfaches von 16 ===
        h, w = x.shape[2], x.shape[3]
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect') if (pad_h > 0 or pad_w > 0) else x
        
        # Inferenz
        x_denoised = denoiser(x_padded, sigma)
        
        # Padding abschneiden
        x_denoised = x_denoised[:, :, :h, :w]
        
        # Output verarbeiten
        out = x_denoised.squeeze().clamp(0, 1).mul(255.0).round().byte().cpu().numpy()
        writer.write(out)

writer.release()
print("Denoising erfolgreich ausgeführt!")