import cv2
import os
import numpy as np


image_folder = '/data/dust/user/karpuzfa/TEMSequence'
fps = 10

images = sorted([f for f in os.listdir(image_folder) if f.endswith('.tif')])

# Erstes Bild laden um Größe zu bestimmen
first = cv2.imread(os.path.join(image_folder, images[0]), cv2.IMREAD_ANYDEPTH)
h, w = first.shape[:2]

out = cv2.VideoWriter('/data/dust/user/karpuzfa/TEMSequenceLong.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (w, h))

for filename in images:
    filepath = os.path.join(image_folder, filename)
    img = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH)
    if img is None:
        print(f"Warnung: konnte {filename} nicht lesen – überspringe.")
        continue

    # 16-bit → 8-bit normalisieren
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

    # Graustufen → BGR (VideoWriter braucht 3 Kanäle)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h))

    out.write(img)

out.release()
print("Video erfolgreich erstellt!")