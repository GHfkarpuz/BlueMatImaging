import cv2
import imageio.v2 as imageio
import os
import numpy as np
import matplotlib.pyplot as plt

mask = np.load("mask.npy")

mask_uint8 = mask.astype(np.uint8)

kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
mask_dilate= mask
for i in range(5):
    mask_eroded = cv2.erode(mask_dilate, kernel_erode, iterations=1)
    mask_dilate = cv2.dilate(mask_eroded, kernel_dilate, iterations=1)
#mask_eroded = cv2.erode(mask, kernel_erode, iterations=20)
#kernel_dilate_post = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
#mask_dilate = cv2.dilate(mask_dilate, kernel_erode, iterations=4)
mask_out=mask_dilate
mask = mask_out.astype(np.float32)

magnitude_u_new = np.sqrt(mask**2)
umax_mag = np.max(magnitude_u_new)
#umax_mag = np.percentile(mask, 99)
plt.figure(figsize=(7,7))

plt.title(f"magnitude of new u")
plt.imshow(magnitude_u_new, cmap="gray", vmin=0, vmax=umax_mag)
plt.axis("off")

output_dir = "Go"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

file_path = os.path.join(output_dir, "dilationErosion.png")
plt.savefig(file_path, dpi=200, bbox_inches="tight")
plt.close()