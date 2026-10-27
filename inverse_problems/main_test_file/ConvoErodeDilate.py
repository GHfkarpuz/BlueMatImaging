import cv2
import imageio.v2 as imageio
import os
import numpy as np
import matplotlib.pyplot as plt

"""
from skimage.metrics import structural_similarity as ssim

# score liegt zwischen -1.0 und 1.0 (1.0 = identisch)
score, diff_map = ssim(frame1, frame2, full=True)
"""

mask = np.load("beforeConv.npy")


output_dir = "GoWithConvoNew"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

inputDimension = mask.shape

num_conv_it=4
for n in range(num_conv_it):
    conv_width = 20
    percent_px = 0.17
    mask_copy=mask.copy()
    for i in range(conv_width,inputDimension[0]-conv_width):
        for j in range(conv_width,inputDimension[1]-conv_width):
            if(mask[i][j]!=0.0):
                sum = 0.0
                for k in range(-conv_width,conv_width,1):
                    for l in range(-conv_width,conv_width,1):
                        sum+=mask_copy[i+k][j+l]
                if(sum<(((2*conv_width)**2)*255.0*percent_px)):
                    mask[i][j]=0.0
    
    #mask_uint8 = mask.astype(np.uint8)
    if(num_conv_it%1==0):
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        mask_dilate= mask
        for i in range(1):
            mask_eroded = cv2.erode(mask_dilate, kernel_erode, iterations=3)
            mask_dilate = cv2.dilate(mask_eroded, kernel_dilate, iterations=3)
        #mask_eroded = cv2.erode(mask, kernel_erode, iterations=20)
        #kernel_dilate_post = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
        #mask_dilate = cv2.dilate(mask_dilate, kernel_erode, iterations=4)
        mask_out=mask_dilate
    
    magnitude_u_new = np.sqrt(mask**2)
    
    umax_mag = np.max(magnitude_u_new)
    #umax_mag = np.percentile(mask, 99)
    plt.figure(figsize=(7,7))

    plt.title(f"magnitude of new u")
    plt.imshow(magnitude_u_new, cmap="gray", vmin=0, vmax=umax_mag)
    plt.axis("off")

    file_path = os.path.join(output_dir, f"differenceFramesPostDenoisedConvWidth{conv_width}Perc{percent_px},{n}.png")
    plt.savefig(file_path, dpi=200, bbox_inches="tight")

    file_path = os.path.join(output_dir, "dilationErosion.png")
    plt.savefig(file_path, dpi=200, bbox_inches="tight")
    plt.close()

for i in range(conv_width):
    for j in range(inputDimension[1]):
        mask[i][j]=0.0
        mask[inputDimension[0]-1-i][j]=0.0
for j in range(conv_width):
    for i in range(inputDimension[0]):
        mask[i][j]=0.0
        mask[i][inputDimension[1]-1-j]=0.0
        
magnitude_u_new = np.sqrt(mask**2)
umax_mag = np.max(magnitude_u_new)
#umax_mag = np.percentile(mask, 99)
plt.figure(figsize=(7,7))

plt.title(f"magnitude of new u")
plt.imshow(magnitude_u_new, cmap="gray", vmin=0, vmax=umax_mag)
plt.axis("off")

file_path = os.path.join(output_dir, f"differenceFramesPostDenoisedConvWidth{conv_width}Perc{percent_px},{n}.png")
plt.savefig(file_path, dpi=200, bbox_inches="tight")

file_path = os.path.join(output_dir, f"finalConvWidth{conv_width}Perc{percent_px}.png")
plt.savefig(file_path, dpi=200, bbox_inches="tight")
plt.close()

kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
mask = cv2.dilate(mask, kernel_dilate, iterations=3)


magnitude_u_new = np.sqrt(mask**2)
umax_mag = np.max(magnitude_u_new)
#umax_mag = np.percentile(mask, 99)
plt.figure(figsize=(7,7))

plt.title(f"magnitude of new u")
plt.imshow(magnitude_u_new, cmap="gray", vmin=0, vmax=umax_mag)
plt.axis("off")

file_path = os.path.join(output_dir, f"differenceFramesPostDenoisedConvWidth{conv_width}Perc{percent_px},{n}.png")
plt.savefig(file_path, dpi=200, bbox_inches="tight")

file_path = os.path.join(output_dir, f"finalConvWidthPlusDilation{conv_width}Perc{percent_px}.png")
plt.savefig(file_path, dpi=200, bbox_inches="tight")
plt.close()


mask_uint8 = mask.astype(np.uint8)

# find contours
contours, _ = cv2.findContours(
    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# list for the rectangles of the segments
bounding_boxes = []

for cnt in contours:
    # x1, y1 = obere linke Ecke | w = Breite | h = Höhe
    x1, y1, w, h = cv2.boundingRect(cnt)

    # Untere rechte Ecke berechnen
    x2 = x1 + w
    y2 = y1 + h

    bounding_boxes.append((x1, x2, y1, y2))
    print(
        f"Segment gefunden -> Oben-Links: ({x1}, {y1}), Unten-Rechts: ({x2}, {y2})"
    )