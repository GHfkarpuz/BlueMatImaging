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

frames = np.load("frames.npy")
frames_full = np.load("frames_full.npy")
framesSubtractedBackground = np.load("framesSubtractedBackground.npy")
framesForDenoising = np.load("framesForDenoising.npy")


output_dir = "segmentation_last"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

inputDimension = frames[0].shape
"""
for i in range(len(frames)):
    frame_uint8= frames[i].astype(np.uint8)
    denoised = cv2.fastNlMeansDenoising(src=frame_uint8, dst=None, h=15, templateWindowSize=7,searchWindowSize=21)
    frames[i] = denoised.astype(np.float32)

for i in range(len(framesForDenoising)):
    frame_uint8= framesForDenoising[i].astype(np.uint8)
    denoised = cv2.fastNlMeansDenoising(src=frame_uint8, dst=None, h=15, templateWindowSize=7,searchWindowSize=21)
    framesForDenoising[i] = denoised.astype(np.float32)
"""
frame_ =framesSubtractedBackground
for j in range(len(framesSubtractedBackground)):
    for i in range(len(framesForDenoising)):
        vari = 20.0
        frame_[j][np.abs(framesForDenoising[i]-frames[0])<vari] = 0.0
        frame_[j][frame_[j]<140] = 0.0
        frame_[j][framesForDenoising[i]>180.0]=0.0

        if(i==len(framesForDenoising)-1):
            magnitude_u_new = np.sqrt(frame_[j]**2)
            
            umax_mag = np.max(magnitude_u_new)
            #umax_mag = np.percentile(mask, 99)
            plt.figure(figsize=(7,7))

            plt.title(f"magnitude of new u")
            plt.imshow(magnitude_u_new, cmap="gray", vmin=0, vmax=umax_mag)
            plt.axis("off")

            file_path = os.path.join(output_dir, f"{j}subtractBackgroundFrame{i}.png")
            plt.savefig(file_path, dpi=200, bbox_inches="tight")

            plt.close()


num_conv_it=1
mask= frame_
mask_copy = mask.copy()
frame_[frame_>0.0]=255.0
for m in range(len(mask)):
    for n in range(num_conv_it):
        conv_width = 5
        percent_px = 0.2
        for i in range(conv_width,inputDimension[0]-conv_width):
            for j in range(conv_width,inputDimension[1]-conv_width):
                if(mask[m][i][j]!=0.0):
                    sum = 0.0
                    for k in range(-conv_width,conv_width,1):
                        for l in range(-conv_width,conv_width,1):
                            sum+=mask_copy[m][i+k][j+l]
                    if(sum<(((2*conv_width)**2)*255.0*percent_px)):
                        mask[m][i][j]=0.0
        
        #mask_uint8 = mask.astype(np.uint8)
        if(num_conv_it%1==0):
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
            mask_eroded= mask[m]
            for i in range(1):
                mask_dilate = cv2.dilate(mask_eroded, kernel_dilate, iterations=1)
                mask_eroded = cv2.erode(mask_dilate, kernel_erode, iterations=1)
            #mask_eroded = cv2.erode(mask, kernel_erode, iterations=20)
            #kernel_dilate_post = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
            #mask_dilate = cv2.dilate(mask_dilate, kernel_erode, iterations=4)
            mask[m]=mask_eroded
        
        magnitude_u_new = np.sqrt(mask[m]**2)
        
        umax_mag = np.max(magnitude_u_new)
        #umax_mag = np.percentile(mask, 99)
        plt.figure(figsize=(7,7))

        plt.title(f"magnitude of new u")
        plt.imshow(magnitude_u_new, cmap="gray", vmin=0, vmax=umax_mag)
        plt.axis("off")

        file_path = os.path.join(output_dir, f"{m}differenceFramesPostDenoisedConvWidth{conv_width}Perc{percent_px},{n}.png")
        plt.savefig(file_path, dpi=200, bbox_inches="tight")

        file_path = os.path.join(output_dir, f"{m}dilationErosion.png")
        plt.savefig(file_path, dpi=200, bbox_inches="tight")
        plt.close()


for m in range(len(frame_)):
    for i in range(conv_width):
        for j in range(inputDimension[1]):
            mask[m][i][j]=0.0
            mask[m][inputDimension[0]-1-i][j]=0.0
for m in range(len(frame_)):
    for j in range(conv_width):
        for i in range(inputDimension[0]):
            mask[m][i][j]=0.0
            mask[m][i][inputDimension[1]-1-j]=0.0

diff = 0.5*(mask[0]+mask[len(framesSubtractedBackground)-1])

magnitude_u_new = np.sqrt(diff**2)
umax_mag = np.max(magnitude_u_new)
#umax_mag = np.percentile(mask, 99)
plt.figure(figsize=(7,7))

plt.title(f"magnitude of new u")
plt.imshow(magnitude_u_new, cmap="gray", vmin=0, vmax=umax_mag)
plt.axis("off")

file_path = os.path.join(output_dir, f"DiffDifferenceFramesPostDenoisedConvWidth{conv_width}Perc{percent_px},{n}.png")
plt.savefig(file_path, dpi=200, bbox_inches="tight")

file_path = os.path.join(output_dir, f"{m}DifffinalConvWidth{conv_width}Perc{percent_px}.png")
plt.savefig(file_path, dpi=200, bbox_inches="tight")
plt.close()

kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
mask[m] = cv2.dilate(mask[m], kernel_dilate, iterations=3)


magnitude_u_new = np.sqrt(mask[m]**2)
umax_mag = np.max(magnitude_u_new)
#umax_mag = np.percentile(mask, 99)
plt.figure(figsize=(7,7))

plt.title(f"magnitude of new u")
plt.imshow(magnitude_u_new, cmap="gray", vmin=0, vmax=umax_mag)
plt.axis("off")

file_path = os.path.join(output_dir, f"DiffDifferenceFramesPostDenoisedConvWidth{conv_width}Perc{percent_px},{n}.png")
plt.savefig(file_path, dpi=200, bbox_inches="tight")

file_path = os.path.join(output_dir, f"DifffinalConvWidthPlusDilation{conv_width}Perc{percent_px}.png")
plt.savefig(file_path, dpi=200, bbox_inches="tight")
plt.close()


mask_uint8 = mask[m].astype(np.uint8)

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
        f"für frame Summe Segment gefunden -> Oben-Links: ({x1}, {y1}), Unten-Rechts: ({x2}, {y2})"
    )

for m in range(len(frame_)):
    magnitude_u_new = np.sqrt(mask[m]**2)
    umax_mag = np.max(magnitude_u_new)
    #umax_mag = np.percentile(mask, 99)
    plt.figure(figsize=(7,7))

    plt.title(f"magnitude of new u")
    plt.imshow(magnitude_u_new, cmap="gray", vmin=0, vmax=umax_mag)
    plt.axis("off")

    file_path = os.path.join(output_dir, f"{m}differenceFramesPostDenoisedConvWidth{conv_width}Perc{percent_px},{n}.png")
    plt.savefig(file_path, dpi=200, bbox_inches="tight")

    file_path = os.path.join(output_dir, f"{m}finalConvWidth{conv_width}Perc{percent_px}.png")
    plt.savefig(file_path, dpi=200, bbox_inches="tight")
    plt.close()

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask[m] = cv2.dilate(mask[m], kernel_dilate, iterations=2)


    magnitude_u_new = np.sqrt(mask[m]**2)
    umax_mag = np.max(magnitude_u_new)
    #umax_mag = np.percentile(mask, 99)
    plt.figure(figsize=(7,7))

    plt.title(f"magnitude of new u")
    plt.imshow(magnitude_u_new, cmap="gray", vmin=0, vmax=umax_mag)
    plt.axis("off")

    file_path = os.path.join(output_dir, f"{m}differenceFramesPostDenoisedConvWidth{conv_width}Perc{percent_px},{n}.png")
    plt.savefig(file_path, dpi=200, bbox_inches="tight")

    file_path = os.path.join(output_dir, f"{m}finalConvWidthPlusDilation{conv_width}Perc{percent_px}.png")
    plt.savefig(file_path, dpi=200, bbox_inches="tight")
    plt.close()


    mask_uint8 = mask[m].astype(np.uint8)

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
            f"für frame {m} Segment gefunden -> Oben-Links: ({x1}, {y1}), Unten-Rechts: ({x2}, {y2})"
        )