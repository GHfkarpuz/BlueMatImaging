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

motionField_0_full = np.load("motionField_0_full.npy")
motionField_1_full = np.load("motionField_1_full.npy")

reconstructed_image_full = np.load("reconstructed_image_full.npy")

motionField_0_frames = np.split(motionField_0_full, 21)
motionField_1_frames = np.split(motionField_1_full, 21)

reconstructed_image_full_frames = np.split(reconstructed_image_full, 21)

output_dir = "adsf"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

inputDimension_full = [819, 410]
for k in range(10):
    for j in range(21-1):
        result_1_v = motionField_0_frames[j].reshape(inputDimension_full, order="F")
        result_2_v = motionField_1_frames[j].reshape(inputDimension_full, order="F")
        result_1_u = reconstructed_image_full_frames[j].reshape(inputDimension_full, order="F")

        #frame_start = j * nPx
        #frame_end = (j + 1) * nPx
        #result_1_u = reconstructed_image[frame_start:frame_end].reshape(inputDimension, order="F")

        magnitude_v = np.sqrt(result_1_v**2 + result_2_v**2)
        #vmax1 = np.max(np.abs(result_1_v))
        #vmax2 = np.max(np.abs(result_2_v))
        vmax_mag = np.percentile(magnitude_v, 99.9-0.01*k)
        vmax1 = np.percentile(result_1_v, 99.99-0.01*k)
        vmax2 = np.percentile(result_2_v, 99.99-0.01*k)
        
        magnitude_u = np.sqrt(result_1_u**2)
        umax1 = np.max(np.abs(result_1_u))
        umax_mag = np.percentile(magnitude_u, 99)

        print(f"vmax1 for i = {j} is ", vmax1)
        print(f"vmax2 for i = {j} is ", vmax2)
        
        print(f"umax1 for i = {j} is ", umax1)          

        plt.figure(figsize=(7,7))
        
        plt.subplot(2,2,1)
        plt.title("magnitude of u")
        plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax1)
        #plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax_mag)
        plt.axis("off")
        
        plt.subplot(2,2,3)
        plt.title("v_x")
        plt.imshow(result_2_v, cmap="seismic", vmin=-vmax2, vmax=vmax2)
        plt.axis("off")

        plt.subplot(2,2,4)
        plt.title("v_y")
        plt.imshow(result_1_v, cmap="seismic", vmin=-vmax1, vmax=vmax1)
        plt.axis("off")

        plt.subplot(2,2,2)
        plt.title("magnitude of v")
        plt.imshow(magnitude_v, cmap="gray", vmin=0, vmax=vmax_mag)
        plt.axis("off")
        
        #plt.savefig(f"result{dataTerm_norm_list[dataTerm_norm]}{reg_u_norm_list[reg_u_norm]}{motionTerm_u_norm_list[motionTerm_u_norm]}{motionTerm_u_list[motionTerm_u]}{motionTerm_v_norm_list[motionTerm_v_norm]}{motionTerm_v_list[motionTerm_v]}{reg_v_norm_list[reg_v_norm]}{i+1},{j+1}.png", dpi=200, bbox_inches="tight")
        file_path = os.path.join(output_dir, f"Converged{j+1},percentile{99-k}.png")
        plt.savefig(file_path, dpi=200, bbox_inches="tight")
        plt.close()