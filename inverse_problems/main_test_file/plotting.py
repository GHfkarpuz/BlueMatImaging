import numpy as np

import sys
sys.path.append("/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/wrappers/python")

import cv2
import numpy as np
import gc
import matplotlib.pyplot as plt
import os
from wrapper import FlexBoxSolver # python wrapper
import errorFunctions as ef
from itertools import product


for i in range(len(frames)):
    result_1_u = frames[i].reshape(inputDimension, order="F")
    magnitude_u_new = np.sqrt(result_1_u**2)

    umax1 = np.max(np.abs(result_1_u))
    umax_mag = np.percentile(magnitude_u, 99)

    plt.figure(figsize=(7,7))

    plt.title(f"magnitude of new u_{i+1}")
    plt.imshow(magnitude_u_new, cmap="gray", vmin=0, vmax=umax_mag)
    plt.axis("off")

    file_path = os.path.join(folder_name_main, f"subtractedAverageFrames{name},{i}.png")
    plt.savefig(file_path, dpi=200, bbox_inches="tight")

    plt.close()
