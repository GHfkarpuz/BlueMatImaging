import sys
sys.path.append("/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/wrappers/python")


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from wrapper import FlexBoxSolver # python wrapper


# parameters
weight_data = 1.0
weight_tv = 0.1


# open image and gray value
img1 = Image.open("frame10.bmp").convert("RGB")
img2 = Image.open("frame11.bmp").convert("RGB")
img1_np = np.array(img1).astype(np.float32)
img2_np = np.array(img2).astype(np.float32)
print(img1_np)
print(img2_np)
print(img2_np - img1_np)

# initialise solver
solver = FlexBoxSolver(maxIt=2000, tol=1e-6, verbose=1)

nPx=img1_np.size()

# add primal variable
solver.add_primal(np.zeros(nPx))


