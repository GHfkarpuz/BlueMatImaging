import sys
sys.path.append("/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/wrappers/python")


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from wrapper import FlexBoxSolver # python wrapper


# test file for the L2-L2 optical flow model: argmin_{v\in \mathbb{R}^{inputDimension*nPx}} \frac[weight_data}{2}*||u_t + grad(u)*v||_{2}^{2} + \frac{weight_L2}{2}*\sum_{i=1}^{inputDimension} ||grad(v_i)||_{2}^{2}


# parameters
weight_data = 1.0
weight_L2 = 0.5
timeStep = 1.0


# images
img1 = Image.open("frame10.bmp").convert("L")
img2 = Image.open("frame11.bmp").convert("L")

img1_np = np.array(img1).astype(np.float32)
img2_np = np.array(img2).astype(np.float32)




print("The 2-distance is:", np.linalg.norm(img2_np-img1_np))
# dimensions
inputDimension = list(img1_np.shape)
nPx = img1_np.size


# flatten image for operator
img1_vec = img1_np.flatten(order="F")

# initialise solver
solver = FlexBoxSolver(maxIt=10000, tol=1e-6, verbose=1)

# primal variables
dim = len(inputDimension)

print(inputDimension)


# add a primal variable for each direction of the motion field
for i in range(dim):
    solver.add_primal(np.zeros(nPx))

corresponding_primals = [i for i in range(dim)]

# operator. It has to be extended for the 3D-case.
opticalFlowOp1 = {
    "type": "opticalFlowOp",
    "image": img1_vec,
    "direction":0,
    "inputDimension": inputDimension
}

opticalFlowOp2 = {
    "type": "opticalFlowOp",
    "image": img1_vec,
    "direction":1,
    "inputDimension": inputDimension
}

# compute -u_t as the data
if(dim == 2):
    negUT = (-(img2_np - img1_np)/timeStep).flatten(order="F")

#TODO ab hier weiter anpassen
solver.add_dual(
        prox_type = "dualOpticalFlowL2DataProx",
        alpha = weight_data,
        f_list = [negUT],
        corresponding_primals = corresponding_primals,
        operator_dict = [opticalFlowOp1, opticalFlowOp2]
    )

#generate list of gradient operators with one entry for each dimension len(inputdimension)
grad_ops = []

for j in range(dim):
    grad_ops.append({
        "type": "gradientOperator",
        "gradType": "central",
        "gradDirection": j,
        "inputDimension": inputDimension
    })
for j in range(dim):
    solver.add_dual(
        prox_type="L2proxDual",
        alpha=weight_L2,
        f_list=[np.zeros(nPx)],  # no Offset
        corresponding_primals=[j],
        operator_dict=[grad_ops[j]]
    )


# start the solver
v_out, _ = solver.solve()

"""
result = v_out.reshape(inputDimension)


# vizualize
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.title("velocity field in first direction")
plt.imshow(result[0], cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("velocity field in second direction")
plt.imshow(result[1], cmap="gray")
plt.axis("off")
"""
v_vec1 = v_out[0]
v_vec2 = v_out[1]

result_1 = v_vec1.reshape(inputDimension, order="F")
result_2 = v_vec2.reshape(inputDimension, order="F")

magnitude = np.sqrt(result_1**2 + result_2**2)


vmax1 = np.max(np.abs(result_1))
vmax2 = np.max(np.abs(result_2))
vmax_mag = np.percentile(magnitude, 99)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("v_x")
plt.imshow(result_1, cmap="seismic", vmin=-vmax1, vmax=vmax1)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("v_y")
plt.imshow(result_2, cmap="seismic", vmin=-vmax2, vmax=vmax2)
plt.axis("off")

plt.subplot(1,3,3)
plt.title("magnitude")
plt.imshow(magnitude, cmap="gray", vmin=0, vmax=vmax_mag)
plt.axis("off")

plt.savefig("resultL2L2OpticalFlow.png", dpi=200, bbox_inches="tight")