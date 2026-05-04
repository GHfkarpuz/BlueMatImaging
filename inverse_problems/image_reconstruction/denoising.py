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
img = Image.open("frame10.bmp").convert("RGB")
img_np = np.array(img).astype(np.float32)
print(img_np)

# Luminosity Method
gray = 0.299 * img_np[:,:,0] + 0.587 * img_np[:,:,1] + 0.114 * img_np[:,:,2]
gray = gray / 255.0

# add noise
noise = gray + 0.1 * np.random.randn(*gray.shape)
noise = np.clip(noise, 0, 1)

# initialise solver
solver = FlexBoxSolver(maxIt=10000, tol=1e-6, verbose=1)


H, W = noise.shape
nPx = H * W

# add primal variable
solver.add_primal(np.zeros(nPx))

# Data term (L2): ||x - f||²
f_vec = noise.flatten()

identity_op = {
    "type": "identityOperator",
    "nPx": nPx
}

solver.add_dual(
    prox_type="L2proxDualData",
    alpha=weight_data,
    f_list=[f_vec],
    corresponding_primals=[0],
    operator_dict=[identity_op]
)


# TV Regularizer: ||∇x||₁ (isotrop)
grad_x = {
    "type": "gradientOperator",
    "gradType": "forward",
    "gradDirection": 1,   # x-direction
    "inputDimension": [H, W]
}

grad_y = {
    "type": "gradientOperator",
    "gradType": "forward",
    "gradDirection": 2,   # y-direction
    "inputDimension": [H, W]
}

solver.add_dual(
    prox_type="L1IsoProxDual",
    alpha=weight_tv,
    f_list=[np.zeros(nPx)],  # no Offset
    corresponding_primals=[0],
    operator_dict=[grad_x, grad_y]
)

# start the solver
x_out, _ = solver.solve()

result = x_out[0].reshape(H, W)


# vizualize
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Noisy")
plt.imshow(noise, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Denoised (TV)")
plt.imshow(result, cmap="gray")
plt.axis("off")
plt.savefig("result.png", dpi=200, bbox_inches="tight")
