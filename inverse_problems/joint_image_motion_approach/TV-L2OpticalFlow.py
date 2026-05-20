import sys
sys.path.append("/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/wrappers/python")

import numpy as np
import wrapper


image_folder = "../../data/"

img1_string = "frame10.bmp"
img2_string = "frame11.bmp"

# parameters
weight_1 = 1.0
weight_2 = 1.0
weight_3 = 1.0
timeStep = 1.0



# images
img1 = Image.open(img1_string).convert("L")
img2 = Image.open(img2_string).convert("L")

img1_np = np.array(img1).astype(np.float32)
img2_np = np.array(img2).astype(np.float32)

#this needs to be made right now, because otherwise the entries get to big. it needs to be fixed
maxImage = np.max(np.abs(img1_np)).astype(np.float32)
img1_np = img1_np/maxImage
img2_np = img2_np/maxImage


print("The 2-distance is:", np.linalg.norm(img2_np-img1_np))
# dimensions
inputDimension = list(img1_np.shape)
nPx = img1_np.size


# flatten image for operator
img1_vec = img1_np.flatten(order="F")

timeSteps = [1.0]
numTimeSteps = len(timeSteps)


# initialise solver
solverU = FlexBoxSolver(maxIt=3000, tol=1e-6, verbose=2)
solverV = FlexBoxSolver(maxIt=3000, tol=1e-6, verbose=2)

# primal variables
dim = len(inputDimension)

print(inputDimension)

#add a primal u for the image reconstruction
solverU.add_primal(np.zeros(nPx))

# add a primal variable v_i for each direction of the motion field
for i in range(dim):
    solverV.add_primal(np.zeros(nPx))


corresponding_primals_u = [0]
corresponding_primals_v = [i for i in range(dim)]

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
        prox_type = "L1proxDualData",
        alpha = weight_data,
        f_list = [negUT],
        corresponding_primals = corresponding_primals,
        operator_dict = [massPreservationOp1, massPreservationOp2]
    )

#generate list of gradient operators with one entry for each dimension len(inputdimension)
grad_ops = []

grad_x(
    "type": "gradientOperator",
    "gradType": "forward",
    "gradDirection": 0,
    "inputDimension": inputDimension
)

grad_y(
    "type": "gradientOperator",
    "gradType": "forward",
    "gradDirection": 1,
    "inputDimension": inputDimension
)

for j in range(dim):
    solver.add_dual(
        prox_type="L1AnisoProxDual",
        alpha=weight_L2,
        f_list=[np.zeros(nPx)],  # no Offset
        corresponding_primals= corresponding_primals,
        operator_dict=grad_ops
    )


# start the solver
v_out, _ = solver.solve()

v_vec1 = v_out[0]
v_vec2 = v_out[1]

result_1 = v_vec1.reshape(inputDimension, order="F")
result_2 = v_vec2.reshape(inputDimension, order="F")

print(np.max(np.abs(result_1)))
print(np.max(np.abs(result_2)))
print(np.any(np.isnan(result_1)))
print(np.any(np.isinf(result_1)))

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

plt.savefig("resultL1TVMassPreservation.png", dpi=200, bbox_inches="tight")