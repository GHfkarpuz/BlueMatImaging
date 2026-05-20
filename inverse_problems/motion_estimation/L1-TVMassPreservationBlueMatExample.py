import sys
sys.path.append("/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/wrappers/python")


import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from wrapper import FlexBoxSolver # python wrapper


# test file for the discrete L2-TV (anisotropic) optical flow model: argmin_{v\in \mathbb{R}^{inputDimension*nPx}} \frac[weight_data}{2}*||u_t + grad(u)*v||_{2}^{2} + weight_TV*\sum_{i=1}^{inputDimension} ||grad(v_i)||_{1,1}


video_path = "media7.avi"

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

start_sec = 24
end_sec = 27
n_frames = 10

start_frame = int(start_sec * fps)
end_frame = int(end_sec * fps)

# Clip bounds
start_frame = max(0, start_frame)
end_frame = min(total_frames - 1, end_frame)

# distribute indices 
frame_indices = np.linspace(start_frame, end_frame, n_frames).astype(int)

frames = []
current_idx = start_frame
target_set = set(frame_indices)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if current_idx in target_set:
        # OpenCV reads BGR, convert to RGB optionally
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scale = 0.2
        frame_rgb = cv2.resize(
            frame_rgb,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )
        frames.append(frame_rgb)

    current_idx += 1

    if current_idx > end_frame:
        break

cap.release()

print(f"Extracted {len(frames)} frames")


# parameters
weight_data = 1.0
weight_TV = 0.1
timeStep = 1.0

for i in range(len(frames)-1):

    # images
    img1_np = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
    img2_np = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY).astype(np.float32)

    """
    img1_np = img1_np
    img2_np = img2_np
    """

    
    print("The 2-distance is:", np.linalg.norm(img2_np-img1_np))
    # dimensions
    inputDimension = list(img1_np.shape)
    nPx = img1_np.size


    # flatten image for operator
    img1_vec = img1_np.flatten(order="F")

    # initialise solver
    solver = FlexBoxSolver(maxIt=1000, tol=1e-6, verbose=2)

    # primal variables
    dim = len(inputDimension)

    print(inputDimension)


    # add a primal variable for each direction of the motion field
    for j in range(dim):
        solver.add_primal(np.zeros(nPx))

    corresponding_primals = [j for j in range(dim)]

    # operator. It has to be extended for the 3D-case.
    massPreservationOp1 = {
        "type": "massPreservationOp",
        "image": img1_vec,
        "direction":0,
        "inputDimension": inputDimension
    }

    massPreservationOp2 = {
        "type": "massPreservationOp",
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

    #generate gradient operators 

    grad_x={"type": "gradientOperator",
            "gradType": "central",
            "gradDirection": 0,
            "inputDimension": inputDimension
        }
    grad_y={"type": "gradientOperator",
            "gradType": "central",
            "gradDirection": 1,
            "inputDimension": inputDimension
    }

    solver.add_dual(
        prox_type="L1AnisoProxDual",
        alpha=weight_TV,
        f_list=[np.zeros(nPx)],  # no Offset
        corresponding_primals= [0],
        operator_dict=[grad_x, grad_y]
    )

    solver.add_dual(
        prox_type="L1AnisoProxDual",
        alpha=weight_TV,
        f_list=[np.zeros(nPx)],  # no Offset
        corresponding_primals= [1],
        operator_dict=[grad_x, grad_y]
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


    plt.savefig(f"resultL1TVMassPreservationBlueMatExample{i}.png", dpi=200, bbox_inches="tight")

    plt.close()

    del solver

    del img1_np
    del img2_np
    del negUT

    del v_out
    del v_vec1
    del v_vec2

    del result_1
    del result_2
    del magnitude