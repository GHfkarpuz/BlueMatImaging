import sys
sys.path.append("/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/wrappers/python")


import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from wrapper import FlexBoxSolver # python wrapper


# test file for the TV-L2 joint motion estimation and image reconstruction optical flow model: argmin_{v\in \mathbb{R}^{inputDimension*nPx}} \frac[weight_data}{2}*||u_t + grad(u)*v||_{2}^{2} + weight_TV*\sum_{i=1}^{inputDimension} ||grad(v_i)||_{1,1}

num_iterations = 100

video_path = "media4.avi"

cut = True
cut_width = [0.0 , 0.0]
cut_height = [0.6 , 0.0]

cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

start_sec = 93
end_sec = 95
n_frames = 5

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

cap = cv2.VideoCapture(video_path)

idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if idx in target_set:
        if frame is None:
            idx += 1
            continue

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if cut:
            x1 = int(width * cut_width[0])
            x2 = int(width * (1 - cut_width[1]))
            y1 = int(height * cut_height[0])
            y2 = int(height * (1 - cut_height[1]))

            frame_gray = frame_gray[y1:y2, x1:x2]

        if frame_gray.size == 0:
            idx += 1
            continue

        #frame_gray = cv2.fastNlMeansDenoising(frame_gray, None, 20, 7, 15)

        frame_gray = cv2.resize(frame_gray, None, fx=0.45, fy=0.45)

        frames.append(frame_gray)

    idx += 1

cap.release()

print(f"Extracted {len(frames)} frames")

print("start_frame:", start_frame)
print("end_frame:", end_frame)
print("target_set:", sorted(target_set))
print("extracted:", len(frames))

# parameters
weight_1 = 0.5
weight_2 = 1.0
weight_3 = 1.0
weight_4 = 2.4
timeStep = 1.0
timeIntervall = len(frames)

nPx = frames[0].shape[0] * frames[0].shape[1]

reconstructed_image = [np.zeros(nPx) for i in range(len(frames))]
motionField_0 = [np.zeros(nPx) for i in range(len(frames)-1)]
motionField_1 = [np.zeros(nPx) for i in range(len(frames)-1)]

for i in range(num_iterations):
    # images
    for j in range(len(frames)-1):
        print("The 2-distance is:", np.linalg.norm(frames[j+1]-frames[j]))
    # dimensions
    inputDimension = [frames[0].shape[0],frames[0].shape[1]]

    # initialise solver
    solver = FlexBoxSolver(maxIt=50, tol=1e-6, verbose=1)

    # primal variables
    dim = len(inputDimension)



    # add a primal variable for each frame
    for j in range(len(frames)):
        solver.add_primal(reconstructed_image[j])

    corresponding_primals = [j for j in range(len(frames))]

    # (0.5)*||Ku - f||
    data_op = {
    "type": "identityOperator",
    "nPx": nPx
    }


    for j in range(len(frames)):
        solver.add_dual(
            prox_type="L2proxDualData",
            alpha=weight_1/timeIntervall,
            f_list=[frames[j].flatten(order="F")],
            corresponding_primals=[corresponding_primals[j]],
            operator_dict=[data_op]
        )
    
    #weight_2*||grad_x u + grad_y u||
    grad_x={"type": "gradientOperator",
        "gradType": "forward",
        "gradDirection": 0,
        "inputDimension": inputDimension
    }
    grad_y={"type": "gradientOperator",
            "gradType": "forward",
            "gradDirection": 1,
            "inputDimension": inputDimension
    }

    for j in range(len(frames)):
        solver.add_dual(
            prox_type="L1IsoProxDual",
            alpha=weight_2/timeIntervall,
            f_list=[np.zeros(nPx)],
            corresponding_primals=[corresponding_primals[j]],
            operator_dict=[grad_x, grad_y]
        )


    #weight_3*||u_t +grad u*v||
    weightedDivergenceOp_list = []

    # operator. It has to be extended for the 3D-case.
    for j in range(len(frames)-1):
        weightedDivergenceOp1 = {
            "type": "weightedDivergenceOperator",
            "gradType": "central",
            "weights": motionField_0[j].flatten(order="F"),
            "gradDirection": 0,
            "inputDimension": inputDimension
        }
        weightedDivergenceOp2 = {
            "type": "weightedDivergenceOperator",
            "gradType": "central",
            "weights": motionField_1[j].flatten(order="F"),
            "gradDirection": 1,
            "inputDimension": inputDimension
        }

        negIdentityOp = {
            "type": "identityOperator",
            "nPx": nPx,
            "isMinus": True
        }

        solver.add_dual(
                prox_type = "L1proxDualData",
                alpha = weight_3/timeIntervall,
                f_list = [-frames[j+1].flatten(order="F")],
                corresponding_primals = [corresponding_primals[j], corresponding_primals[j], corresponding_primals[j]],
                operator_dict = [weightedDivergenceOp1, weightedDivergenceOp2, negIdentityOp]
            )
    
    weightedDivergenceOp1 = {
        "type": "weightedDivergenceOperator",
        "gradType": "central",
        "weights": motionField_0[len(frames)-2].flatten(order="F"),
        "gradDirection": 0,
        "inputDimension": inputDimension
    }
    weightedDivergenceOp2 = {
        "type": "weightedDivergenceOperator",
        "gradType": "central",
        "weights": motionField_1[len(frames)-2].flatten(order="F"),
        "gradDirection": 1,
        "inputDimension": inputDimension
    }
    identityOp = {
        "type": "identityOperator",
        "nPx": nPx,
        "isMinus": False
    }

    solver.add_dual(
            prox_type = "L1proxDualData",
            alpha = weight_3/timeIntervall,
            f_list = [frames[len(frames)-1].flatten(order="F")],
            corresponding_primals = [corresponding_primals[len(frames)-1], corresponding_primals[len(frames)-1], corresponding_primals[len(frames)-1]],
            operator_dict = [weightedDivergenceOp1, weightedDivergenceOp2, identityOp]
        )
    
    reconstructed_image, _ = solver.solve()

    del solver

    # initialise solver
    solver = FlexBoxSolver(maxIt=50, tol=1e-6, verbose=1)

    for j in range(len(frames)-1):
        solver.add_primal(motionField_0[j])#motion field for first direction for time t_j 
        solver.add_primal(motionField_1[j])#motion field for second direction for time t_j

    corresponding_primals = [j for j in range(2*(len(frames)-1))]#ordered (v^1_x,v^1_y, v^2_x, v^2_y,...)

    negUT = [np.zeros(nPx) for j in range(len(frames)-1)]
    # compute -u_t as the data
    for j in range(len(reconstructed_image)-1):
        diff = reconstructed_image[j+1] - reconstructed_image[j]
        scale = max(np.abs(np.max(diff)), 1e-6)

        negUT[j] = -(diff / timeStep / scale).flatten(order="F")

        negUT[j] = -(((reconstructed_image[j+1] - reconstructed_image[j])/timeStep)/max(np.abs(np.max(reconstructed_image[j+1] - reconstructed_image[j])), 1e-6)).flatten(order="F")

    for j in range(len(frames)-1):

        # operator. It has to be extended for the 3D-case.
        opticalFlowOp1 = {
            "type": "opticalFlowOp",
            "image": reconstructed_image[j].flatten(order="F"),
            "direction":0,
            "inputDimension": inputDimension
        }

        opticalFlowOp2 = {
            "type": "opticalFlowOp",
            "image": reconstructed_image[j].flatten(order="F"),
            "direction":1,
            "inputDimension": inputDimension
        }

        solver.add_dual(
                prox_type = "L1proxDualData",
                alpha = 1.0/timeIntervall,
                f_list = [negUT[j]],
                corresponding_primals = [corresponding_primals[2*j], corresponding_primals[2*j+1]],
                operator_dict = [opticalFlowOp1, opticalFlowOp2]
            )

    #generate gradient operators 

    grad_x={"type": "gradientOperator",
            "gradType": "forward",
            "gradDirection": 0,
            "inputDimension": inputDimension
        }
    grad_y={"type": "gradientOperator",
            "gradType": "forward",
            "gradDirection": 1,
            "inputDimension": inputDimension
    }

    for j in range(len(frames)-1):
        # TV on v_i
        solver.add_dual(
            prox_type="L1IsoProxDual",
            alpha=weight_4/timeIntervall,
            f_list=[np.zeros(nPx)],
            corresponding_primals=[corresponding_primals[2*j],corresponding_primals[2*j + 1]],
            operator_dict=[grad_x, grad_y]
        )

    # start the solver
    v_out, _ = solver.solve()

    for j in range(len(frames) - 1):
        motionField_0[j] = v_out[2*j]
        motionField_1[j] = v_out[2*j+1]

    del solver
    
    for j in range(len(frames)-1):
        if((i+1)%5==0):

            result_1_v = motionField_0[j].reshape(inputDimension, order="F")
            result_2_v = motionField_1[j].reshape(inputDimension, order="F")

            magnitude_v = np.sqrt(result_1_v**2 + result_2_v**2)


            vmax1 = np.max(np.abs(result_1_v))
            vmax2 = np.max(np.abs(result_2_v))
            vmax_mag = np.percentile(magnitude_v, 99)

            print(f"vmax1 for i = {i} is ", vmax1)
            print(f"vmax2 for i = {i} is ", vmax2)


            result_1_u = reconstructed_image[j].reshape(inputDimension, order="F")

            magnitude_u = np.sqrt(result_1_u**2)


            umax1 = np.max(np.abs(result_1_u))
            umax_mag = np.percentile(magnitude_u, 99)

            print(f"umax1 for i = {i} is ", umax1)


            plt.figure(figsize=(12,4))

            plt.subplot(1,4,1)
            plt.title("magnitude of u")
            plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax_mag)
            plt.axis("off")

            plt.subplot(1,4,2)
            plt.title("v_x")
            plt.imshow(result_2_v, cmap="seismic", vmin=-vmax2, vmax=vmax2)
            plt.axis("off")

            plt.subplot(1,4,3)
            plt.title("v_y")
            plt.imshow(result_1_v, cmap="seismic", vmin=-vmax1, vmax=vmax1)
            plt.axis("off")

            plt.subplot(1,4,4)
            plt.title("magnitude of v")
            plt.imshow(magnitude_v, cmap="gray", vmin=0, vmax=vmax_mag)
            plt.axis("off")
            

            plt.savefig(f"resultL2TVMassPreservationBlueMatExample{i+1},{j+1}.png", dpi=200, bbox_inches="tight")

            plt.close()