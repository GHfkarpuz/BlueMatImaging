import sys
sys.path.append("/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/wrappers/python")


import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from wrapper import FlexBoxSolver # python wrapper

# test file for the TV-TV joint motion estimation and image reconstruction optical flow model: argmin_{v\in \mathbb{R}^{inputDimension*nPx}} \frac[weight_data}{2}*||u_t + grad(u)*v||_{2}^{2} + weight_TV*\sum_{i=1}^{inputDimension} ||grad(v_i)||_{1,1}

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

start_sec = 92
end_sec = 93
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

        frame_gray = cv2.resize(frame_gray, None, fx=0.8, fy=0.8)

        frame_gray = frame_gray.astype(np.float32) / 255.0

        frames.append(frame_gray)

    idx += 1

cap.release()

print(f"Extracted {len(frames)} frames")

print("start_frame:", start_frame)
print("end_frame:", end_frame)
print("target_set:", sorted(target_set))
print("extracted:", len(frames))

# parameters
weight_1 = 1.0 #data term for u
weight_2 = 0.001 #regularizer for u
weight_3 = 1.0 #motion estimation condition for u
weight_4 = 1.0 #regularizer for v

timeStep = 1.0
timeIntervall = len(frames)

tol = 1e-4

nPx = frames[0].shape[0] * frames[0].shape[1]



reconstructed_image = [np.ones(nPx)*(1.0/(i+2.0)) for i in range(len(frames))]
motionField_0 = [np.ones(nPx)*0.1 for i in range(len(frames)-1)]
motionField_1 = [np.ones(nPx)*0.1 for i in range(len(frames)-1)]



# images
for j in range(len(frames)-1):
    print("The 2-distance is:", np.linalg.norm(frames[j+1]-frames[j]))

# dimensions
inputDimension = [frames[0].shape[0],frames[0].shape[1]]

print(inputDimension)

dim = len(inputDimension)

for i in range(num_iterations):
    motionField_0_old = [x.copy() for x in motionField_0]
    motionField_1_old = [x.copy() for x in motionField_1]
    reconstructed_image_old = [x.copy() for x in reconstructed_image]


    # initialise solver
    solver = FlexBoxSolver(maxIt=2000, tol=1e-6, verbose=0)

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
    print("1")
    #weight_2*||grad_x u + grad_y u||??Aniso?
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
    print("2")
    for j in range(len(frames)):
        solver.add_dual(
            prox_type="L1IsoProxDual",
            alpha=weight_2/timeIntervall,
            f_list=[np.zeros(nPx)],
            corresponding_primals=[corresponding_primals[j]],
            operator_dict=[grad_x, grad_y]
        )
    print("1")

    #weight_3*||u_t +grad u*v||
    # operator. It has to be extended for the 3D-case.
    for j in range(0,len(frames)-2):
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
        identityOp = {
            "type": "identityOperator",
            "nPx": nPx,
            "isMinus": False
        }

        diagonalElements = np.ones(reconstructed_image[0].size)*(0.5)

        diagonalOp = {
            "type": "diagonalOperator",
            "diagonalElements": diagonalElements,
            "isMinus" : False
        }
        
        halfNegIdentityOp = {
            "type": "concatOperator",
            "operation": "composition",
            "A": negIdentityOp,
            "B": diagonalOp,
            "isMinus" : False
        }

        halfIdentityOp = {
            "type": "concatOperator",
            "operation": "composition",
            "A": identityOp,
            "B": diagonalOp,
            "isMinus" : False
        }

        solver.add_dual(
                prox_type = "L1IsoProxDual",
                alpha = weight_3/timeIntervall,
                f_list = [-reconstructed_image[j+1].flatten(order="F")],
                corresponding_primals = [corresponding_primals[j+1],corresponding_primals[j+1],corresponding_primals[j],corresponding_primals[j+2]],
                operator_dict = [weightedDivergenceOp1, weightedDivergenceOp2, halfNegIdentityOp, halfIdentityOp]
            )

        """
        print(f"1{j}")
        solver.add_dual(
                prox_type = "L1IsoProxDual",
                alpha = weight_3/timeIntervall,
                f_list = [-reconstructed_image[j+1].flatten(order="F")],
                corresponding_primals = [corresponding_primals[j],corresponding_primals[j],corresponding_primals[j],corresponding_primals[j+1]],
                operator_dict = [weightedDivergenceOp1, weightedDivergenceOp2, negIdentityOp, identityOp]
            )
        """
    """
    print("1")
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
    negIdentityOp = {
        "type": "identityOperator",
        "nPx": nPx,
        "isMinus": True
    }
    identityOp = {
        "type": "identityOperator",
        "nPx": nPx,
        "isMinus": False
    }
    print("3")
    solver.add_dual(
            prox_type = "L1IsoProxDual",
            alpha = weight_3/timeIntervall,
            f_list = [reconstructed_image[len(frames)-1].flatten(order="F")],
            corresponding_primals = [corresponding_primals[len(frames)-1], corresponding_primals[len(frames)-1], corresponding_primals[len(frames)-1],corresponding_primals[len(frames)-2]],
            operator_dict = [weightedDivergenceOp1, weightedDivergenceOp2, identityOp, negIdentityOp]
        )
    print("1")"""
    reconstructed_image, _ = solver.solve()

    del solver

    # initialise solver
    solver = FlexBoxSolver(maxIt=2000, tol=1e-6, verbose=0)

    for j in range(len(frames)-2):
        solver.add_primal(motionField_0[j])#motion field for first direction for time t_j 
        solver.add_primal(motionField_1[j])#motion field for second direction for time t_j

    corresponding_primals = [j for j in range(2*(len(frames)-2))]#ordered (v^1_x,v^1_y, v^2_x, v^2_y,...)

    negUT = [np.zeros(nPx) for j in range(len(reconstructed_image)-2)]
    # compute -u_t as the data
    for j in range(len(reconstructed_image)-2):
        diff = (reconstructed_image[j+2] - reconstructed_image[j])*0.5
        scale = max(np.abs(np.max(diff)), 1e-6)

        negUT[j] = -(diff / timeStep).flatten(order="F")

        #negUT[j] = -(((reconstructed_image[j+1] - reconstructed_image[j])/timeStep)/max(np.abs(np.max(reconstructed_image[j+1] - reconstructed_image[j])), 1e-6)).flatten(order="F")

    for j in range(len(frames)-2):

        # operator. It has to be extended for the 3D-case.
        massPreservationOp1 = {
            "type": "massPreservationOp",
            "image": reconstructed_image[j+1].flatten(order="F"),
            "direction":0,
            "inputDimension": inputDimension
        }

        massPreservationOp2 = {
            "type": "massPreservationOp",
            "image": reconstructed_image[j+1].flatten(order="F"),
            "direction":1,
            "inputDimension": inputDimension
        }

        solver.add_dual(
                prox_type = "L1proxDualData",
                alpha = 1.0/timeIntervall,
                f_list = [negUT[j]],
                corresponding_primals = [corresponding_primals[2*j], corresponding_primals[2*j+1]],
                operator_dict = [massPreservationOp1, massPreservationOp2]
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

    for j in range(len(frames)-2):
        # TV on v_i
        solver.add_dual(
            prox_type="L2proxDual",
            alpha=weight_4/timeIntervall,
            f_list=[np.zeros(nPx)],
            corresponding_primals=[corresponding_primals[2*j],corresponding_primals[2*j + 1]],
            operator_dict=[grad_x, grad_y]
        )

    # start the solver
    v_out, _ = solver.solve()

    for j in range(len(frames) - 2):
        motionField_0[j] = v_out[2*j]
        motionField_1[j] = v_out[2*j+1]

    del solver
    
    diffV0 = max(np.linalg.norm(a - b) for a, b in zip(motionField_0, motionField_0_old))
    diffV1 = max(np.linalg.norm(a - b) for a, b in zip(motionField_1, motionField_1_old))
    diffU  = max(np.linalg.norm(a - b) for a, b in zip(reconstructed_image, reconstructed_image_old))

    print("diffV0 is", diffV0)
    print("diffV1 is", diffV1)
    print("diffU is", diffU)

    if((i+1)%1==0):
        # compute total energy directly in outer loop

        E_data = 0.0
        E_tv_u = 0.0
        E_flow = 0.0
        E_tv_v = 0.0


        # data term + TV(u)

        for j in range(len(frames)):

            u = reconstructed_image[j].reshape(inputDimension, order="F")
            f = frames[j]

            # data fidelity
            E_data += 0.5 * weight_1 * np.sum((u - f)**2)

            # forward gradients
            gx = np.zeros_like(u)
            gy = np.zeros_like(u)

            gx[:-1, :] = u[1:, :] - u[:-1, :]
            gy[:, :-1] = u[:, 1:] - u[:, :-1]

            # anisotropic TV
            E_tv_u += weight_2 * (
                np.sum(np.abs(gx)) +
                np.sum(np.abs(gy))
            )


        # optical flow term + TV(v)

        for j in range(len(frames)-1):
            u1 = reconstructed_image[j].reshape(inputDimension, order="F")
            u2 = reconstructed_image[j+1].reshape(inputDimension, order="F")

            vx = motionField_0[j].reshape(inputDimension, order="F")
            vy = motionField_1[j].reshape(inputDimension, order="F")

            # temporal derivative
            ut = (u2 - u1) / timeStep

            # central gradients
            ux = np.zeros_like(u1)
            uy = np.zeros_like(u1)

            ux[1:-1, :] = 0.5 * (u1[2:, :] - u1[:-2, :])
            uy[:, 1:-1] = 0.5 * (u1[:, 2:] - u1[:, :-2])

            # optical flow residual
            residual = ut + ux * vx + uy * vy

            E_flow += weight_3 * np.sum(np.abs(residual))

            # TV(vx)
            gx_vx = np.zeros_like(vx)
            gy_vx = np.zeros_like(vx)

            gx_vx[:-1, :] = vx[1:, :] - vx[:-1, :]
            gy_vx[:, :-1] = vx[:, 1:] - vx[:, :-1]

            E_tv_v += weight_4 * (
                np.sum(np.abs(gx_vx)) +
                np.sum(np.abs(gy_vx))
            )

            # TV(vy)
            gx_vy = np.zeros_like(vy)
            gy_vy = np.zeros_like(vy)

            gx_vy[:-1, :] = vy[1:, :] - vy[:-1, :]
            gy_vy[:, :-1] = vy[:, 1:] - vy[:, :-1]

            E_tv_v += weight_4 * (
                np.sum(np.abs(gx_vy) + np.abs(gy_vy))
            )

        
        # total energy
        E_total = E_data + E_tv_u + E_flow + E_tv_v

        print("====================================")
        print(f"Outer iteration {i}")
        print(f"E_total : {E_total}")
        print(f"E_data  : {E_data}")
        print(f"E_tv_u  : {E_tv_u}")
        print(f"E_flow  : {E_flow}")
        print(f"E_tv_v  : {E_tv_v}")
        print("====================================")


    if(diffV0<tol and diffV1<tol and diffU<tol):
        print(f"converged for outer iterations in the {i}-th iteration.")
        break

    for j in range(len(frames)-2):
        if((i+1)%1==0):

            result_1_v = motionField_0[j].reshape(inputDimension, order="F")
            result_2_v = motionField_1[j].reshape(inputDimension, order="F")

            magnitude_v = np.sqrt(result_1_v**2 + result_2_v**2)


            vmax1 = np.max(np.abs(result_1_v))
            vmax2 = np.max(np.abs(result_2_v))
            vmax_mag = np.percentile(magnitude_v, 99)

            print(f"vmax1 for i = {i} is ", vmax1)
            print(f"vmax2 for i = {i} is ", vmax2)


            result_1_u = reconstructed_image[j+1].reshape(inputDimension, order="F")

            magnitude_u = np.sqrt(result_1_u**2)


            umax1 = np.max(np.abs(result_1_u))
            umax_mag = np.percentile(magnitude_u, 99)

            print(f"umax1 for i = {i} is ", umax1)


            plt.figure(figsize=(7,7))

            plt.subplot(2,2,1)
            plt.title("magnitude of u")
            plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax_mag)
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
            

            plt.savefig(f"resultTVL2MassPreservationBlueMatExample{i+1},{j+1}.png", dpi=200, bbox_inches="tight")

            plt.close()