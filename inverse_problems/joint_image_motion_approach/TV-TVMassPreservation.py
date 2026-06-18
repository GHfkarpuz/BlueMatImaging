import sys
sys.path.append("/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/wrappers/python")


import cv2
import numpy as np
import gc
from PIL import Image
import matplotlib.pyplot as plt

from wrapper import FlexBoxSolver # python wrapper

# test file for the TV-TV joint motion estimation and image reconstruction optical flow model: argmin_{v\in \mathbb{R}^{inputDimension*nPx}} \frac[weight_data}{2}*||u_t + grad(u)*v||_{2}^{2} + weight_TV*\sum_{i=1}^{inputDimension} ||grad(v_i)||_{1,1}

num_iterations = 100

video_path = "media4.avi"
denoisedImage = np.load("denoisedImage0,3.npy")

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
n_frames = 4

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

        frame_gray = cv2.resize(frame_gray, None, fx=0.1, fy=0.1)

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
weight_2 = 0.3 #regularizer for u
weight_3 = 1.0 #motion estimation condition for u
weight_4 = 1.0 #regularizer for v
weight_5 = 0.0 #regularizer for v to stop uncontrolled increase of values

timeStep = 1.0
timeIntervall = len(frames)

tol = 1e-6

nPx = frames[0].shape[0] * frames[0].shape[1]


reconstructed_image = np.ones(nPx*len(frames), dtype=np.float32) * 0.5
#reconstructed_image = np.random.normal(0.0, 0.9, nPx*len(frames)).astype(np.float32)
reconstructed_image_frames = [np.ones(nPx, dtype=np.float32) * (1/(i+2)) for i in range(len(frames))]
#motionField_0 = [np.ones(nPx)*0.5 for j in range(len(frames))]
#motionField_1 = [np.ones(nPx)*0.5 for j in range(len(frames))]
motionField_combined = [np.ones(nPx*2)*0.5 for j in range(len(frames))] # ordered like v¹_x,v¹_<,v²_x,v²_y,....

motionField_0 = [np.ones(nPx*2)*0.5 for j in range(len(frames))]
motionField_1 = [np.ones(nPx*2)*0.5 for j in range(len(frames))]

flat_frames = [np.ascontiguousarray(f.flatten(order="F"), dtype=np.float32) for f in frames]

# images
for j in range(len(frames)-1):
    print("The 2-distance is:", np.linalg.norm(frames[j+1]-frames[j]))

# dimensions
inputDimension = [frames[0].shape[0],frames[0].shape[1]]

print(inputDimension)


dim = len(inputDimension)
dimensionsWithTime = [inputDimension[0], inputDimension[1], len(frames)]
dimensionsCombinedMotionFields = [inputDimension[0], inputDimension[1], 2]

print(1)
framesBig = [np.zeros(inputDimension[0]*inputDimension[1]*len(frames)) for i in range(len(frames))]
print(2)

for i in range(len(framesBig)):
    print(i)
    for j in range(nPx):
        framesBig[i][i*nPx + j]=frames[i].flatten(order="F")[j]


for i in range(num_iterations):
    motionField_0_old = [np.array(x, copy=True) for x in motionField_0]
    motionField_1_old = [np.array(x, copy=True) for x in motionField_1]
    reconstructed_image_old = np.array(reconstructed_image, copy=True)
    # initialise solver
    solver = FlexBoxSolver(maxIt=1000, tol=1e-6, verbose=1)
    try:
        print(reconstructed_image[0].shape)
    except:
        print("no shape")
    # add a primal variable which holds all frames
    # Erzwinge Fortran-Order, passend zu deinem .flatten(order="F")
    u_input = np.ascontiguousarray(reconstructed_image, dtype=np.float32) 

    u_input = np.array(reconstructed_image, dtype=np.float32, order="F")
    solver.add_primal(u_input)
    #solver.add_primal(reconstructed_image)
    #solver.add_primal(reconstructed_image)

    corresponding_primals = [0]
    print("rfwe")
    print(len(reconstructed_image))
    print(reconstructed_image.shape)
    print(reconstructed_image.size)

    for j in range(len(frames)):
        # (0.5)*||Ku - f||
        data_op = {
            "type": "identityOperator",
            "nPx": nPx
        }
        projectionOp = {
            "type": "projectionOp",
            "operator":data_op,
            "index": j,
            "inputDimension": dimensionsWithTime
        }
        #frames[j] = frames[j].astype(np.float32)
        solver.add_dual(
            prox_type="L2proxDualData",
            alpha=weight_1/timeIntervall,
            f_list=[flat_frames[j]],
            corresponding_primals=[corresponding_primals[0]],
            operator_dict=[projectionOp]
        )
    print("1")
    """
    # (0.5)*||Ku - f||
    data_op = {
        "type": "identityOperator",
        "nPx": nPx*len(frames)
    }
    #frames[j] = frames[j].astype(np.float32)
    solver.add_dual(
        prox_type="L2proxDualData",
        alpha=weight_1/timeIntervall,
        f_list=[flat_frames[j]],
        corresponding_primals=[corresponding_primals[0]],
        operator_dict=[data_op]
    )
    """
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
        projectionOpX = {
            "type": "projectionOp",
            "operator":grad_x,
            "index": j,
            "inputDimension": dimensionsWithTime
            }
        projectionOpY = {
            "type": "projectionOp",
            "operator":grad_y,
            "index": j,
            "inputDimension": dimensionsWithTime
            }
        solver.add_dual(
            prox_type="L1IsoProxDual",
            alpha=weight_2/timeIntervall,
            f_list=[np.zeros(nPx)],
            corresponding_primals=[corresponding_primals[0]],
            operator_dict=[projectionOpX, projectionOpY]
        )
    print("1")
    
    #weight_3*||u_t +grad u*v||
    # operator. It has to be extended for the 3D-case.
    for j in range(len(frames)):
        weightedDivergenceOp1 = {
            "type": "weightedDivergenceOperator",
            "gradType": "forward",
            "weights": motionField_0[j].flatten(order="F"),
            "gradDirection": 0,
            "inputDimension": inputDimension
        }
        weightedDivergenceOp2 = {
            "type": "weightedDivergenceOperator",
            "gradType": "forward",
            "weights": motionField_1[j].flatten(order="F"),
            "gradDirection": 1,
            "inputDimension": inputDimension
        }
        projectionOp1 = {
            "type": "projectionOp",
            "operator":weightedDivergenceOp1,
            "index": j,
            "inputDimension": dimensionsWithTime
        }
        projectionOp2 = {
            "type": "projectionOp",
            "operator":weightedDivergenceOp2,
            "index": j,
            "inputDimension": dimensionsWithTime
        }
        timeDerivativeOp = {
            "type": "gradientOperator",
            "gradType": "forward",
            "gradDirection": 2,
            "inputDimension": dimensionsWithTime
        }
        projectionOutputOp = {
            "type": "projectionOutputOp",
            "index": j,
            "inputDimension": dimensionsWithTime,
            "isMinus" : False
        }
        concatOpForTimeDerivative = {
            "type": "concatOperator",
            "operation": "composition",
            "A": projectionOutputOp,
            "B": timeDerivativeOp,
            "isMinus" : False
        }
        solver.add_dual(
            prox_type = "L1AnisoProxDual",
            alpha = weight_3/timeIntervall,
            f_list = [np.zeros(nPx * len(frames), dtype=np.float32)],
            corresponding_primals = [corresponding_primals[0]],
            operator_dict = [projectionOp1, projectionOp2, concatOpForTimeDerivative]
        )
    """
    print("before first solver")
    reconstructed_image, _ = solver.solve()

    print("after first solver")

    # Jetzt ist reconstructed_image garantiert ein NumPy-Array und .flatten() oder .reshape() funktionieren!
    #reconstructed_image = reconstructed_image.flatten(order="F") 

    # Danach kommt dein Code für das Reshaping der Frames:
    reconstructed_image_frames = [frame.copy() for frame in reconstructed_image[0].reshape(len(frames), nPx)]
    """
    print("before first solver")
    u_out, _ = solver.solve()
    print("after first solver")

    # 1. RETTE DAS ARRAY: Hole das echte Array aus der Liste und überschreibe es sauber
    reconstructed_image = np.array(u_out[0], dtype=np.float32, order="F").flatten(
        order="F"
    )

    # 2. FRAMES EXTRAHIEREN: Nutze das saubere Array für die Einzelbilder
    reconstructed_image_frames = [
        frame.copy() for frame in reconstructed_image.reshape(len(frames), nPx)
    ]

    del solver

    #reconstructed_image.flatten(order="F")
    #for j in range(len(frames)):
        #for k in range(nPx):
            #reconstructed_image_frames[j][k] = reconstructed_image[j*nPx + k]
    
    #reconstructed_image_frames = [frame.copy() for frame in reconstructed_image.reshape(len(frames), nPx)]

    for j in range(len(frames)):
        for k in range(nPx):
            motionField_combined[j][k]=motionField_0[j][k]
            motionField_combined[j][nPx+k]=motionField_1[j][k]
    
    # initialise solver
    solver = FlexBoxSolver(maxIt=1000, tol=1e-6, verbose=1)
    print("after first solver")
    for j in range(len(frames)):
        v_input = np.ascontiguousarray(motionField_combined[j], dtype=np.float32)
        solver.add_primal(v_input)
        #solver.add_primal(motionField_combined[j])#motion field for both directions for time t_j 

    corresponding_primals = [j for j in range(len(frames))]

    negUT = [np.zeros(nPx) for j in range(len(reconstructed_image_frames))]
    # compute -u_t as the data
    for j in range(len(reconstructed_image_frames)-1):
        diff = (reconstructed_image_frames[j+1] - reconstructed_image_frames[j])
        scale = max(np.abs(np.max(diff)), 1e-6)

        negUT[j] = -(diff / timeStep).flatten(order="F")

        #negUT[j] = -(((reconstructed_image_frames[j+1] - reconstructed_image_frames[j])/timeStep)/max(np.abs(np.max(reconstructed_image_frames[j+1] - reconstructed_image_frames[j])), 1e-6)).flatten(order="F")
    negUT[len(reconstructed_image_frames)-1]=negUT[len(reconstructed_image_frames)-2]

    for j in range(len(frames)):
        # operator. It has to be extended for the 3D-case.
        massPreservationOp1 = {
            "type": "massPreservationOp",
            "image": reconstructed_image_frames[j].flatten(order="F"),
            "direction":0,
            "inputDimension": inputDimension
        }

        massPreservationOp2 = {
            "type": "massPreservationOp",
            "image": reconstructed_image_frames[j].flatten(order="F"),
            "direction":1,
            "inputDimension": inputDimension
        }

        projectionOp1 = {
            "type": "projectionOp",
            "operator":massPreservationOp1,
            "index": 0,
            "inputDimension": dimensionsCombinedMotionFields
        }

        projectionOp2 = {
            "type": "projectionOp",
            "operator":massPreservationOp2,
            "index": 1,
            "inputDimension": dimensionsCombinedMotionFields
        }

        concatOp = {
            "type": "concatOperator",
            "operation": "addition",
            "A": projectionOp1,
            "B": projectionOp2,
            "isMinus" : False
        }

        solver.add_dual(
                prox_type = "L1proxDualData",
                alpha = 1.0/timeIntervall,
                f_list = [negUT[j]],
                corresponding_primals = [corresponding_primals[j]],
                operator_dict = [concatOp]
            )

    #generate gradient operators 

    grad_x={"type": "gradientOperator",
            "gradType": "forward",
            "gradDirection": 0,
            "inputDimension": dimensionsCombinedMotionFields
        }
    grad_y={"type": "gradientOperator",
            "gradType": "forward",
            "gradDirection": 1,
            "inputDimension": dimensionsCombinedMotionFields
    }

    for j in range(len(frames)):
        # TV on v_i
        solver.add_dual(
            prox_type="L1AnisoProxDual",
            alpha=weight_4/timeIntervall,
            f_list=[np.zeros(nPx)],#last one is zero array on purpose, because of forward differences for the time derivative
            corresponding_primals=[corresponding_primals[j]],
            operator_dict=[grad_x, grad_y]
        )
        """
        solver.add_dual(
            prox_type="L1AnisoProxDual",
            alpha=weight_4/timeIntervall,
            f_list=[np.zeros(nPx)],#last one is zero array on purpose, because of forward differences for the time derivative
            corresponding_primals=[corresponding_primals[j + 1]],
            operator_dict=[grad_x, grad_y]
        )"""
    
    print("before second solver")
    # start the solver
    v_out, _ = solver.solve()
    print("after second solver")
    for j in range(len(frames)):
        motionField_combined[j] = np.copy(v_out[j])
        for k in range(nPx):
            motionField_0[j][k] = v_out[j][k]
            motionField_1[j][k] = v_out[j][nPx + k]


    del solver


    diffV0 = max(np.linalg.norm(a - b) for a, b in zip(motionField_0, motionField_0_old))
    diffV1 = max(np.linalg.norm(a - b) for a, b in zip(motionField_1, motionField_1_old))
    diffU  = max(np.linalg.norm(a - b) for a, b in zip(reconstructed_image, reconstructed_image_old))

    print("diffV0 is", diffV0)
    print("diffV1 is", diffV1)
    print("diffU is", diffU)


    
    if i>0 and (diffV0<tol and diffV1<tol and diffU<tol):
        print(f"converged for outer iterations in the {i}-th iteration.")
        break
    
    for j in range(len(frames)):
        if((i+1)%1==0):
            
            result_1_v = motionField_0[j].reshape(inputDimension, order="F")
            result_2_v = motionField_1[j].reshape(inputDimension, order="F")
            
            magnitude_v = np.sqrt(result_1_v**2 + result_2_v**2)

            
            vmax1 = np.max(np.abs(result_1_v))
            vmax2 = np.max(np.abs(result_2_v))
            vmax_mag = np.percentile(magnitude_v, 99)

            print(f"vmax1 for i = {i} is ", vmax1)
            print(f"vmax2 for i = {i} is ", vmax2)
            
            result_1_u = reconstructed_image_frames[j].reshape(inputDimension, order="F")

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
            
            
            plt.savefig(f"resultTVTVMassPreservation{i+1},{j+1}.png", dpi=200, bbox_inches="tight")

            plt.close()
    gc.collect()