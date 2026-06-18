import sys
sys.path.append("/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/wrappers/python")

import cv2
import numpy as np
import gc
import matplotlib.pyplot as plt
import os
from wrapper import FlexBoxSolver # python wrapper

num_iterations = 50

video_path = "city.mp4"
name = "City"

denoiseByAveraging = False

resize_value = 0.02#choose a value between 0.0 and 1.0

cut = True
cut_width = [0.0 , 0.0]
cut_height = [0.0 , 0.0]

start_sec = 1.0
end_sec = 1.2
n_frames = 4

figures_frequence = 1


# Parameter
weight_1 = 1.0    # Datenterm u
weight_2 = 0.05  # Regularizer of u
weight_3 = 1.0    # motion term for u
weight_4 = 1.0    # motion term for v
weight_5 = 0.001    # Regularizer of v

tol = 1e-4 #for outer iterations

#parameters for first solver
maxIterations_first_solver = 2000
tol_first_solver = 1e-6
verbose_first_solver = 1

#parameters for second solver
maxIterations_second_solver = 2000
tol_second_solver = 1e-6
verbose_second_solver = 1


dataTerm_norm_list      = ["","L1",             "L2"                    ]
reg_u_norm_list         = ["","TVIso",          "TVAniso",          "L2"]
motionTerm_u_list       = ["","OpticalFlow",    "MassPreservation"      ]
motionTerm_u_norm_list  = ["","L1Iso",          "L1Aniso",          "L2"]
motionTerm_v_list       = ["","OpticalFlow",    "MassPreservation"      ]
motionTerm_v_norm_list  = ["","L1",             "L2"                    ]
reg_v_norm_list         = ["","TVIso",          "TVAniso",          "L2"]


dataTerm_norm = 2 # choose from dataTerm_list above
reg_u_norm = 1
motionTerm_u = 1
motionTerm_u_norm = 2
motionTerm_v = 1
motionTerm_v_norm = 1
reg_v_norm = 2



folder_name = (
    f"result{name}"
    f"{weight_1}{dataTerm_norm_list[dataTerm_norm]}"
    f"{weight_2}{reg_u_norm_list[reg_u_norm]}"
    f"{weight_3}{motionTerm_u_norm_list[motionTerm_u_norm]}"
    f"{motionTerm_u_list[motionTerm_u]}"
    f"{weight_4}{motionTerm_v_norm_list[motionTerm_v_norm]}"
    f"{motionTerm_v_list[motionTerm_v]}"
    f"{weight_5}{reg_v_norm_list[reg_v_norm]}"
)
            
# create thje folder in which we will store the figures
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

file_path = os.path.join(folder_name, "parameters.txt")

# write the parameters in the textfile
with open(file_path, "w", encoding="utf-8") as f:
    f.write(f"the parameters are\nweight_1: {weight_1}\nweight_2: {weight_2}\nweight_3: {weight_3}\nweight_4: {weight_4}\nweight_5: {weight_5}")


if(denoiseByAveraging):
    denoisedImage = np.load("denoisedImage0,8.npy")

cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"There are {fps} frames per second")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))



start_frame = int(start_sec * fps)
end_frame = int(end_sec * fps)
start_frame = max(0, start_frame)
end_frame = min(total_frames - 1, end_frame)

frame_indices = np.linspace(start_frame, end_frame, n_frames).astype(int)
frames = []
target_set = set(frame_indices)


idx = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    if idx in target_set:
        if frame is None:
            idx += 1
            pass

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if cut:
            x1 = int(width * cut_width[0])
            x2 = int(width * (1 - cut_width[1]))
            y1 = int(height * cut_height[0])
            y2 = int(height * (1 - cut_height[1]))
            frame_gray = frame_gray[y1:y2, x1:x2]

        if frame_gray.size == 0:
            idx += 1
            pass
        frame_gray = cv2.resize(frame_gray, None, fx=resize_value, fy=resize_value)
        frame_gray = frame_gray.astype(np.float32) / 255.0
        #frame_gray -= denoisedImage
        #frame_gray = cv2.normalize(frame_gray, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        frames.append(frame_gray)

    idx += 1


if(denoiseByAveraging):
    denoisedImagebyAveraging = frames[0]
    for f in frames:
        denoiseByAveraging = (denoiseByAveraging+f)*0.5
    for i in range(len(frames)):
        frames[i]-=denoiseByAveraging
        frames[i]=cv2.normalize(frames[i], None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


cap.release()

print(f"Extracted {len(frames)} frames")

timeIntervall = len(frames)

nPx = frames[0].shape[0] * frames[0].shape[1]

# Dimensionen (0=Y/Höhe, 1=X/Breite, 2=Z/Zeit)
inputDimension = [frames[0].shape[0], frames[0].shape[1]]
dimensionsWithTime = [inputDimension[0], inputDimension[1], len(frames)]
print(inputDimension)

# Initialisierung der Primalen als C-Contiguous flache Arrays (Inhalt wird im Fortran-Stil verwaltet)
reconstructed_image = np.ascontiguousarray(np.ones(nPx * len(frames), dtype=np.float32) * 0.5)
#motionField_0 = np.ascontiguousarray(np.random.normal(0.0, 0.1, nPx * len(frames)).astype(np.float32))
#motionField_1 = np.ascontiguousarray(np.random.normal(0.0, 0.1, nPx * len(frames)).astype(np.float32))

#motionField_0 = np.ascontiguousarray(np.zeros(nPx * len(frames)).astype(np.float32))
#motionField_1 = np.ascontiguousarray(np.zeros(nPx * len(frames)).astype(np.float32))

# approximate the motionfields by the difference of consecutive frames
motionField_0_list = []
motionField_1_list = []

for j in range(len(frames) - 1):
    diff = frames[j+1] - frames[j]
    motionField_0_list.append(diff.flatten(order="F"))
    motionField_1_list.append(diff.flatten(order="F"))

# last frame is the copy of the second last frame
motionField_0_list.append(motionField_0_list[-1])
motionField_1_list.append(motionField_1_list[-1])

motionField_0 = np.ascontiguousarray(np.concatenate(motionField_0_list), dtype=np.float32)
motionField_1 = np.ascontiguousarray(np.concatenate(motionField_1_list), dtype=np.float32)

# Initialisierung mit minimalem Rauschen (Standardabweichung 0.001)
#motionField_0 = np.ascontiguousarray(
#    np.random.normal(0.0, 0.001, nPx * len(frames)).astype(np.float32)
#)
#motionField_1 = np.ascontiguousarray(
#    np.random.normal(0.0, 0.001, nPx * len(frames)).astype(np.float32)
#)
#motionField_0_ = np.ascontiguousarray(np.concatenate([np.array(frames[i+1]-frames[i]).astype(np.float32) for i in range(len(frames)-1)]))
#motionField_0 = np.concatenate([motionField_0_[0],motionField_0_[1],motionField_0_[2],np.zeros(nPx)])

#motionField_1_ = np.ascontiguousarray(np.concatenate([np.array(frames[i+1]-frames[i]).astype(np.float32) for i in range(len(frames)-1)]))
#motionField_1 = np.concatenate([motionField_1_[0],motionField_1_[1],motionField_1_[2],np.zeros(nPx)])

# Eingangsdaten im Fortran-Stil flachklopfen
framesBig = np.ascontiguousarray(
    np.concatenate([f.flatten(order="F") for f in frames]), 
    dtype=np.float32
)

for i in range(num_iterations):
    # copy for testing convergence
    motionField_0_old = motionField_0.copy()
    motionField_1_old = motionField_1.copy()
    reconstructed_image_old = reconstructed_image.copy()
    
    # first solver, image reconstruction
    solver = FlexBoxSolver(maxIt=maxIterations_first_solver, tol=tol_first_solver, verbose=verbose_first_solver)
    solver.add_primal(reconstructed_image)
    corresponding_primals = [0]

    # Data Term: (0.5) * ||u - f||^2
    if(dataTerm_norm==0):
        pass
    else:
        data_op = {
            "type": "identityOperator",
            "nPx": nPx * len(frames)
        }
        if(dataTerm_norm==1):
            solver.add_dual(
                prox_type="L1proxDualData",
                alpha=weight_1 / timeIntervall,
                f_list=[framesBig],
                corresponding_primals=[corresponding_primals[0]],
                operator_dict=[data_op]
            )
        if(dataTerm_norm==2):
            solver.add_dual(
                prox_type="L2proxDualData",
                alpha=weight_1 / timeIntervall,
                f_list=[framesBig],
                corresponding_primals=[corresponding_primals[0]],
                operator_dict=[data_op]
            )
    
    # spatial TV of u
    if(reg_u_norm==0):
        pass
    else:
        grad_x = {
            "type": "gradientOperator",
            "gradType": "forward",
            "gradDirection": 0,
            "inputDimension": dimensionsWithTime
        }
        grad_y = {
            "type": "gradientOperator",
            "gradType": "forward",
            "gradDirection": 1,
            "inputDimension": dimensionsWithTime
        }
        if(reg_u_norm==1):
            solver.add_dual(
                prox_type="L1IsoProxDual",
                alpha=weight_2 / timeIntervall,
                f_list=[np.ascontiguousarray(np.zeros(nPx * len(frames), dtype=np.float32))],
                corresponding_primals=[corresponding_primals[0]],
                operator_dict=[grad_x, grad_y]
            )
        if(reg_u_norm==2):
            solver.add_dual(
                prox_type="L1AnisoProxDual",
                alpha=weight_2 / timeIntervall,
                f_list=[np.ascontiguousarray(np.zeros(nPx * len(frames), dtype=np.float32))],
                corresponding_primals=[corresponding_primals[0]],
                operator_dict=[grad_x, grad_y]
            )
        if(reg_u_norm==3):
            solver.add_dual(
                prox_type="L2proxDual",
                alpha=weight_2 / timeIntervall,
                f_list=[np.ascontiguousarray(np.zeros(nPx * len(frames), dtype=np.float32))],
                corresponding_primals=[corresponding_primals[0]],
                operator_dict=[grad_x, grad_y]
            )
    
    # motion term for u
    if(motionTerm_u==0):
        pass
    elif(motionTerm_u == 1):
        #optical flow for u
        if(motionTerm_u_norm==0):
            pass
        else:
            upwindGradientOp1 = {
                "type": "upwindGradientOperator",
                "motionField": motionField_0,
                #"gradType": "forward",
                "gradDirection": 0,
                "inputDimension": dimensionsWithTime
            }
            upwindGradientOp2 = {
                "type": "upwindGradientOperator",
                "motionField": motionField_1,
                #"gradType": "forward",
                "gradDirection": 1,
                "inputDimension": dimensionsWithTime
            }
            concatUpwindGradientOp = {
                "type": "concatOperator",
                "operation": "addition",
                "A": upwindGradientOp1,
                "B": upwindGradientOp2,
                "isMinus": False
            }
            timeDerivativeOp = {
                "type": "gradientOperator",
                "gradType": "forward",
                "gradDirection": 2,  # Zeitableitung entlang Achse 2
                "inputDimension": dimensionsWithTime
            }
            concatUpwindGradientTimeDerivativeOp = {
                "type": "concatOperator",
                "operation": "addition",
                "A": concatUpwindGradientOp,
                "B": timeDerivativeOp,
                "isMinus": False
            }
            if(motionTerm_u_norm==1):
                solver.add_dual(
                    prox_type="L1IsoProxDual",
                    alpha=weight_3 / timeIntervall,
                    f_list=[np.ascontiguousarray(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[concatUpwindGradientTimeDerivativeOp]
                )
            if(motionTerm_u_norm==2):
                solver.add_dual(
                    prox_type="L1AnisoProxDual",
                    alpha=weight_3 / timeIntervall,
                    f_list=[np.ascontiguousarray(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[concatUpwindGradientTimeDerivativeOp]
                )
            if(motionTerm_u_norm==3):
                solver.add_dual(
                    prox_type="L2proxDual",
                    alpha=weight_3 / timeIntervall,
                    f_list=[np.ascontiguousarray(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[concatUpwindGradientTimeDerivativeOp]
                )
    elif(motionTerm_u == 2):
        #mass preservation for u
        if(motionTerm_u_norm==0):
            pass
        else:
            upwindDivergenceOp1 = {
                "type": "upwindDivergenceOperator",
                "weights": motionField_0,
                "gradType": "forward",
                "gradDirection": 0,
                "inputDimension": dimensionsWithTime
            }
            upwindDivergenceOp2 = {
                "type": "upwindDivergenceOperator",
                "weights": motionField_1,
                "gradType": "forward",
                "gradDirection": 1,
                "inputDimension": dimensionsWithTime
            }
            concatUpwindDivergenceOp = {
                "type": "concatOperator",
                "operation": "addition",
                "A": upwindDivergenceOp1,
                "B": upwindDivergenceOp2,
                "isMinus": False
            }
            timeDerivativeOp = {
                "type": "gradientOperator",
                "gradType": "forward",
                "gradDirection": 2,  # Zeitableitung entlang Achse 2
                "inputDimension": dimensionsWithTime
            }
            concatUpwindDivergenceTimeDerivativeOp = {
                "type": "concatOperator",
                "operation": "addition",
                "A": concatUpwindDivergenceOp,
                "B": timeDerivativeOp,
                "isMinus": False
            }
            if(motionTerm_u_norm==1):
                solver.add_dual(
                    prox_type="L1IsoProxDual",
                    alpha=weight_3 / timeIntervall,
                    f_list=[np.ascontiguousarray(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[concatUpwindDivergenceTimeDerivativeOp]
                )
            if(motionTerm_u_norm==2):
                solver.add_dual(
                    prox_type="L1AnisoProxDual",
                    alpha=weight_3 / timeIntervall,
                    f_list=[np.ascontiguousarray(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[concatUpwindDivergenceTimeDerivativeOp]
                )
            if(motionTerm_u_norm==3):
                solver.add_dual(
                    prox_type="L2proxDual",
                    alpha=weight_3 / timeIntervall,
                    f_list=[np.ascontiguousarray(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[concatUpwindDivergenceTimeDerivativeOp]
                )
    
    print("before first solver")
    u_out, _ = solver.solve()
    print("after first solver")
    
    #reorder
    reconstructed_image = np.ascontiguousarray(np.array(u_out[0], dtype=np.float32))
    reconstructed_image_frames = np.split(reconstructed_image, len(frames))
    
    del solver
    
    # second solver for v
    solver = FlexBoxSolver(maxIt=maxIterations_second_solver, tol=tol_second_solver, verbose=verbose_second_solver)
    solver.add_primal(motionField_0)
    solver.add_primal(motionField_1)
    corresponding_primals = [0, 1]

    # calcualtion of -u_t as the data for the motion term
    negUT_list = [np.zeros(nPx, dtype=np.float32) for j in range(len(frames))]
    for j in range(len(frames) - 1):
        negUT_list[j] = -(reconstructed_image_frames[j+1] - reconstructed_image_frames[j])
    #negUT_list[len(frames) - 1] = negUT_list[len(frames) - 2]
    negUT = np.ascontiguousarray(np.concatenate(negUT_list), dtype=np.float32)

    # motion term for v
    if(motionTerm_v==0):
        pass
    elif(motionTerm_v == 1):
        #optical flow for v
        if(motionTerm_v_norm==0):
            pass
        else:
            opticalFlowOp1 = {
                "type": "opticalFlowOp",
                "image": reconstructed_image,
                "direction": 0,
                "inputDimension": dimensionsWithTime
            }
            opticalFlowOp2 = {
                "type": "opticalFlowOp",
                "image": reconstructed_image,
                "direction": 1,
                "inputDimension": dimensionsWithTime
            }
            concatMassPreservationOp = {
                "type": "concatOperator",
                "operation": "addition",
                "A": opticalFlowOp1,
                "B": opticalFlowOp2,
                "isMinus": False
            }
            if(motionTerm_v_norm==1):
                solver.add_dual(
                    prox_type="L1proxDualData",
                    alpha=weight_4 / timeIntervall,
                    f_list=[negUT],
                    corresponding_primals=[corresponding_primals[0], corresponding_primals[1]],
                    operator_dict=[opticalFlowOp1,opticalFlowOp2]
                    #operator_dict=[concatMassPreservationOp]
                )
            if(motionTerm_v_norm==2):
                solver.add_dual(
                    prox_type="L2proxDualData",
                    alpha=weight_4 / timeIntervall,
                    f_list=[negUT],
                    corresponding_primals=[corresponding_primals[0], corresponding_primals[1]],
                    operator_dict=[opticalFlowOp1,opticalFlowOp2]
                    #operator_dict=[concatMassPreservationOp]
                )
    elif(motionTerm_v == 2):
        #mass preservation for v
        if(motionTerm_v_norm==0):
            pass
        else:
            massPreservationOp1 = {
                "type": "massPreservationOp",
                "image": reconstructed_image,
                "direction": 0,
                "inputDimension": dimensionsWithTime
            }
            massPreservationOp2 = {
                "type": "massPreservationOp",
                "image": reconstructed_image,
                "direction": 1,
                "inputDimension": dimensionsWithTime
            }
            concatMassPreservationOp = {
                "type": "concatOperator",
                "operation": "addition",
                "A": massPreservationOp1,
                "B": massPreservationOp2,
                "isMinus": False
            }
            if(motionTerm_v_norm==1):
                solver.add_dual(
                    prox_type="L1proxDualData",
                    alpha=weight_4 / timeIntervall,
                    f_list=[negUT],
                    corresponding_primals=[corresponding_primals[0], corresponding_primals[1]],
                    operator_dict=[massPreservationOp1,massPreservationOp2]
                    #operator_dict=[concatMassPreservationOp]
                )
            if(motionTerm_v_norm==2):
                solver.add_dual(
                    prox_type="L2proxDualData",
                    alpha=weight_4 / timeIntervall,
                    f_list=[negUT],
                    corresponding_primals=[corresponding_primals[0], corresponding_primals[1]],
                    operator_dict=[massPreservationOp1,massPreservationOp2]
                    #operator_dict=[concatMassPreservationOp]
                )
    # TV for v
    # spatial TV of u
    if(reg_v_norm==0):
        pass
    else:
        grad_x = {
            "type": "gradientOperator",
            "gradType": "forward",
            "gradDirection": 0,
            "inputDimension": dimensionsWithTime
        }
        grad_y = {
            "type": "gradientOperator",
            "gradType": "forward",
            "gradDirection": 1,
            "inputDimension": dimensionsWithTime
        }
        if(reg_v_norm==1):
            solver.add_dual(
                prox_type="L1IsoProxDual",
                alpha=weight_5 / timeIntervall,
                f_list=[np.ascontiguousarray(np.zeros(nPx * len(frames), dtype=np.float32))],
                corresponding_primals=[corresponding_primals[0]],
                operator_dict=[grad_x,grad_y]
            )
            solver.add_dual(
                prox_type="L1IsoProxDual",
                alpha=weight_5 / timeIntervall,
                f_list=[np.ascontiguousarray(np.zeros(nPx * len(frames), dtype=np.float32))],
                corresponding_primals=[corresponding_primals[1]],
                operator_dict=[grad_x,grad_y]
            )
        if(reg_v_norm==2):
            solver.add_dual(
                prox_type="L1AnisoProxDual",
                alpha=weight_5 / timeIntervall,
                f_list=[np.ascontiguousarray(np.zeros(nPx * len(frames), dtype=np.float32))],
                corresponding_primals=[corresponding_primals[0]],
                operator_dict=[grad_x,grad_y]
            )
            solver.add_dual(
                prox_type="L1AnisoProxDual",
                alpha=weight_5 / timeIntervall,
                f_list=[np.ascontiguousarray(np.zeros(nPx * len(frames), dtype=np.float32))],
                corresponding_primals=[corresponding_primals[1]],
                operator_dict=[grad_x,grad_y]
            )
        if(reg_v_norm==3):
            solver.add_dual(
                prox_type="L1AnisoProxDual",
                alpha=weight_5 / timeIntervall,
                f_list=[np.ascontiguousarray(np.zeros(nPx * len(frames), dtype=np.float32))],
                corresponding_primals=[corresponding_primals[0]],
                operator_dict=[grad_x,grad_y]
            )
            solver.add_dual(
                prox_type="L2proxDual",
                alpha=weight_5 / timeIntervall,
                f_list=[np.ascontiguousarray(np.zeros(nPx * len(frames), dtype=np.float32))],
                corresponding_primals=[corresponding_primals[1]],
                operator_dict=[grad_x,grad_y]
            )
    
    print("before second solver")
    v_out, _ = solver.solve()
    print("after second solver")
    
    motionField_0 = np.ascontiguousarray(np.array(v_out[0], dtype=np.float32))
    motionField_1 = np.ascontiguousarray(np.array(v_out[1], dtype=np.float32))
    motionField_0_frames = np.split(motionField_0, len(frames))
    motionField_1_frames = np.split(motionField_1, len(frames))

    del solver

    # for testing convergence
    diffV0 = np.linalg.norm(motionField_0 - motionField_0_old)
    diffV1 = np.linalg.norm(motionField_1 - motionField_1_old)
    diffU  = np.linalg.norm(reconstructed_image - reconstructed_image_old)

    print("diffV0 is", diffV0)
    print("diffV1 is", diffV1)
    print("diffU is", diffU)
    
    if i > 1 and (diffV0 < tol and diffV1 < tol and diffU < tol):
        print(f"converged for outer iterations in the {i}-th iteration.")
        break
    
    # visualization
    for j in range(len(frames)-1):
        if ((i+1) % figures_frequence == 0):
            result_1_v = motionField_0_frames[j].reshape(inputDimension, order="F")
            result_2_v = motionField_1_frames[j].reshape(inputDimension, order="F")
            result_1_u = reconstructed_image_frames[j].reshape(inputDimension, order="F")

            magnitude_v = np.sqrt(result_1_v**2 + result_2_v**2)
            vmax1 = np.max(np.abs(result_1_v))
            vmax2 = np.max(np.abs(result_2_v))
            vmax_mag = np.percentile(magnitude_v, 99)
            
            magnitude_u = np.sqrt(result_1_u**2)
            umax1 = np.max(np.abs(result_1_u))
            umax_mag = np.percentile(magnitude_u, 99)

            print(f"vmax1 for i = {i} is ", vmax1)
            print(f"vmax2 for i = {i} is ", vmax2)
            
            print(f"umax1 for i = {i} is ", umax1)

            

            plt.figure(figsize=(7,7))
            
            plt.subplot(2,2,1)
            plt.title("magnitude of u")
            plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax_mag)
            plt.axis("off")
            
            plt.subplot(2,2,3)
            plt.title("v_x")
            plt.imshow(result_1_v, cmap="seismic", vmin=-vmax1, vmax=vmax1)
            plt.axis("off")

            plt.subplot(2,2,4)
            plt.title("v_y")
            plt.imshow(result_2_v, cmap="seismic", vmin=-vmax2, vmax=vmax2)
            plt.axis("off")

            plt.subplot(2,2,2)
            plt.title("magnitude of v")
            plt.imshow(magnitude_v, cmap="gray", vmin=0, vmax=vmax_mag)
            plt.axis("off")
            
            #plt.savefig(f"result{dataTerm_norm_list[dataTerm_norm]}{reg_u_norm_list[reg_u_norm]}{motionTerm_u_norm_list[motionTerm_u_norm]}{motionTerm_u_list[motionTerm_u]}{motionTerm_v_norm_list[motionTerm_v_norm]}{motionTerm_v_list[motionTerm_v]}{reg_v_norm_list[reg_v_norm]}{i+1},{j+1}.png", dpi=200, bbox_inches="tight")
            file_path = os.path.join(folder_name, f"{i+1},{j+1}.png")
            plt.savefig(file_path, dpi=200, bbox_inches="tight")
            plt.close()
            
    gc.collect()