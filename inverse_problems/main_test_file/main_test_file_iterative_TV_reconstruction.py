import sys
sys.path.append("/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/wrappers/python")

import cv2
import numpy as np
import glob
import gc
import matplotlib.pyplot as plt
import os
from wrapper import FlexBoxSolver # python wrapper
import errorFunctions as ef
from itertools import product
from skimage.restoration import denoise_nl_means

num_iterations = 20

throughVideoPath = True
video_path = "media3.avi"
image_folder = "MiddleburyDataSet/Hydrangea"
image_folder_gt = "MiddleburyDataSetGroundTruth/Hydrangea"

groundFlowExists = False
if groundFlowExists:
    flo_path = image_folder_gt
best_aae = 10000.0
best_epe = 10000.0
best_aae_epe_combined = [10000.0,10000.0]
best_aae_parameters = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
best_epe_parameters = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
best_aae_epe_combined_parameters = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
name = "BlueMatMedia3OwnReconstr"

if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video '{video_path}' does not exist. Stopping execution.")



resize_value = 0.5#choose a value between 0.0 and 1.0

cut = True
cut_width = [0.5 , 0.0]
cut_height = [0.0 , 0.0]

start_sec = 58.0
end_sec = 60.0
n_frames = 1

"""
cut = True
cut_width = [0.45 , 0.0]
cut_height = [0.0 , 0.0]

start_sec = 58.5
end_sec = 60.5
n_frames = 3
"""

denoiseByAveraging = False
#denoiseByAveraging = False
denoiseWithSameFrames = False

denoise_start_sec = 79.0
denoise_end_sec = 104.8
denoise_n_frames = 170


figures_frequence = 1
errors_frequence = 1

# Parameter
weight_1_all = [1.0]    # Datenterm u
weight_2_all = [22.0]    # Regularizer of u
weight_3_all = [1.0]    # motion term for u
weight_4_all = [1.0]    # motion term for v
weight_5_all = [1.0]   # Regularizer of v


tol = 1e-4 #for outer iterations

#parameters for first solver
maxIterations_first_solver = 100
tol_first_solver = 1e-6
verbose_first_solver = 1

#parameters for second solver
maxIterations_second_solver = 10
tol_second_solver = 1e-6
verbose_second_solver = 1


dataTerm_norm_list      = ["","L1",             "L2"                    ]
reg_u_norm_list         = ["","TVIso",          "TVAniso",          "L2"]
motionTerm_u_list       = ["","OpticalFlow",    "MassPreservation"      ]
motionTerm_u_norm_list  = ["","L1Iso",          "L1Aniso",          "L2"]
motionTerm_u_upwind     = ["",True, False]
motionTerm_v_list       = ["","OpticalFlow",    "MassPreservation"      ]
motionTerm_v_norm_list  = ["","L1",             "L2"                    ]
reg_v_norm_list         = ["","TVIso",          "TVAniso",          "L2"]


dataTerm_norm_all = [2] # choose from dataTerm_list above
reg_u_norm_all = [1]
motionTerm_u_all = [0]
motionTerm_u_norm_all = [0]
motionTerm_u_upwind_all = [1]
motionTerm_v_all = [0]
motionTerm_v_norm_all = [0]
reg_v_norm_all = [0]


def read_flo_file(file_path):
    with open(file_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            raise ValueError(f"Ungültiges .flo Format (Magic Number falsch) in {file_path}")
        
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        
        # Daten einlesen: abwechselnd vx, vy für jedes Pixel
        data = np.fromfile(f, np.float32, count=2 * w * h)
        flow = data.reshape((h, w, 2))
        
        # In separate vx und vy Maps aufteilen
        return flow[:, :, 0], flow[:, :, 1]


folder_name_main = (f"result{name}")

            
# create the folder in which we will store the figures
if not os.path.exists(folder_name_main):
    os.makedirs(folder_name_main)

file_path = os.path.join(folder_name_main, f"errors{name}.txt")
"""
# write the parameters in the textfile
with open(file_path, "w", encoding="utf-8") as f:
    f.write(f"the parameters are\nweight_1: {weight_1}\nweight_2: {weight_2}\nweight_3: {weight_3}\nweight_4: {weight_4}\nweight_5: {weight_5}")
"""
cap = cv2.VideoCapture(video_path)
if throughVideoPath:
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
            #frame_gray = cv2.fastNlMeansDenoisingMulti(frame_gray, imgToDenoiseIndex=2, temporalWindowSize=5, h=7, templateWindowSize=7, searchWindowSize=21)
            #frame_gray = cv2.fastNlMeansDenoising(src=frame_gray, dst=None, h=35, templateWindowSize=7,searchWindowSize=21)
            frame_gray = frame_gray.astype(np.float32)
            frames.append(frame_gray)

        idx += 1
    #frames = cv2.fastNlMeansDenoisingMulti(frames, imgToDenoiseIndex=2, temporalWindowSize=5, h=7, templateWindowSize=7, searchWindowSize=21)
    #for f in frames:
    #    f=f.astype(np.float32)

else:
    png_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.png')])

    # check if frames are found
    if len(png_files) == 0:
        raise FileNotFoundError(f"no png files found in '{image_folder}'")

    print(f"{len(png_files)} png files were found")

    frames = []

    for file_name in png_files:
        file_path = os.path.join(image_folder, file_name)
        
        frame_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if frame_gray is None:
            print(f"Warning: could not read {file_name}. We skip.")
            continue

        # Dimensionen des aktuellen Bildes holen (für das Zuschneiden)
        height, width = frame_gray.shape

        # Optionales Zuschneiden (Cutton), falls im Skript aktiviert
        if cut:
            x1 = int(width * cut_width[0])
            x2 = int(width * (1 - cut_width[1]))
            y1 = int(height * cut_height[0])
            y2 = int(height * (1 - cut_height[1]))
            frame_gray = frame_gray[y1:y2, x1:x2]

        # Scale
        frame_gray = cv2.resize(frame_gray, None, fx=resize_value, fy=resize_value)
        
        # normalize to [0.0, 1.0] 
        frame_gray = frame_gray.astype(np.float32)
        
        frames.append(frame_gray)

    # Aktualisiere die Frame-Anzahl-Variable für den restlichen Solver-Code
    n_frames = len(frames)
    print(f"succesfully loaded {n_frames} frames.")


print(f"Extracted {len(frames)} frames")

timeIntervall = len(frames)

nPx = frames[0].shape[0] * frames[0].shape[1]

# Dimensionen (0=Y/Höhe, 1=X/Breite, 2=Z/Zeit)
inputDimension = [frames[0].shape[0], frames[0].shape[1]]
dimensionsWithTime = [inputDimension[0], inputDimension[1], len(frames)]
print(inputDimension)



# save the frames that we start with
for i in range(len(frames)):
    result_1_u = frames[i].reshape(inputDimension, order="F")
    magnitude_u = np.sqrt(result_1_u**2)

    umax1 = np.max(np.abs(result_1_u))
    umax_mag = np.percentile(magnitude_u, 99)

    plt.figure(figsize=(7,7))

    plt.title(f"magnitude of u_{i+1}")
    plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax_mag)
    plt.axis("off")

    file_path = os.path.join(folder_name_main, f"{name},{i}.png")
    plt.savefig(file_path, dpi=200, bbox_inches="tight")

    plt.close()



if(denoiseByAveraging):
    if denoiseWithSameFrames:
        denoisedImagebyAveraging = np.zeros(nPx)
        for f in frames:
            denoisedImagebyAveraging += f
        denoisedImagebyAveraging=denoisedImagebyAveraging/(float(len(frames)))
        for i in range(len(frames)):
            frames[i]-=denoisedImagebyAveraging
            frames[i]=cv2.normalize(frames[i], None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            #frames[i] = cv2.fastNlMeansDenoising(frames[i], None, 20, 7, 15)
    else:
        denoisedImagebyAveraging = np.zeros(inputDimension, dtype=np.float32)

        denoise_start_frame = int(denoise_start_sec * fps)
        denoise_end_frame = int(denoise_end_sec * fps)
        denoise_start_frame = max(0, denoise_start_frame)
        denoise_end_frame = min(total_frames - 1, denoise_end_frame)

        denoise_frame_indices = np.arange(denoise_start_frame, denoise_end_frame + 1)
        denoise_frames = []
        target_set = set(denoise_frame_indices)


        idx = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
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
                frame_gray = frame_gray.astype(np.float32)
                #frame_gray -= denoisedImage
                #frame_gray = cv2.normalize(frame_gray, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                denoisedImagebyAveraging += frame_gray

            idx += 1

        denoisedImagebyAveraging = denoisedImagebyAveraging/float(len(denoise_frame_indices))

        for i in range(len(frames)):
            frames[i]-=denoisedImagebyAveraging
            frames[i]+=np.ones(frames[i].shape)*255.0
            frames[i]*=0.5
            #frames[i] = cv2.fastNlMeansDenoising(src=frames[i], dst=None, h=10, templateWindowSize=5,searchWindowSize=15)
            #frames[i]+=np.ones(nPx).reshape(inputDimension, order="F")
            #frames[i]=cv2.normalize(frames[i], None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


    # save the denoised frames that we start with
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

cap.release()

# Initialisierung der Primalen als C-Contiguous flache Arrays (Inhalt wird im Fortran-Stil verwaltet)
reconstructed_image = np.concatenate([frames[i].flatten(order="F") for i in range(len(frames))]) 
#motionField_0 = ((np.random.normal(0.0, 0.1, nPx * len(frames)).astype(np.float32))
#motionField_1 = ((np.random.normal(0.0, 0.1, nPx * len(frames)).astype(np.float32))

#motionField_0 = ((np.zeros(nPx * len(frames)).astype(np.float32))
#motionField_1 = ((np.zeros(nPx * len(frames)).astype(np.float32))

# approximate the motionfields by the difference of consecutive frames

motionField_0_list = []
motionField_1_list = []
"""
for j in range(len(frames) - 1):
    diff = frames[j+1] - frames[j]
    motionField_0_list.append(diff.flatten(order="F"))
    motionField_1_list.append(diff.flatten(order="F"))

# last frame is the copy of the second last frame
motionField_0_list.append(motionField_0_list[-1])
motionField_1_list.append(motionField_1_list[-1])

"""
motionField_0_list = [np.zeros(nPx) for i in range(len(frames))]
motionField_1_list = [np.zeros(nPx) for i in range(len(frames))]

motionField_0 = np.concatenate(motionField_0_list)
motionField_1 = np.concatenate(motionField_1_list)



#motionField_0 = ((
#    np.random.normal(0.0, 0.001, nPx * len(frames)).astype(np.float32)
#)
#motionField_1 = ((
#    np.random.normal(0.0, 0.001, nPx * len(frames)).astype(np.float32)
#)
#motionField_0_ = ((np.concatenate([np.array(frames[i+1]-frames[i]).astype(np.float32) for i in range(len(frames)-1)]))
#motionField_0 = np.concatenate([motionField_0_[0],motionField_0_[1],motionField_0_[2],np.zeros(nPx)])

#motionField_1_ = ((np.concatenate([np.array(frames[i+1]-frames[i]).astype(np.float32) for i in range(len(frames)-1)]))
#motionField_1 = np.concatenate([motionField_1_[0],motionField_1_[1],motionField_1_[2],np.zeros(nPx)])


framesBig = np.concatenate([f.flatten(order="F") for f in frames])
"""
# save the frames that we start with
for i in range(len(frames)):
    result_1_u = framesBig[i].reshape(inputDimension, order="F")
    magnitude_u = np.sqrt(result_1_u**2)

    umax1 = np.max(np.abs(result_1_u))
    umax_mag = np.percentile(magnitude_u, 99)

    plt.figure(figsize=(7,7))

    plt.title("magnitude of u")
    plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax_mag)
    plt.axis("off")

    file_path = os.path.join(folder_name, f"{name},{i}.png")
    plt.savefig(file_path, dpi=200, bbox_inches="tight")

    plt.close()
"""
frames_versions = [np.zeros(nPx*len(frames)) for i in range(num_iterations)]

for weight_1, weight_2, weight_3, weight_4, weight_5, dataTerm_norm, reg_u_norm, motionTerm_u, motionTerm_u_norm, motionTerm_u_upwind, motionTerm_v, motionTerm_v_norm, reg_v_norm in product(weight_1_all, weight_2_all, weight_3_all, weight_4_all, weight_5_all, dataTerm_norm_all, reg_u_norm_all, motionTerm_u_all, motionTerm_u_norm_all, motionTerm_u_upwind_all, motionTerm_v_all, motionTerm_v_norm_all, reg_v_norm_all):
    for i in range(num_iterations):
        print(f"it's the {i}. outer iteration")
        framesBig_versions = framesBig.copy()
        for j in range(i):
            framesBig_versions -= frames_versions[j]

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
                    f_list=[framesBig_versions],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[data_op]
                )
            if(dataTerm_norm==2):
                solver.add_dual(
                    prox_type="L2proxDualData",
                    alpha=weight_1 / timeIntervall,
                    f_list=[framesBig_versions],
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
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[grad_x, grad_y]
                )
            if(reg_u_norm==2):
                solver.add_dual(
                    prox_type="L1AnisoProxDual",
                    alpha=weight_2 / timeIntervall,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[grad_x, grad_y]
                )
            if(reg_u_norm==3):
                solver.add_dual(
                    prox_type="L2proxDual",
                    alpha=weight_2 / timeIntervall,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
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
                weightedGradientOp1 = {
                    "type": "weightedGradientOperator",
                    "weights": motionField_0,
                    "gradType": "forward",
                    "gradDirection": 0,
                    "inputDimension": dimensionsWithTime
                }
                weightedGradientOp2 = {
                    "type": "weightedGradientOperator",
                    "weights": motionField_1,
                    "gradType": "forward",
                    "gradDirection": 1,
                    "inputDimension": dimensionsWithTime
                }
                concatGradientOp = {}
                if(motionTerm_u_upwind==1):
                    concatGradientOp = {
                    "type": "concatOperator",
                    "operation": "addition",
                    "A": upwindGradientOp1,
                    "B": upwindGradientOp2,
                    "isMinus": False
                    }
                else:
                    concatGradientOp = {
                    "type": "concatOperator",
                    "operation": "addition",
                    "A": weightedGradientOp1,
                    "B": weightedGradientOp2,
                    "isMinus": False
                    }
                
                timeDerivativeOp = {
                    "type": "gradientOperator",
                    "gradType": "forward",
                    "gradDirection": 2,  # Zeitableitung entlang Achse 2
                    "inputDimension": dimensionsWithTime,
                    "isMinus": False
                }
                weightedTimeDerivativeOp = {
                    "type": "weightedGradientOperator",
                    "weights": np.ones(nPx*len(frames))*0.05,
                    "gradType": "forward",
                    "gradDirection": 2,  # Zeitableitung entlang Achse 2
                    "inputDimension": dimensionsWithTime,
                    "isMinus": False
                }
                concatGradientTimeDerivativeOp = {
                    "type": "concatOperator",
                    "operation": "addition",
                    "A": concatGradientOp,
                    "B": timeDerivativeOp,
                    "isMinus": False
                }
                concatWeightedGradientTimeDerivativeOp = {
                    "type": "concatOperator",
                    "operation": "addition",
                    "A": concatGradientOp,
                    "B": weightedTimeDerivativeOp,
                    "isMinus": False
                }
                if(motionTerm_u_norm==1):
                    solver.add_dual(
                        prox_type="L1IsoProxDual",
                        alpha=weight_3 / timeIntervall,
                        f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0]],
                        operator_dict=[concatGradientTimeDerivativeOp]
                    )
                    """
                    solver.add_dual(
                        prox_type="L1IsoProxDual",
                        alpha=weight_3 / timeIntervall,
                        f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0],corresponding_primals[0],corresponding_primals[0]],
                        operator_dict=[upwindGradientOp1,upwindGradientOp2,timeDerivativeOp]
                    )
                    """
                if(motionTerm_u_norm==2):
                    solver.add_dual(
                        prox_type="L1AnisoProxDual",
                        alpha=weight_3 / timeIntervall,
                        f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0]],
                        operator_dict=[concatGradientTimeDerivativeOp]
                    )
                    """
                    solver.add_dual(
                        prox_type="L1AnisoProxDual",
                        alpha=weight_3 / timeIntervall,
                        f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0],corresponding_primals[0],corresponding_primals[0]],
                        operator_dict=[upwindGradientOp1,upwindGradientOp2,timeDerivativeOp]
                    )
                    """
                if(motionTerm_u_norm==3):
                    solver.add_dual(
                        prox_type="L2proxDual",
                        alpha=weight_3 / timeIntervall,
                        f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0]],
                        operator_dict=[concatGradientTimeDerivativeOp]
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

                weightedDivergenceOp1 = {
                    "type": "weightedDivergenceOperator",
                    "weights": motionField_0,
                    "gradType": "forward",
                    "gradDirection": 0,
                    "inputDimension": dimensionsWithTime
                }
                weightedDivergenceOp2 = {
                    "type": "weightedDivergenceOperator",
                    "weights": motionField_1,
                    "gradType": "forward",
                    "gradDirection": 1,
                    "inputDimension": dimensionsWithTime
                }
                concatDivergenceOp = {}
                if(motionTerm_u_upwind==1):
                    concatDivergenceOp = {
                    "type": "concatOperator",
                    "operation": "addition",
                    "A": upwindDivergenceOp1,
                    "B": upwindDivergenceOp2,
                    "isMinus": False
                    }
                else:
                    concatDivergenceOp = {
                    "type": "concatOperator",
                    "operation": "addition",
                    "A": weightedDivergenceOp1,
                    "B": weightedDivergenceOp2,
                    "isMinus": False
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
                concatDivergenceTimeDerivativeOp = {
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
                        f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0]],
                        operator_dict=[concatDivergenceTimeDerivativeOp]
                    )
                    """
                    solver.add_dual(
                        prox_type="L1IsoProxDual",
                        alpha=weight_3 / timeIntervall,
                        f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0],corresponding_primals[0],corresponding_primals[0]],
                        operator_dict=[upwindDivergenceOp1,upwindDivergenceOp2,timeDerivativeOp]
                    )
                    """
                if(motionTerm_u_norm==2):
                    solver.add_dual(
                        prox_type="L1AnisoProxDual",
                        alpha=weight_3 / timeIntervall,
                        f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0]],
                        operator_dict=[concatDivergenceTimeDerivativeOp]
                    )
                    """
                    solver.add_dual(
                        prox_type="L1AnisoProxDual",
                        alpha=weight_3 / timeIntervall,
                        f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0],corresponding_primals[0],corresponding_primals[0]],
                        operator_dict=[upwindDivergenceOp1,upwindDivergenceOp2,timeDerivativeOp]
                    )
                    """
                if(motionTerm_u_norm==3):
                    solver.add_dual(
                        prox_type="L2proxDual",
                        alpha=weight_3 / timeIntervall,
                        f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                        corresponding_primals=[corresponding_primals[0]],
                        operator_dict=[concatDivergenceTimeDerivativeOp]
                    )
        
        print("before first solver")
        u_out, _ = solver.solve()
        print("after first solver")

        #reorder
        reconstructed_image = np.array(u_out[0], dtype=np.float32)

        frames_versions[i] = reconstructed_image.copy()
        
        del solver

        reconstructed_image_frames = np.split(reconstructed_image, len(frames))
        
        # second solver for v
        solver = FlexBoxSolver(maxIt=maxIterations_second_solver, tol=tol_second_solver, verbose=verbose_second_solver)
        solver.add_primal(motionField_0)
        solver.add_primal(motionField_1)
        corresponding_primals = [0, 1]

        # calcualtion of -u_t as the data for the motion term
        negUT_list = [np.zeros(nPx, dtype=np.float32) for j in range(len(frames))]
        for j in range(len(frames) - 1):
            negUT_list[j] = -(reconstructed_image_frames[j+1] - reconstructed_image_frames[j]).flatten(order="F")
        #negUT_list[len(frames) - 1] = negUT_list[len(frames) - 2]
        negUT = np.concatenate(negUT_list)

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
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[grad_x,grad_y]
                )
                solver.add_dual(
                    prox_type="L1IsoProxDual",
                    alpha=weight_5 / timeIntervall,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[1]],
                    operator_dict=[grad_x,grad_y]
                )
            if(reg_v_norm==2):
                solver.add_dual(
                    prox_type="L1AnisoProxDual",
                    alpha=weight_5 / timeIntervall,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[grad_x,grad_y]
                )
                solver.add_dual(
                    prox_type="L1AnisoProxDual",
                    alpha=weight_5 / timeIntervall,
                    #alpha=5.0 / timeIntervall,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[1]],
                    operator_dict=[grad_x,grad_y]
                )
            if(reg_v_norm==3):
                solver.add_dual(
                    prox_type="L2proxDual",
                    alpha=weight_5 / timeIntervall,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[grad_x,grad_y]
                )
                solver.add_dual(
                    prox_type="L2proxDual",
                    alpha=weight_5 / timeIntervall,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[1]],
                    operator_dict=[grad_x,grad_y]
                )
        
        print("before second solver")
        v_out, _ = solver.solve()
        print("after second solver")
        
        motionField_0 = np.array(v_out[0], dtype=np.float32)
        motionField_1 = np.array(v_out[1], dtype=np.float32)
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

        #tol_first_solver = (diffV0 + diffV1 + diffU)/(2*nPx*len(frames))
        #tol_second_solver = (diffV0 + diffV1 + diffU)/(2*nPx*len(frames))

        folder_name = ""
        if(motionTerm_u_upwind==1):
            folder_name = (
                f"result{name}"
                f"resizeValue{resize_value}"
                f"{weight_1}{dataTerm_norm_list[dataTerm_norm]}"
                f"{weight_2}{reg_u_norm_list[reg_u_norm]}"
                f"{weight_3}{motionTerm_u_norm_list[motionTerm_u_norm]}"
                "Upwind"
                f"{motionTerm_u_list[motionTerm_u]}"
                f"{weight_4}{motionTerm_v_norm_list[motionTerm_v_norm]}"
                f"{motionTerm_v_list[motionTerm_v]}"
                f"{weight_5}{reg_v_norm_list[reg_v_norm]}"
            )
        else:
            folder_name = (
                f"result{name}"
                f"resizeValue{resize_value}"
                f"{weight_1}{dataTerm_norm_list[dataTerm_norm]}"
                f"{weight_2}{reg_u_norm_list[reg_u_norm]}"
                f"{weight_3}{motionTerm_u_norm_list[motionTerm_u_norm]}"
                f"{motionTerm_u_list[motionTerm_u]}"
                f"{weight_4}{motionTerm_v_norm_list[motionTerm_v_norm]}"
                f"{motionTerm_v_list[motionTerm_v]}"
                f"{weight_5}{reg_v_norm_list[reg_v_norm]}"
            )

        folder_path = os.path.join(folder_name_main, folder_name)

        os.makedirs(folder_path, exist_ok=True)
        
        if i > 2 and diffV0<tol and diffV1<tol and diffU<tol:
            print(f"converged for outer iterations in the {i}-th iteration.")
            for j in range(len(frames)-1):
                result_1_v = motionField_0_frames[j].reshape(inputDimension, order="F")
                result_2_v = motionField_1_frames[j].reshape(inputDimension, order="F")
                result_1_u = reconstructed_image_frames[j].reshape(inputDimension, order="F")

                #frame_start = j * nPx
                #frame_end = (j + 1) * nPx
                #result_1_u = reconstructed_image[frame_start:frame_end].reshape(inputDimension, order="F")

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
                plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax1)
                #plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax_mag)
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
                
                #plt.savefig(f"result{dataTerm_norm_list[dataTerm_norm]}{reg_u_norm_list[reg_u_norm]}{motionTerm_u_norm_list[motionTerm_u_norm]}{motionTerm_u_list[motionTerm_u]}{motionTerm_v_norm_list[motionTerm_v_norm]}{motionTerm_v_list[motionTerm_v]}{reg_v_norm_list[reg_v_norm]}{i+1},{j+1}.png", dpi=200, bbox_inches="tight")
                file_path = os.path.join(folder_path, f"Converged{i+1},{j+1}.png")
                plt.savefig(file_path, dpi=200, bbox_inches="tight")
                plt.close()
            break

        # visualization
        for j in range(len(frames)):
            if ((i+1) % figures_frequence == 0):
                current_reconstruction = np.zeros(nPx*len(frames))
                for k in range(i+1):
                    current_reconstruction += frames_versions[k]
                result_1_v = motionField_0_frames[j].reshape(inputDimension, order="F")
                result_2_v = motionField_1_frames[j].reshape(inputDimension, order="F")
                result_1_u = reconstructed_image_frames[j].reshape(inputDimension, order="F")
                result_1_current_reconstruction = current_reconstruction.reshape(inputDimension, order="F")


                #frame_start = j * nPx
                #frame_end = (j + 1) * nPx
                #result_1_u = reconstructed_image[frame_start:frame_end].reshape(inputDimension, order="F")

                magnitude_v = np.sqrt(result_1_v**2 + result_2_v**2)
                vmax1 = np.max(np.abs(result_1_v))
                vmax1_mag = np.percentile(np.abs(result_1_v), 99)
                vmax2 = np.max(np.abs(result_2_v))
                vmax2_mag = np.percentile(np.abs(result_2_v), 99)
                vmax_mag = np.percentile(magnitude_v, 99)
                
                umax1 = np.max(np.abs(result_1_u))
                umax_mag = np.percentile(magnitude_u, 99)

                magnitude_current_reconstruction = np.sqrt(result_1_current_reconstruction**2)
                umax_current_reconstruction = np.max(np.abs(result_1_current_reconstruction))
                

                print(f"vmax1 for i = {i} is ", vmax1)
                print(f"vmax2 for i = {i} is ", vmax2)
                print(f"vmax1_mag for i = {i} is ", vmax1_mag)
                print(f"vmax2_mag for i = {i} is ", vmax2_mag)
                
                print(f"umax1 for i = {i} is ", umax1)   
                print(f"u_current_reconstruction for i = {i} is ", umax_current_reconstruction)          

                plt.figure(figsize=(7,7))
                
                plt.subplot(1,2,1)
                plt.title("current one")
                plt.imshow(result_1_u, cmap="seismic", vmin=-umax1, vmax=umax1)
                #plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax_mag)
                plt.axis("off")
                
                plt.subplot(1,2,2)
                plt.title("summed images")
                plt.imshow(magnitude_current_reconstruction, cmap="gray", vmin=0, vmax=umax_current_reconstruction)
                plt.axis("off")
                
                #plt.savefig(f"result{dataTerm_norm_list[dataTerm_norm]}{reg_u_norm_list[reg_u_norm]}{motionTerm_u_norm_list[motionTerm_u_norm]}{motionTerm_u_list[motionTerm_u]}{motionTerm_v_norm_list[motionTerm_v_norm]}{motionTerm_v_list[motionTerm_v]}{reg_v_norm_list[reg_v_norm]}{i+1},{j+1}.png", dpi=200, bbox_inches="tight")
                file_path = os.path.join(folder_path, f"{i+1},{j+1}.png")
                plt.savefig(file_path, dpi=200, bbox_inches="tight")
                plt.close()
            """
            if ((i+1) == num_iterations):
                result_1_v = motionField_0_frames[j].reshape(inputDimension, order="F")
                #result_1_v = result_1_v.astype(np.uint8)
                #result_1_v = cv2.fastNlMeansDenoising(src=result_1_v, dst=None, h=14, templateWindowSize=5,searchWindowSize=15)
                
                result_2_v = motionField_1_frames[j].reshape(inputDimension, order="F")
                #result_2_v = result_2_v.astype(np.uint8)
                #result_2_v = cv2.fastNlMeansDenoising(src=result_2_v, dst=None, h=14, templateWindowSize=5,searchWindowSize=15)


                result_1_v = denoise_nl_means(
                    result_1_v, 
                    h=0.05, 
                    patch_size=5, 
                    patch_distance=15, 
                    fast_mode=True
                )

                result_2_v = denoise_nl_means(
                    result_2_v, 
                    h=0.05, 
                    patch_size=5, 
                    patch_distance=15, 
                    fast_mode=True
                )
                #frame_start = j * nPx
                #frame_end = (j + 1) * nPx
                #result_1_u = reconstructed_image[frame_start:frame_end].reshape(inputDimension, order="F")

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
                plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax1)
                #plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax_mag)
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
                
                #plt.savefig(f"result{dataTerm_norm_list[dataTerm_norm]}{reg_u_norm_list[reg_u_norm]}{motionTerm_u_norm_list[motionTerm_u_norm]}{motionTerm_u_list[motionTerm_u]}{motionTerm_v_norm_list[motionTerm_v_norm]}{motionTerm_v_list[motionTerm_v]}{reg_v_norm_list[reg_v_norm]}{i+1},{j+1}.png", dpi=200, bbox_inches="tight")
                file_path = os.path.join(folder_path, f"motionFieldDenoised{i+1},{j+1}.png")
                plt.savefig(file_path, dpi=200, bbox_inches="tight")
                plt.close()
        """
        # visualization
        #if ((i+1) % errors_frequence == 0 or (i+1)==num_iterations):
        if ((i+1)==num_iterations):
            if groundFlowExists:
                flo_files = [f for f in os.listdir(image_folder_gt) if f.lower().endswith('.flo')]
                flo_path = os.path.join(image_folder_gt, flo_files[0])
                # use the same cropping as for the images
                gt_vx, gt_vy = read_flo_file(flo_path)
                if cut:
                    h_orig, w_orig = gt_vx.shape
                    x1 = int(w_orig * cut_width[0])
                    x2 = int(w_orig * (1 - cut_width[1]))
                    y1 = int(h_orig * cut_height[0])
                    y2 = int(h_orig * (1 - cut_height[1]))
                    gt_vx = gt_vx[y1:y2, x1:x2]
                    gt_vy = gt_vy[y1:y2, x1:x2]
                
                if(resize_value!=1.0):
                    #resize
                    gt_vx = cv2.resize(gt_vx, None, fx=resize_value, fy=resize_value)
                    gt_vy = cv2.resize(gt_vy, None, fx=resize_value, fy=resize_value)
                    
                invalid_mask = (np.abs(gt_vx) > 1e8) | (np.abs(gt_vy) > 1e8)
                valid_mask = ~invalid_mask
                
                """
                calc_vx = motionField_1[:nPx].reshape(gt_vx.shape)
                calc_vy = motionField_0[:nPx].reshape(gt_vy.shape)

                calc_vx = motionField_0[:nPx].reshape(gt_vx.shape, order="F")
                calc_vy = motionField_1[:nPx].reshape(gt_vy.shape, order="F")
                """
                calc_vx = motionField_1[:nPx].reshape(gt_vx.shape, order="F")
                calc_vy = motionField_0[:nPx].reshape(gt_vy.shape, order="F")

                print("gt shape:", gt_vx.shape)
                print("motionField shape:", motionField_0.shape)
                """
                calc_vx = motionField_1[:nPx].reshape(gt_vx.shape, order="F")
                calc_vy = -motionField_0[:nPx].reshape(gt_vy.shape, order="F")
                """
                epe = ef.compute_epe(calc_vx,calc_vy,gt_vx,gt_vy,valid_mask)
                aae = ef.compute_aae(calc_vx,calc_vy,gt_vx,gt_vy,valid_mask)

                if (epe<best_epe):
                    best_epe_parameters = [weight_1, weight_2, weight_3, weight_4, weight_5, dataTerm_norm, reg_u_norm, motionTerm_u, motionTerm_u_norm, motionTerm_u_upwind, motionTerm_v, motionTerm_v_norm, reg_v_norm]
                    best_epe=epe
                

                if (aae<best_aae):
                    best_aae_parameters = [weight_1, weight_2, weight_3, weight_4, weight_5, dataTerm_norm, reg_u_norm, motionTerm_u, motionTerm_u_norm, motionTerm_u_upwind, motionTerm_v, motionTerm_v_norm, reg_v_norm]
                    best_aae=aae
                

                if (aae+epe<best_aae_epe_combined[0] + best_aae_epe_combined[1]):
                    best_aae_epe_combined_parameters = [weight_1, weight_2, weight_3, weight_4, weight_5, dataTerm_norm, reg_u_norm, motionTerm_u, motionTerm_u_norm, motionTerm_u_upwind, motionTerm_v, motionTerm_v_norm, reg_v_norm]
                    best_aae_epe_combined=[aae,epe]
                
                """
                mag_calc = cv2.magnitude(calc_vx, calc_vy)
                mag_gt   = cv2.magnitude(gt_vx, gt_vy)
                
                numerator = calc_vx * gt_vx + calc_vy * gt_vy + 1.0
                denominator = np.sqrt(mag_calc**2 + 1.0) * np.sqrt(mag_gt**2 + 1.0)
                
                clipped_cos = np.clip(numerator / denominator, -1.0, 1.0)
                mean_aee_rad = np.mean(np.arccos(clipped_cos))
                mean_aee_deg = np.degrees(mean_aee_rad)
                
                print("\n========= METRICS EVALUATION (.flo) =========")
                print(f"Verglichene Datei           : {flo_files[0]}")
                print(f"sum(alid_mask)/(valid_mask.size)     : {np.sum(valid_mask)} / {valid_mask.size}")
                print(f"Average Endpoint Error (EPE): {mean_epe:.4f} Pixel")
                print(f"Average Angular Error (AAE)  : {mean_aee_deg:.4f}° Grad")
                print("=============================================")
                """
                error_file_path = os.path.join(folder_name_main, f"errors{name}.txt")
                # Ergebnisse an die parameters.txt anhängen
                with open(error_file_path, "a", encoding="utf-8") as f:
                    f.write(f"weight for {dataTerm_norm_list[dataTerm_norm]} data term is {weight_1}\nweight for the {reg_u_norm_list[reg_u_norm]} regularizer of the image u is {weight_2}\nweight for the {motionTerm_u_norm_list[motionTerm_u_norm]} {motionTerm_u_list[motionTerm_u]} motion term of u is {weight_3}\nweight for the {motionTerm_v_norm_list[motionTerm_v_norm]} {motionTerm_v_list[motionTerm_v]} motion term for v is {weight_4}\nweight for the {reg_v_norm_list[reg_v_norm]} regularizer of the motion fields v is {weight_5}\nThe error values are: \n---Errors at outer iteration {i+1}--- \nEPE: {epe:.4f}\nAEE (Deg): {aae:.4f}\n\n")
                    #f.write(f"weight for the {motionTerm_v_norm_list[motionTerm_v_norm]} {motionTerm_v_list[motionTerm_v]} motion term for v is {weight_4}\nweight for the {reg_v_norm_list[reg_v_norm]} regularizer of the motion fields v is {weight_5}\nThe error values are: \n---Errors at outer iteration {i+1}--- \nEPE: {epe:.4f}\nAEE (Deg): {aae:.4f}\n\n")
        gc.collect()


error_file_path = os.path.join(folder_name_main, f"errors{name}.txt")
# Ergebnisse an die parameters.txt anhängen
if groundFlowExists:
    with open(error_file_path, "a", encoding="utf-8") as f:
        f.write(
            f"The best epe is achieved for:\n"
            f"weight for {dataTerm_norm_list[best_epe_parameters[5]]} data term is {best_epe_parameters[0]}\n"
            f"weight for the {reg_u_norm_list[best_epe_parameters[6]]} regularizer of the image u is {best_epe_parameters[1]}\n"
            f"weight for the {motionTerm_u_norm_list[best_epe_parameters[8]]} {motionTerm_u_list[best_epe_parameters[7]]} motion term of u is {best_epe_parameters[2]}\n"
            f"weight for the {motionTerm_v_norm_list[best_epe_parameters[11]]} {motionTerm_v_list[best_epe_parameters[10]]} motion term for v is {best_epe_parameters[3]}\n"
            f"weight for the {reg_v_norm_list[best_epe_parameters[12]]} regularizer of the motion fields v is {best_epe_parameters[4]}\n"
            f"The epe is: {best_epe:.4f}\n\n"
        )
        f.write(
            f"The best aae is {best_aae:.4f} and is achieved for:\n"
            f"weight for {dataTerm_norm_list[best_aae_parameters[5]]} data term is {best_aae_parameters[0]}\n"
            f"weight for the {reg_u_norm_list[best_aae_parameters[6]]} regularizer of the image u is {best_aae_parameters[1]}\n"
            f"weight for the {motionTerm_u_norm_list[best_aae_parameters[8]]} {motionTerm_u_list[best_aae_parameters[7]]} motion term of u is {best_aae_parameters[2]}\n"
            f"weight for the {motionTerm_v_norm_list[best_aae_parameters[11]]} {motionTerm_v_list[best_aae_parameters[10]]} motion term for v is {best_aae_parameters[3]}\n"
            f"weight for the {reg_v_norm_list[best_aae_parameters[12]]} regularizer of the motion fields v is {best_aae_parameters[4]}\n"
            f"The aae is: {best_aae:.4f}\n\n"
        )
        f.write(
            f"The best combined epe and aae is {best_aae_epe_combined[0]:.4f} (epe), {best_aae_epe_combined[1]:.4f} (aae) and is achieved for:\n"
            f"weight for {dataTerm_norm_list[best_aae_epe_combined_parameters[5]]} data term is {best_aae_epe_combined_parameters[0]}\n"
            f"weight for the {reg_u_norm_list[best_aae_epe_combined_parameters[6]]} regularizer of the image u is {best_aae_epe_combined_parameters[1]}\n"
            f"weight for the {motionTerm_u_norm_list[best_aae_epe_combined_parameters[8]]} {motionTerm_u_list[best_aae_epe_combined_parameters[7]]} motion term of u is {best_aae_epe_combined_parameters[2]}\n"
            f"weight for the {motionTerm_v_norm_list[best_aae_epe_combined_parameters[11]]} {motionTerm_v_list[best_aae_epe_combined_parameters[10]]} motion term for v is {best_aae_epe_combined_parameters[3]}\n"
            f"weight for the {reg_v_norm_list[best_aae_epe_combined_parameters[12]]} regularizer of the motion fields v is {best_aae_epe_combined_parameters[4]}\n"
            f"The error values are: {best_aae_epe_combined[0]:.4f} and {best_aae_epe_combined[1]:.4f}\n\n"
        )