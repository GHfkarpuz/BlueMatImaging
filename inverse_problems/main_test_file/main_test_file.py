import sys
sys.path.append("/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/wrappers/python")

import cv2
import numpy as np
import gc
import matplotlib.pyplot as plt
import os
from wrapper import FlexBoxSolver # python wrapper

num_iterations = 20

throughVideoPath = True
video_path = "city.mp4"
image_folder = "Urban2_twoFrames"

groundFlowExists = False
if groundFlowExists:
    flo_path="Urban2"
name = "City"

if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video '{video_path}' does not exist. Stopping execution.")



resize_value = 0.2#choose a value between 0.0 and 1.0

cut = False
cut_width = [0.3 , 0.0]
cut_height = [0.0 , 0.0]

start_sec = 1.0
end_sec = 1.2
n_frames = 4

#denoiseByAveraging = True
denoiseByAveraging = False
denoiseWithSameFrames = False

denoise_start_sec = 45.0
denoise_end_sec = 110.0
denoise_n_frames = n_frames

visualizeInEachStep = False

figures_frequence = 1


# Parameter
weight_1 = 1.0    # Datenterm u
weight_2 = 0.0    # Regularizer of u
weight_3 = 1.0    # motion term for u
weight_4 = 1.0    # motion term for v
weight_5 = 0.05    # Regularizer of v

tol = 1e-4 #for outer iterations

#parameters for first solver
maxIterations_first_solver = 300
tol_first_solver = 1e-6
verbose_first_solver = 1

#parameters for second solver
maxIterations_second_solver = 300
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
            
# create thje folder in which we will store the figures
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

file_path = os.path.join(folder_name, "parameters.txt")

# write the parameters in the textfile
with open(file_path, "w", encoding="utf-8") as f:
    f.write(f"the parameters are\nweight_1: {weight_1}\nweight_2: {weight_2}\nweight_3: {weight_3}\nweight_4: {weight_4}\nweight_5: {weight_5}")

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
            frame_gray = frame_gray.astype(np.float32) / 255.0
            #frame_gray -= denoisedImage
            #frame_gray = cv2.normalize(frame_gray, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            frames.append(frame_gray)

        idx += 1
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
        frame_gray = frame_gray.astype(np.float32) / 255.0
        
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

    file_path = os.path.join(folder_name, f"{name},{i}.png")
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
                frame_gray = frame_gray.astype(np.float32) / 255.0
                #frame_gray -= denoisedImage
                #frame_gray = cv2.normalize(frame_gray, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                denoisedImagebyAveraging += frame_gray

            idx += 1

        denoisedImagebyAveraging = denoisedImagebyAveraging/float(len(denoise_frame_indices))

        for i in range(len(frames)):
            frames[i]-=denoisedImagebyAveraging
            #frames[i]+=np.ones(nPx).reshape(inputDimension, order="F")
            frames[i]=cv2.normalize(frames[i], None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


    # save the denoised frames that we start with
    for i in range(len(frames)):
        result_1_u = frames[i].reshape(inputDimension, order="F")
        magnitude_u = np.sqrt(result_1_u**2)

        umax1 = np.max(np.abs(result_1_u))
        umax_mag = np.percentile(magnitude_u, 99)

        plt.figure(figsize=(7,7))

        plt.title(f"magnitude of denoised u_{i+1}")
        plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax_mag)
        plt.axis("off")

        file_path = os.path.join(folder_name, f"denoised{name},{i}.png")
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
"""
motionField_0 = np.concatenate(motionField_0_list)
motionField_1 = np.concatenate(motionField_1_list)

# save the frames that we start with
for i in range(len(frames)):
    result_1_u = frames[i].reshape(inputDimension, order="F")
    magnitude_u = np.sqrt(result_1_u**2)

    umax1 = np.max(np.abs(result_1_u))
    umax_mag = np.percentile(magnitude_u, 99)

    result_1_v = motionField_0_list[i].reshape(inputDimension, order="F")
    result_2_v = motionField_1_list[i].reshape(inputDimension, order="F")

    magnitude_v = np.sqrt(result_1_v**2 + result_2_v**2)
    vmax1 = np.max(np.abs(result_1_v))
    vmax2 = np.max(np.abs(result_2_v))
    vmax_mag = np.percentile(magnitude_v, 99)
    
    print(f"vmax1 for i = {i} is ", vmax1)
    print(f"vmax2 for i = {i} is ", vmax2)

    plt.figure(figsize=(7,7))

    plt.subplot(2,2,1)
    plt.title("magnitude of u")
    plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax1)
    #plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax_mag)
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

    file_path = os.path.join(folder_name, f"{name},{i}.png")
    plt.savefig(file_path, dpi=200, bbox_inches="tight")

    plt.close()


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
                "inputDimension": dimensionsWithTime,
                "isMinus": False
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
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[concatUpwindGradientTimeDerivativeOp]
                )
            if(motionTerm_u_norm==2):
                solver.add_dual(
                    prox_type="L1AnisoProxDual",
                    alpha=weight_3 / timeIntervall,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[concatUpwindGradientTimeDerivativeOp]
                )
            if(motionTerm_u_norm==3):
                solver.add_dual(
                    prox_type="L2proxDual",
                    alpha=weight_3 / timeIntervall,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
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
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[concatUpwindDivergenceTimeDerivativeOp]
                )
            if(motionTerm_u_norm==2):
                solver.add_dual(
                    prox_type="L1AnisoProxDual",
                    alpha=weight_3 / timeIntervall,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[concatUpwindDivergenceTimeDerivativeOp]
                )
            if(motionTerm_u_norm==3):
                solver.add_dual(
                    prox_type="L2proxDual",
                    alpha=weight_3 / timeIntervall,
                    f_list=[(np.zeros(nPx * len(frames), dtype=np.float32))],
                    corresponding_primals=[corresponding_primals[0]],
                    operator_dict=[concatUpwindDivergenceTimeDerivativeOp]
                )
    
    print("before first solver")
    u_out, _ = solver.solve()
    print("after first solver")
    
    #reorder
    reconstructed_image = np.array(u_out[0], dtype=np.float32)
    reconstructed_image_frames = np.split(reconstructed_image, len(frames))

    # save the frames after first solver
    if(visualizeInEachStep):
        for j in range(len(frames)):
            umax1 = np.max(np.abs(result_1_u))
            umax_mag = np.percentile(magnitude_u, 99)

            result_1_uF = reconstructed_image_frames[j].reshape(inputDimension, order="F")
            magnitude_uF = np.sqrt(result_1_uF**2)

            umax1F = np.max(np.abs(result_1_uF))
            umax_magF = np.percentile(magnitude_uF, 99)

            plt.figure(figsize=(7,7))

            plt.title("magnitude of u with order F")
            plt.imshow(magnitude_uF, cmap="gray", vmin=0, vmax=umax_magF)
            plt.axis("off")

            file_path = os.path.join(folder_name, f"AfterFirstSolver{name},{i+1}{j+1}.png")
            plt.savefig(file_path, dpi=200, bbox_inches="tight")

            plt.close()
    
    del solver
    
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
    
    if i > 2 and diffV0<tol and diffV1<tol and diffU<tol:
        print(f"converged for outer iterations in the {i}-th iteration.")
        break
    
    # visualization
    for j in range(len(frames)-1):
        if ((i+1) % figures_frequence == 0):
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
            plt.imshow(result_2_v, cmap="seismic", vmin=-vmax1, vmax=vmax2)
            plt.axis("off")

            plt.subplot(2,2,4)
            plt.title("v_y")
            plt.imshow(result_1_v, cmap="seismic", vmin=-vmax2, vmax=vmax1)
            plt.axis("off")

            plt.subplot(2,2,2)
            plt.title("magnitude of v")
            plt.imshow(magnitude_v, cmap="gray", vmin=0, vmax=vmax_mag)
            plt.axis("off")
            
            #plt.savefig(f"result{dataTerm_norm_list[dataTerm_norm]}{reg_u_norm_list[reg_u_norm]}{motionTerm_u_norm_list[motionTerm_u_norm]}{motionTerm_u_list[motionTerm_u]}{motionTerm_v_norm_list[motionTerm_v_norm]}{motionTerm_v_list[motionTerm_v]}{reg_v_norm_list[reg_v_norm]}{i+1},{j+1}.png", dpi=200, bbox_inches="tight")
            file_path = os.path.join(folder_name, f"{i+1},{j+1}.png")
            plt.savefig(file_path, dpi=200, bbox_inches="tight")
            plt.close()

    
    if groundFlowExists:
        flo_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.flo')]
        flo_path = os.path.join(image_folder, flo_files[0])
        # 3. Exakt dasselbe Cropping (Zuschneiden) wie bei den Bildern anwenden
        gt_vx, gt_vy = read_flo_file(flo_path)
        if cut:
            h_orig, w_orig = gt_vx.shape
            x1 = int(w_orig * cut_width[0])
            x2 = int(w_orig * (1 - cut_width[1]))
            y1 = int(h_orig * cut_height[0])
            y2 = int(h_orig * (1 - cut_height[1]))
            gt_vx = gt_vx[y1:y2, x1:x2]
            gt_vy = gt_vy[y1:y2, x1:x2]
            
        #resize
        gt_vx = cv2.resize(gt_vx, None, fx=resize_value, fy=resize_value) * resize_value
        gt_vy = cv2.resize(gt_vy, None, fx=resize_value, fy=resize_value) * resize_value
        

        invalid_mask = (np.abs(gt_vx) > 1e9) | (np.abs(gt_vy) > 1e9)
        valid_mask = ~invalid_mask
        
        calculated_vx = motionField_0[:nPx].reshape(gt_vx.shape)
        calculated_vy = motionField_1[:nPx].reshape(gt_vy.shape)
        
        diff_x = calculated_vx[valid_mask] - gt_vx[valid_mask]
        diff_y = calculated_vy[valid_mask] - gt_vy[valid_mask]
        
        # Endpoint Error (EPE)
        epe_per_pixel = cv2.magnitude(diff_x, diff_y)
        mean_epe = np.mean(epe_per_pixel)
        
        # Angular Error (AAE)
        calc_vx_v = calculated_vx[valid_mask]
        calc_vy_v = calculated_vy[valid_mask]
        gt_vx_v = gt_vx[valid_mask]
        gt_vy_v = gt_vy[valid_mask]
        
        mag_calc = cv2.magnitude(calc_vx_v, calc_vy_v)
        mag_gt   = cv2.magnitude(gt_vx_v, gt_vy_v)
        
        numerator = calc_vx_v * gt_vx_v + calc_vy_v * gt_vy_v + 1.0
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
        
        # Ergebnisse an die parameters.txt anhängen
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"\n\nEvaluation (.flo):\nSource File: {flo_files[0]}\nEPE: {mean_epe:.4f}\nAEE (Deg): {mean_aee_deg:.4f}")
    
    gc.collect()