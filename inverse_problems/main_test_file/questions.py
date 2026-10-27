import numpy as np

import sys
sys.path.append("/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/wrappers/python")

import cv2
import numpy as np
import gc
import matplotlib.pyplot as plt
import os
from wrapper import FlexBoxSolver # python wrapper
import errorFunctions as ef
from itertools import product

num_iterations = 50

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
name = "summedFrames"

if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video '{video_path}' does not exist. Stopping execution.")



resize_value = 0.35#choose a value between 0.0 and 1.0

cut = True
cut_width = [0.45 , 0.0]
cut_height = [0.0 , 0.0]

start_sec = 63.5
end_sec = 65.5
n_frames = 4


denoiseByAveraging = True
#denoiseByAveraging = False
denoiseWithSameFrames = False

denoise_start_sec = 79.0
denoise_end_sec = 104.8
denoise_n_frames = 170

visualizeInEachStep = False

figures_frequence = 10
errors_frequence = 1

# Parameter
weight_1_all = [1.0]    # Datenterm u
weight_2_all = [0.7,0.3,0.1,0.01]    # Regularizer of u
weight_3_all = [1.0]    # motion term for u
weight_4_all = [1.0]    # motion term for v
weight_5_all = [20.0,10.0]   # Regularizer of v


tol = 1e-4 #for outer iterations

#parameters for first solver
maxIterations_first_solver = 10
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
motionTerm_u_all = [1]
motionTerm_u_norm_all = [3]
motionTerm_u_upwind_all = [1]
motionTerm_v_all = [1]
motionTerm_v_norm_all = [2]
reg_v_norm_all = [1]


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
            frame_gray = cv2.fastNlMeansDenoising(src=frame_gray, dst=None, h=10, templateWindowSize=7,searchWindowSize=21)
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

    summedFrames = np.zeros(frames[0].size)
    for i in range(len(frames)):
        summedFrames += frames[i]
    summedFrames=summedFrames/len(frames)


    result_summedFrames = summedFrames.reshape(inputDimension, order="F")
    magnitude_summed_frames = np.sqrt(result_summedFrames**2)

    umax = np.max(np.abs(result_summedFrames))
    umax_mag = np.percentile(magnitude_summed_frames, 99)

    plt.figure(figsize=(7,7))

    plt.title(f"summed frames")
    plt.imshow(magnitude_summed_frames, cmap="gray", vmin=0, vmax=umax_mag)
    plt.axis("off")

    file_path = os.path.join(folder_name_main, f"summedFrames{name},{i}.png")
    plt.savefig(file_path, dpi=200, bbox_inches="tight")

    plt.close()

cap.release()