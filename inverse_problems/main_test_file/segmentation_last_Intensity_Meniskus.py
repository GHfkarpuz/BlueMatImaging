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
from skimage.restoration import denoise_nl_means

num_iterations = 30

throughVideoPath = True
video_path = "media3.avi"
image_folder = "/home/karpuzfa/CodeFatih/BlueMatImaging/inverse_problems/main_test_file/DSG43_a1_V10_8500t8800_drying/Movement"
background_folder = "/home/karpuzfa/CodeFatih/BlueMatImaging/inverse_problems/main_test_file/DSG43_a1_V10_8500t8800_drying/ForBackground"
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
name = "SegmentationPlayNNNOlfNewSequenceHighResolutionMedia3FNewSegmentsSlices"

if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video '{video_path}' does not exist. Stopping execution.")



resize_value = 0.5 #choose a value between 0.0 and 1.0


cut = True
cut_width = [0.5 , 0.0]
cut_height = [0.0 , 0.0]

start_sec = 58.0
end_sec = 60.0
n_frames = 3
"""

cut = True
cut_width = [0.0 , 0.0]
cut_height = [0.5 , 0.0]

start_sec = 72.5
end_sec = 73.5
n_frames = 2
"""
denoiseByAveraging = False
#denoiseByAveraging = False
denoiseWithSameFrames = False


denoise_start_sec = 80.0
denoise_end_sec = 100.0
denoise_n_frames = 170
denoise_postprocessing_frames = 20
"""

denoise_start_sec = 103.4
denoise_end_sec = 105.0
denoise_n_frames = 170
denoise_postprocessing_frames = 20
"""

figures_frequence = 10
errors_frequence = 1

# Parameter
weight_1_all = [1.0]    # Datenterm u
weight_2_all = [2.0,1.0,0.1]    # Regularizer of u
weight_3_all = [1.0]    # motion term for u
weight_4_all = [1.0]    # motion term for v
weight_5_all = [1.0,0.4,0.1]   # Regularizer of v


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
motionTerm_u_norm_all = [1]
motionTerm_u_upwind_all = [1]
motionTerm_v_all = [1]
motionTerm_v_norm_all = [1]
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

    framesForDenoising_indices = np.linspace(int(denoise_start_sec*fps), int(denoise_end_sec*fps), denoise_postprocessing_frames).astype(int)
    framesForDenoising = []

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
            frame_gray = cv2.fastNlMeansDenoising(src=frame_gray, dst=None, h=25, templateWindowSize=7,searchWindowSize=21)
            frame_gray = frame_gray.astype(np.float32)
            frames.append(frame_gray)

        idx += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idxDenoise = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idxDenoise in framesForDenoising_indices:
            if frame is None:
                idxDenoise += 1
                pass

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if cut:
                x1 = int(width * cut_width[0])
                x2 = int(width * (1 - cut_width[1]))
                y1 = int(height * cut_height[0])
                y2 = int(height * (1 - cut_height[1]))
                frame_gray = frame_gray[y1:y2, x1:x2]

            if frame_gray.size == 0:
                idxDenoise += 1
                pass
            frame_gray = cv2.resize(frame_gray, None, fx=resize_value, fy=resize_value)
            #frame_gray = cv2.fastNlMeansDenoisingMulti(frame_gray, imgToDenoiseIndex=2, temporalWindowSize=5, h=7, templateWindowSize=7, searchWindowSize=21)
            frame_gray = cv2.fastNlMeansDenoising(src=frame_gray, dst=None, h=15, templateWindowSize=7,searchWindowSize=21)
            frame_gray = frame_gray.astype(np.float32)
            framesForDenoising.append(frame_gray)

        idxDenoise += 1
    #frames = cv2.fastNlMeansDenoisingMulti(frames, imgToDenoiseIndex=2, temporalWindowSize=5, h=7, templateWindowSize=7, searchWindowSize=21)
    #for f in frames:
    #    f=f.astype(np.float32)

else:
    png_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.png')])
    tif_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.tif')])

    # check if frames are found
    if ((len(png_files)==0) and (len(tif_files) == 0)):
        raise FileNotFoundError(f"no png or tif files found in '{image_folder}'")

    if len(png_files)!=0:
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
    
    if len(tif_files)!=0:
        print(f"{len(tif_files)} tif files were found")

        frames = []

        for file_name in tif_files:
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


num_slices = 6
# save the frames that we start with
for j in range(num_slices):
    for i in range(len(frames)):
        result_1_u = frames[i].reshape(inputDimension, order="F").copy()
        lower_cut = int((255.0/num_slices)*(j))
        upper_cut = int((255.0/num_slices)*(j+1))
        result_1_u[(result_1_u>upper_cut)|(result_1_u<lower_cut)]=0.0
        magnitude_u = np.sqrt(result_1_u**2)

        umax = np.max(np.abs(result_1_u))
        umax_mag = np.percentile(magnitude_u, 99)

        plt.figure(figsize=(7,7))

        plt.title(f"magnitude of u_{i+1}")
        plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax)
        plt.axis("off")

        file_path = os.path.join(folder_name_main, f"{name},{i},cuts{lower_cut},{upper_cut}.png")
        plt.savefig(file_path, dpi=200, bbox_inches="tight")

        plt.close()


if(denoiseByAveraging):
    if denoiseWithSameFrames:
        denoisedImagebyAveraging = np.zeros(inputDimension, dtype=np.float32)
        for f in frames:
            denoisedImagebyAveraging += f
        denoisedImagebyAveraging=denoisedImagebyAveraging/(float(len(frames)))
        for i in range(len(frames)):
            frames[i]-=denoisedImagebyAveraging
            frames[i]+=np.ones(frames[i].shape)*255.0
            frames[i]*=0.5
            #frames[i].reshape(inputDimension, dtype=np.float32)
            #frames[i]=cv2.normalize(frames[i], None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            #frames[i] = cv2.fastNlMeansDenoising(frames[i], None, 20, 7, 15)
    else:
        if(throughVideoPath):
            denoisedImagebyAveraging = np.zeros(inputDimension, dtype=np.float32)

            denoise_start_frame = int(denoise_start_sec * fps)
            denoise_end_frame = int(denoise_end_sec * fps)
            denoise_start_frame = max(0, denoise_start_frame)
            denoise_end_frame = min(total_frames - 1, denoise_end_frame)

            denoise_frame_indices = np.linspace(int(denoise_start_frame), int(denoise_end_frame), denoise_n_frames).astype(int)
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
        else:
            tif_files = sorted([f for f in os.listdir(background_folder) if f.lower().endswith('.tif')])
            denoisedImagebyAveraging = np.zeros(inputDimension, dtype=np.float32)
                    
            for file_name in tif_files:
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
                denoisedImagebyAveraging += frame_gray
            
            denoisedImagebyAveraging = denoisedImagebyAveraging/float(len(tif_files))

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
denoisedImagebyAveraging = np.zeros(inputDimension, dtype=np.float32)
for f in frames:
    denoisedImagebyAveraging += f
denoisedImagebyAveraging=denoisedImagebyAveraging/(float(len(frames)))
for i in range(len(frames)):
    frames[i]-=denoisedImagebyAveraging
    frames[i]+=np.ones(frames[i].shape)*255.0
    frames[i]*=0.5

num_slices = 25
# save the frames that we start with
for j in range(num_slices):
    for i in range(len(frames)):
        result_1_u = frames[i].reshape(inputDimension, order="F").copy()
        lower_cut = int(100 + (50.0/num_slices)*(j))
        upper_cut = int(100 + (50.0/num_slices)*(j+1))
        result_1_u[(result_1_u>upper_cut)|(result_1_u<lower_cut)]=0.0
        magnitude_u = np.sqrt(result_1_u**2)

        umax = np.max(np.abs(result_1_u))
        umax_mag = np.percentile(magnitude_u, 99)

        plt.figure(figsize=(7,7))

        plt.title(f"magnitude of u_{i+1}")
        plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax)
        plt.axis("off")

        file_path = os.path.join(folder_name_main, f"{name},{i},cuts{lower_cut},{upper_cut}.png")
        plt.savefig(file_path, dpi=200, bbox_inches="tight")

        plt.close()
breakpoint()

"""
mask = np.abs((frames[1].astype(np.float32)-frames[0].astype(np.float32))**2)
mask = (mask/np.max(mask))*255.0
#mask = mask.reshape(inputDimension, order="F")
diff = 5.0
mn = np.mean(mask)
plt.figure(figsize=(7,7))
for j in range(inputDimension[0]):
    for k in range(inputDimension[1]):
        if(mask[j][k]<(mn+diff) and mask[j][k]>(mn-diff)):
            mask[j][k]=0.0
# save the denoised frames that we start with
#umax1 = np.max(np.abs(mask))
umax_mag = np.percentile(mask, 99)

plt.title(f"magnitude of new u")
plt.imshow(mask, cmap="gray", vmin=0, vmax=umax_mag)
plt.axis("off")

file_path = os.path.join(folder_name_main, f"subtractedAverageFramesNew{name}.png")
plt.savefig(file_path, dpi=200, bbox_inches="tight")

plt.close()


diffFrame = np.abs((frames[1]-frames[0])**4)
diffFrame = (diffFrame/np.max(diffFrame))*255.0
diffFrame = diffFrame.reshape(inputDimension, order="F")
magnitude_u_new = np.sqrt(diffFrame**2)
umax_mag = np.max(magnitude_u_new)
plt.figure(figsize=(7,7))

plt.title(f"magnitude of new u_{i+1}")
plt.imshow(magnitude_u_new, cmap="gray", vmin=0, vmax=umax_mag)
plt.axis("off")

file_path = os.path.join(folder_name_main, f"differenceFramesNew{name}.png")
plt.savefig(file_path, dpi=200, bbox_inches="tight")


# Vektorisiere die Operation (kein [j][k] Zugriff nötig!):
for frame in framesForDenoising:
    mask[frame > 145] = 0.0

magnitude_u_new = np.sqrt(mask**2)
#umax_mag = np.max(magnitude_u_new)
umax_mag = np.percentile(mask, 99)
plt.figure(figsize=(7,7))

plt.title(f"magnitude of new u_{i+1}")
plt.imshow(magnitude_u_new, cmap="gray", vmin=0, vmax=umax_mag)
plt.axis("off")

file_path = os.path.join(folder_name_main, f"differenceFramesPostDenoised{name}.png")
plt.savefig(file_path, dpi=200, bbox_inches="tight")


mask[mask>0.0]=255.0
np.save("beforeConv.npy", mask)

num_conv_it=20
for n in range(num_conv_it):
    conv_width = 10
    percent_px = 0.14
    mask_copy=mask.copy()
    for i in range(conv_width,inputDimension[0]-conv_width):
        for j in range(conv_width,inputDimension[1]-conv_width):
            if(mask[i][j]!=0.0):
                sum = 0.0
                for k in range(-conv_width,conv_width,1):
                    for l in range(-conv_width,conv_width,1):
                        sum+=mask_copy[i+k][j+l]
                if(sum<(((2*conv_width)**2)*255.0*percent_px)):
                    mask[i][j]=0.0
    
    #mask_uint8 = mask.astype(np.uint8)
    if(num_conv_it%2==0):
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        mask_dilate= mask
        for i in range(1):
            mask_eroded = cv2.erode(mask_dilate, kernel_erode, iterations=1)
            mask_dilate = cv2.dilate(mask_eroded, kernel_dilate, iterations=1)
        #mask_eroded = cv2.erode(mask, kernel_erode, iterations=20)
        #kernel_dilate_post = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
        #mask_dilate = cv2.dilate(mask_dilate, kernel_erode, iterations=4)
        mask_out=mask_dilate
    
    magnitude_u_new = np.sqrt(mask**2)
    
    umax_mag = np.max(magnitude_u_new)
    #umax_mag = np.percentile(mask, 99)
    plt.figure(figsize=(7,7))

    plt.title(f"magnitude of new u")
    plt.imshow(magnitude_u_new, cmap="gray", vmin=0, vmax=umax_mag)
    plt.axis("off")

    file_path = os.path.join(folder_name_main, f"differenceFramesPostDenoisedConv{name},{n}.png")
    plt.savefig(file_path, dpi=200, bbox_inches="tight")

np.save("mask.npy", mask)
mask_uint8 = mask.astype(np.uint8)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
mask_eroded = cv2.erode(mask, kernel, iterations=2)
mask_out = cv2.dilate(mask_eroded, kernel, iterations=2)

mask = mask_out.astype(np.float32)

magnitude_u_new = np.sqrt(mask**2)
umax_mag = np.max(magnitude_u_new)
#umax_mag = np.percentile(mask, 99)
plt.figure(figsize=(7,7))

plt.title(f"magnitude of new u_{i+1}")
plt.imshow(magnitude_u_new, cmap="gray", vmin=0, vmax=umax_mag)
plt.axis("off")

file_path = os.path.join(folder_name_main, f"dilationErosion{name}.png")
plt.savefig(file_path, dpi=200, bbox_inches="tight")
"""
cap.release()


# Initialisierung der Primalen als C-Contiguous flache Arrays (Inhalt wird im Fortran-Stil verwaltet)

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

newFrames = [f[1159:1268+1,323:661+1] for f in frames]
framesBig = np.concatenate([f.flatten(order="F") for f in newFrames])

reconstructed_image = np.concatenate([newFrames[i].flatten(order="F") for i in range(len(newFrames))]) 

inputDimension = newFrames[0].shape

motionField_0_list = [np.zeros(inputDimension[0]*inputDimension[1]) for i in range(len(newFrames))]
motionField_1_list = [np.zeros(inputDimension[0]*inputDimension[1]) for i in range(len(newFrames))]

motionField_0 = np.concatenate(motionField_0_list)
motionField_1 = np.concatenate(motionField_1_list)

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
inputDimension = [newFrames[0].shape[0], newFrames[0].shape[1]]
dimensionsWithTime = [inputDimension[0], inputDimension[1], len(newFrames)]
print(inputDimension)

nPx=inputDimension[0]*inputDimension[1]

output_dir = "segmentation_last"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

inputDimension = frames[0].shape
"""
for i in range(len(frames)):
    frame_uint8= frames[i].astype(np.uint8)
    denoised = cv2.fastNlMeansDenoising(src=frame_uint8, dst=None, h=15, templateWindowSize=7,searchWindowSize=21)
    frames[i] = denoised.astype(np.float32)

for i in range(len(framesForDenoising)):
    frame_uint8= framesForDenoising[i].astype(np.uint8)
    denoised = cv2.fastNlMeansDenoising(src=frame_uint8, dst=None, h=15, templateWindowSize=7,searchWindowSize=21)
    framesForDenoising[i] = denoised.astype(np.float32)
"""
frame_ =framesSubtractedBackground
for j in range(len(framesSubtractedBackground)):
    for i in range(len(framesForDenoising)):
        vari = 20.0
        frame_[j][np.abs(framesForDenoising[i]-frames[0])<vari] = 0.0
        frame_[j][frame_[j]<140] = 0.0
        frame_[j][framesForDenoising[i]>180.0]=0.0

        if(i==len(framesForDenoising)-1):
            magnitude_u_new = np.sqrt(frame_[j]**2)
            
            umax_mag = np.max(magnitude_u_new)
            #umax_mag = np.percentile(mask, 99)
            plt.figure(figsize=(7,7))

            plt.title(f"magnitude of new u")
            plt.imshow(magnitude_u_new, cmap="gray", vmin=0, vmax=umax_mag)
            plt.axis("off")

            file_path = os.path.join(output_dir, f"{j}subtractBackgroundFrame{i}.png")
            plt.savefig(file_path, dpi=200, bbox_inches="tight")

            plt.close()


num_conv_it=1
mask= frame_
mask_copy = mask.copy()
frame_[frame_>0.0]=255.0
for m in range(len(mask)):
    for n in range(num_conv_it):
        conv_width = 5
        percent_px = 0.2
        for i in range(conv_width,inputDimension[0]-conv_width):
            for j in range(conv_width,inputDimension[1]-conv_width):
                if(mask[m][i][j]!=0.0):
                    sum = 0.0
                    for k in range(-conv_width,conv_width,1):
                        for l in range(-conv_width,conv_width,1):
                            sum+=mask_copy[m][i+k][j+l]
                    if(sum<(((2*conv_width)**2)*255.0*percent_px)):
                        mask[m][i][j]=0.0
        
        #mask_uint8 = mask.astype(np.uint8)
        if(num_conv_it%1==0):
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
            mask_eroded= mask[m]
            for i in range(1):
                mask_dilate = cv2.dilate(mask_eroded, kernel_dilate, iterations=1)
                mask_eroded = cv2.erode(mask_dilate, kernel_erode, iterations=1)
            #mask_eroded = cv2.erode(mask, kernel_erode, iterations=20)
            #kernel_dilate_post = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
            #mask_dilate = cv2.dilate(mask_dilate, kernel_erode, iterations=4)
            mask[m]=mask_eroded
        
        magnitude_u_new = np.sqrt(mask[m]**2)
        
        umax_mag = np.max(magnitude_u_new)
        #umax_mag = np.percentile(mask, 99)
        plt.figure(figsize=(7,7))

        plt.title(f"magnitude of new u")
        plt.imshow(magnitude_u_new, cmap="gray", vmin=0, vmax=umax_mag)
        plt.axis("off")

        file_path = os.path.join(output_dir, f"{m}differenceFramesPostDenoisedConvWidth{conv_width}Perc{percent_px},{n}.png")
        plt.savefig(file_path, dpi=200, bbox_inches="tight")

        file_path = os.path.join(output_dir, f"{m}dilationErosion.png")
        plt.savefig(file_path, dpi=200, bbox_inches="tight")
        plt.close()


for m in range(len(frame_)):
    for i in range(conv_width):
        for j in range(inputDimension[1]):
            mask[m][i][j]=0.0
            mask[m][inputDimension[0]-1-i][j]=0.0
for m in range(len(frame_)):
    for j in range(conv_width):
        for i in range(inputDimension[0]):
            mask[m][i][j]=0.0
            mask[m][i][inputDimension[1]-1-j]=0.0

diff = 0.5*(mask[0]+mask[len(framesSubtractedBackground)-1])

magnitude_u_new = np.sqrt(diff**2)
umax_mag = np.max(magnitude_u_new)
#umax_mag = np.percentile(mask, 99)
plt.figure(figsize=(7,7))

plt.title(f"magnitude of new u")
plt.imshow(magnitude_u_new, cmap="gray", vmin=0, vmax=umax_mag)
plt.axis("off")

file_path = os.path.join(output_dir, f"DiffDifferenceFramesPostDenoisedConvWidth{conv_width}Perc{percent_px},{n}.png")
plt.savefig(file_path, dpi=200, bbox_inches="tight")

file_path = os.path.join(output_dir, f"{m}DifffinalConvWidth{conv_width}Perc{percent_px}.png")
plt.savefig(file_path, dpi=200, bbox_inches="tight")
plt.close()

kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
mask[m] = cv2.dilate(mask[m], kernel_dilate, iterations=3)


magnitude_u_new = np.sqrt(mask[m]**2)
umax_mag = np.max(magnitude_u_new)
#umax_mag = np.percentile(mask, 99)
plt.figure(figsize=(7,7))

plt.title(f"magnitude of new u")
plt.imshow(magnitude_u_new, cmap="gray", vmin=0, vmax=umax_mag)
plt.axis("off")

file_path = os.path.join(output_dir, f"DiffDifferenceFramesPostDenoisedConvWidth{conv_width}Perc{percent_px},{n}.png")
plt.savefig(file_path, dpi=200, bbox_inches="tight")

file_path = os.path.join(output_dir, f"DifffinalConvWidthPlusDilation{conv_width}Perc{percent_px}.png")
plt.savefig(file_path, dpi=200, bbox_inches="tight")
plt.close()


mask_uint8 = mask[m].astype(np.uint8)

# find contours
contours, _ = cv2.findContours(
    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# list for the rectangles of the segments
bounding_boxes = []

for cnt in contours:
    # x1, y1 = obere linke Ecke | w = Breite | h = Höhe
    x1, y1, w, h = cv2.boundingRect(cnt)

    # Untere rechte Ecke berechnen
    x2 = x1 + w
    y2 = y1 + h

    bounding_boxes.append((x1, x2, y1, y2))
    print(
        f"für frame Summe Segment gefunden -> Oben-Links: ({x1}, {y1}), Unten-Rechts: ({x2}, {y2})"
    )

for m in range(len(frame_)):
    magnitude_u_new = np.sqrt(mask[m]**2)
    umax_mag = np.max(magnitude_u_new)
    #umax_mag = np.percentile(mask, 99)
    plt.figure(figsize=(7,7))

    plt.title(f"magnitude of new u")
    plt.imshow(magnitude_u_new, cmap="gray", vmin=0, vmax=umax_mag)
    plt.axis("off")

    file_path = os.path.join(output_dir, f"{m}differenceFramesPostDenoisedConvWidth{conv_width}Perc{percent_px},{n}.png")
    plt.savefig(file_path, dpi=200, bbox_inches="tight")

    file_path = os.path.join(output_dir, f"{m}finalConvWidth{conv_width}Perc{percent_px}.png")
    plt.savefig(file_path, dpi=200, bbox_inches="tight")
    plt.close()

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask[m] = cv2.dilate(mask[m], kernel_dilate, iterations=2)


    magnitude_u_new = np.sqrt(mask[m]**2)
    umax_mag = np.max(magnitude_u_new)
    #umax_mag = np.percentile(mask, 99)
    plt.figure(figsize=(7,7))

    plt.title(f"magnitude of new u")
    plt.imshow(magnitude_u_new, cmap="gray", vmin=0, vmax=umax_mag)
    plt.axis("off")

    file_path = os.path.join(output_dir, f"{m}differenceFramesPostDenoisedConvWidth{conv_width}Perc{percent_px},{n}.png")
    plt.savefig(file_path, dpi=200, bbox_inches="tight")

    file_path = os.path.join(output_dir, f"{m}finalConvWidthPlusDilation{conv_width}Perc{percent_px}.png")
    plt.savefig(file_path, dpi=200, bbox_inches="tight")
    plt.close()


    mask_uint8 = mask[m].astype(np.uint8)

    # find contours
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # list for the rectangles of the segments
    bounding_boxes = []

    for cnt in contours:
        # x1, y1 = obere linke Ecke | w = Breite | h = Höhe
        x1, y1, w, h = cv2.boundingRect(cnt)

        # Untere rechte Ecke berechnen
        x2 = x1 + w
        y2 = y1 + h

        bounding_boxes.append((x1, x2, y1, y2))
        print(
            f"für frame {m} Segment gefunden -> Oben-Links: ({x1}, {y1}), Unten-Rechts: ({x2}, {y2})"
        )