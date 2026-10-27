import cv2
import imageio.v2 as imageio
import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

num_iterations = 100

throughVideoPath = True
video_path = "media3.avi"

name = "SSIM"

if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video '{video_path}' does not exist. Stopping execution.")



resize_value = 0.5#choose a value between 0.0 and 1.0

cut = False
cut_width = [0.45 , 0.0]
cut_height = [0.0 , 0.0]

start_sec = 44.0
end_sec =72.0
n_frames = 40

"""
cut = True
cut_width = [0.45 , 0.0]
cut_height = [0.0 , 0.0]

start_sec = 58.5
end_sec = 60.5
n_frames = 3
"""

denoiseByAveraging = True
#denoiseByAveraging = False
denoiseWithSameFrames = False

denoise_start_sec = 79.0
denoise_end_sec = 104.8
denoise_n_frames = 170

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
            frame_gray = cv2.fastNlMeansDenoising(src=frame_gray, dst=None, h=40, templateWindowSize=7,searchWindowSize=21)
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

inputDimension = frames[0].shape


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
        umax_mag = np.percentile(magnitude_u_new, 99)

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


cut = 0.1
framesToCompare = [(frames[i]-frames[0]) for i in range(len(frames))]
min_ssim = 0.0
while(min_ssim<0.7):
    m = 1.0
    for f in framesToCompare:
        f[(f<cut) & (f>-cut)] = 0.0
    for i in range(len(framesToCompare)):
        score, diff_map = ssim(framesToCompare[0], framesToCompare[i], full=True, data_range=framesToCompare[i].max() - framesToCompare[i].min())
        if(score<m):
            m = score
    min_ssim = m
    cut = cut + 1.0
print(cut)

for i in range(len(frames)):
    result = framesToCompare[i].reshape(inputDimension, order = "F")
    magnitude_u_new = result
    #magnitude_u_new = np.sqrt(result**2)
    umax_mag = np.max(np.abs(magnitude_u_new))
    #umax_mag = np.percentile(mask, 99)
    plt.figure(figsize=(7,7))

    plt.title(f"magnitude of new u_{i+1}")
    plt.imshow(magnitude_u_new, cmap="seismic", vmin=-umax_mag, vmax=umax_mag)
    plt.axis("off")

    file_path = os.path.join(folder_name_main, f"framesToCompare{i}.png")
    plt.savefig(file_path, dpi=200, bbox_inches="tight")


#framesToCompare = [(frames[i]-frames[0]) for i in range(len(frames))]
# score liegt zwischen -1.0 und 1.0 (1.0 = identisch)
ssim_values=np.zeros(len(framesToCompare))
mse_values=np.zeros(len(framesToCompare))
max_values=np.zeros(len(framesToCompare))
min_values=np.zeros(len(framesToCompare))
sum_max_min_values=np.zeros(len(framesToCompare))
for i in range(len(frames)):
    score, diff_map = ssim(framesToCompare[0], framesToCompare[i], full=True, data_range=framesToCompare[1].max() - framesToCompare[1].min())
    ssim_values[i]=score
    print(f"for frame {i} the SSIM is {score}\n")
for i in range(len(frames)):
    mse_value = mse(framesToCompare[0],framesToCompare[i])
    mse_values[i]=mse_value
    print(f"for frame {i} the MSE is {mse_value}\n")
for i in range(len(frames)):
    max_value = np.max(framesToCompare[i])
    max_values[i]=max_value
    print(f"for frame {i} the max value is {max_value}\n")
for i in range(len(frames)):
    min_value = np.min(framesToCompare[i])
    min_values[i]=min_value
    print(f"for frame {i} the min value is {min_value}\n")
for i in range(len(frames)):
    min_value = np.min(framesToCompare[i])
    max_value = np.min(framesToCompare[i])
    sum = np.abs(max_value)+np.abs(min_value)
    sum_max_min_values[i]=sum
    print(f"for frame {i} the sum of the absolute values of the min and max value is {sum}\n")

for i in range(len(frames)-1):
    score = ssim_values[i+1] - ssim_values[i]
    print(f"for frame {i} the derivative of SSIM is {score}\n")
for i in range(len(frames)-1):
    mse_value = mse_values[i+1] - mse_values[i]
    print(f"for frame {i} the derivative of MSE is {mse_value}\n")
for i in range(len(frames)-1):
    max_value = max_values[i+1] - max_values[i]
    print(f"for frame {i} the derivative of max value is {max_value}\n")
for i in range(len(frames)-1):
    min_value = min_values[i+1] - min_values[i]
    print(f"for frame {i} the derivative of min value is {min_value}\n")
for i in range(len(frames)-1):
    sum = sum_max_min_values[i+1] - sum_max_min_values[i]
    print(f"for frame {i} the derivative of sum of the absolute values of the min and max value is {sum}\n")

for i in range(len(frames)-2):
    score = ssim_values[i+2] - 2*ssim_values[i+1] + ssim_values[i]
    print(f"for frame {i} the second derivative of SSIM is {score}\n")
for i in range(len(frames)-2):
    mse_value = mse_values[i+2] - 2*mse_values[i+1] + mse_values[i]
    print(f"for frame {i} the second derivative of MSE is {mse_value}\n")
for i in range(len(frames)-2):
    max_value = max_values[i+2] - 2*max_values[i+1] + max_values[i]
    print(f"for frame {i} the second derivative of max value is {max_value}\n")
for i in range(len(frames)-2):
    min_value = min_values[i+2] - 2*min_values[i+1] + min_values[i]
    print(f"for frame {i} the second derivative of min value is {min_value}\n")
for i in range(len(frames)-2):
    sum = sum_max_min_values[i+2] - 2*sum_max_min_values[i+1] + sum_max_min_values[i]
    print(f"for frame {i} the second derivative of sum of the absolute values of the min and max value is {sum}\n")