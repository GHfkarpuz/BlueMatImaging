# video_loader.py

import cv2
import numpy as np


def load_frames_from_video(video_path, start_sec, end_sec, n_frames,
                           resize_value, cut, cut_width, cut_height):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"There are {fps} frames per second")

    start_frame = max(0, int(start_sec * fps))
    end_frame   = min(total_frames - 1, int(end_sec * fps))
    frame_indices = set(np.linspace(start_frame, end_frame, n_frames).astype(int))

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in frame_indices:
            if frame is None:
                idx += 1
                continue                          # war: pass (Bug)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = _crop(frame_gray, width, height, cut, cut_width, cut_height)
            if frame_gray.size == 0:
                idx += 1
                continue                          # war: pass (Bug)
            frame_gray = cv2.resize(frame_gray, None, fx=resize_value, fy=resize_value)
            frame_gray = cv2.fastNlMeansDenoising(
                src=frame_gray, dst=None, h=20,
                templateWindowSize=7, searchWindowSize=21
            )
            frames.append(frame_gray.astype(np.float32))
        idx += 1

    cap.release()
    return frames


def load_frames_from_folder(image_folder, resize_value, cut, cut_width, cut_height):
    import os
    png_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.png')])
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in '{image_folder}'")
    print(f"{len(png_files)} PNG files found")

    frames = []
    for file_name in png_files:
        frame_gray = cv2.imread(os.path.join(image_folder, file_name), cv2.IMREAD_GRAYSCALE)
        if frame_gray is None:
            print(f"Warning: could not read {file_name}. Skipping.")
            continue
        h, w = frame_gray.shape
        frame_gray = _crop(frame_gray, w, h, cut, cut_width, cut_height)
        frame_gray = cv2.resize(frame_gray, None, fx=resize_value, fy=resize_value)
        frames.append(frame_gray.astype(np.float32))

    print(f"Successfully loaded {len(frames)} frames.")
    return frames


def load_frames_from_synthetic(data_folder, resize_value, cut, cut_width, cut_height):
    import os
    u_noisy = np.load(os.path.join(data_folder, "u_noisy.npy"))
    frames = []
    for j in range(len(u_noisy)):
        h, w = u_noisy[j].shape
        frame = _crop(u_noisy[j], w, h, cut, cut_width, cut_height)
        if resize_value != 1.0:
            frame = cv2.resize(frame, None, fx=resize_value, fy=resize_value)
        frames.append(frame)
    return frames


def apply_denoise_by_averaging(frames, cap, fps, total_frames,
                               denoise_start_sec, denoise_end_sec,
                               resize_value, cut, cut_width, cut_height,
                               denoise_with_same_frames):
    if denoise_with_same_frames:
        avg = sum(frames) / len(frames)
        result = []
        for f in frames:
            f = f - avg
            f = cv2.normalize(f, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
            result.append(f)
        return result

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    h_f, w_f = frames[0].shape

    start_frame = max(0, int(denoise_start_sec * fps))
    end_frame   = min(total_frames - 1, int(denoise_end_sec * fps))
    target_set  = set(np.arange(start_frame, end_frame + 1))

    avg = np.zeros((h_f, w_f), dtype=np.float32)
    count = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in target_set:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = _crop(frame_gray, width, height,
                               cut, cut_width, cut_height)
            frame_gray = cv2.resize(frame_gray, None,
                                    fx=resize_value, fy=resize_value)
            avg += frame_gray.astype(np.float32)
            count += 1
        idx += 1
    avg /= float(count)

    result = []
    for f in frames:
        f = f - avg + 255.0
        f = (f * 0.5).astype(np.uint8)
        f = cv2.fastNlMeansDenoising(src=f, dst=None, h=40,
                                     templateWindowSize=5, searchWindowSize=15)
        result.append(f.astype(np.float32))
    return result


def _crop(frame, width, height, cut, cut_width, cut_height):
    if not cut:
        return frame
    x1 = int(width  * cut_width[0])
    x2 = int(width  * (1 - cut_width[1]))
    y1 = int(height * cut_height[0])
    y2 = int(height * (1 - cut_height[1]))
    return frame[y1:y2, x1:x2]