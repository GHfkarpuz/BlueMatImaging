# visualization.py

import os
import numpy as np
import matplotlib.pyplot as plt


def save_initial_frames(frames, input_dimension, folder_name, name):
    for i, frame in enumerate(frames):
        img = frame.reshape(input_dimension, order="F")
        mag = np.sqrt(img ** 2)
        vmax = np.percentile(mag, 99)

        plt.figure(figsize=(7, 7))
        plt.title(f"magnitude of u_{i + 1}")
        plt.imshow(mag, cmap="gray", vmin=0, vmax=vmax)
        plt.axis("off")

        path = os.path.join(folder_name, f"{name},{i}.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()


def save_iteration_figure(u_frame, mf0_frame, mf1_frame, pf_frame,
                           input_dimension, folder_path, iteration, frame_idx):
    u  = u_frame.reshape(input_dimension, order="F")
    v0 = mf0_frame.reshape(input_dimension, order="F")
    v1 = mf1_frame.reshape(input_dimension, order="F")
    pf = pf_frame.reshape(input_dimension, order="F")

    mag_v  = np.sqrt(v0 ** 2 + v1 ** 2)
    mag_u  = np.sqrt(u ** 2)
    vmax0  = np.max(np.abs(v0))
    vmax1  = np.max(np.abs(v1))
    vmax_m = np.percentile(mag_v, 99)
    umax   = np.max(np.abs(u))
    pf_min = np.min(pf)
    pf_max = np.max(pf)

    fig, axes = plt.subplots(3, 2, figsize=(7, 7))
    axes[0, 0].set_title("magnitude of u")
    axes[0, 0].imshow(mag_u, cmap="gray", vmin=0, vmax=umax)
    axes[0, 0].axis("off")

    axes[1, 0].set_title("phase field (seismic)")
    axes[1, 0].imshow(pf, cmap="seismic", vmin=pf_min, vmax=pf_max)
    axes[1, 0].axis("off")

    axes[1, 1].set_title("phase field (gray)")
    axes[1, 1].imshow(pf, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1, 1].axis("off")

    axes[2, 0].set_title("v_x")
    axes[2, 0].imshow(v1, cmap="seismic", vmin=-vmax1, vmax=vmax1)
    axes[2, 0].axis("off")

    axes[2, 1].set_title("v_y")
    axes[2, 1].imshow(v0, cmap="seismic", vmin=-vmax0, vmax=vmax0)
    axes[2, 1].axis("off")

    axes[0, 1].set_title("magnitude of v")
    axes[0, 1].imshow(mag_v, cmap="gray", vmin=0, vmax=vmax_m)
    axes[0, 1].axis("off")

    path = os.path.join(folder_path, f"{iteration + 1},{frame_idx + 1}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_final_figures(u_frames, mf0_frames, mf1_frames, pf_frames,
                        input_dimension, folder_path, iteration):
    subfolders = {
        "images":                   lambda u, v0, v1, pf: (np.sqrt(u**2), "gray",    0, np.max(np.abs(u))),
        "phase_field":              lambda u, v0, v1, pf: (pf, "seismic", np.min(pf), np.max(pf)),
        "motionField_0":            lambda u, v0, v1, pf: (v0, "seismic", -np.max(np.abs(v0)), np.max(np.abs(v0))),
        "motionField_1":            lambda u, v0, v1, pf: (v1, "seismic", -np.max(np.abs(v1)), np.max(np.abs(v1))),
        "motionField_magnitude":    lambda u, v0, v1, pf: (np.sqrt(v0**2+v1**2), "gray", 0, np.percentile(np.sqrt(v0**2+v1**2), 99)),
    }

    paths = {}
    for name in subfolders:
        p = os.path.join(folder_path, name)
        os.makedirs(p, exist_ok=True)
        paths[name] = p

    for j in range(len(u_frames) - 1):
        u  = u_frames[j].reshape(input_dimension, order="F")
        v0 = mf0_frames[j].reshape(input_dimension, order="F")
        v1 = mf1_frames[j].reshape(input_dimension, order="F")
        pf = pf_frames[j].reshape(input_dimension, order="F")

        for name, fn in subfolders.items():
            img, cmap, vmin, vmax = fn(u, v0, v1, pf)
            plt.figure()
            plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.axis("off")
            plt.savefig(os.path.join(paths[name],
                        f"{iteration+1},{j+1}.png"),
                        dpi=200, bbox_inches="tight")
            plt.close()