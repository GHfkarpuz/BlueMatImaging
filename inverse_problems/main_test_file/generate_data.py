import numpy as np
import cv2

import matplotlib.pyplot as plt
import os


def generate_meniscus_dataset(
    num_frames=6, 
    shape=(2048, 2048), 
    wall_thickness_arc=35, # Beliebige Dicke des Meniskus (z. B. 15, 35, 60 Pixel)
    bg_val=95.0,           # Dunkler Hintergrund (float)
    i_outer_val=96.0,      # Äußere Schicht des Meniskus (float)
    i_inner_val=160.0,     # Innere Schicht des Meniskus (float)
    wall_val=190.0,        # Porenwand (heller als i_inner_val, z. B. 200.0)
    noise_level=2.4,       # Rausch-Level (z. B. 0.0=aus, 1.0=starkes Rauschen)
    i0=50.0,               # Photonenzahl für Rausch-Simulation
    gaussNoise = True
):
    H, W = shape
    
    # Sicherstellen, dass die Hierarchie eingehalten wird
    assert i_inner_val < wall_val, "Die hellste Meniskus-Schicht muss dunkler als die Porenwand sein!"
    assert bg_val < i_outer_val, "Der Hintergrund muss dunkler als der Meniskus sein!"

    frames_clean = []
    frames_noisy = []
    edges_gt = []
    flow_gt = []

    # ---------------------------------------------------------
    # 1. DEFINITION DER SCHRÄGEN PORENWÄNDE
    # ---------------------------------------------------------
    poren_waende = [
        ((200, 0), (250, H)),   # Wand 0 (Links)
        ((750, 0), (700, H)),   # Wand 1 -> Pore 1 (Links)
        ((1300, 0), (1350, H)), # Wand 2 -> Pore 2 (Mitte - LEER)
        ((1850, 0), (1800, H))  # Wand 3 -> Pore 3 (Rechts)
    ]
    
    wall_thickness = 65
    half_wall = wall_thickness / 2.0

    def get_wall_x(wall_idx, y_pos):
        (x1, y1), (x2, y2) = poren_waende[wall_idx]
        return x1 + (x2 - x1) * (y_pos / float(H))

    # Pre-Berechnung der Intensitätswerte von außen nach innen als float32
    layer_intensities = np.linspace(i_outer_val, i_inner_val, wall_thickness_arc, dtype=np.float32)

    # ---------------------------------------------------------
    # 2. KONFIGURATION DER MENISKEN (Pore 1 & Pore 3)
    # ---------------------------------------------------------
    ellipses_config = [
        {
            "pore_left_wall": 0, "pore_right_wall": 1, # Pore 1 (Links)
            "center_y": 320.0,
            "axes_y": 120.0,
            "velocity_y": 2.5,
            "scale_y_rate": 1.6,
        },
        {
            "pore_left_wall": 2, "pore_right_wall": 3, # Pore 3 (Rechts)
            "center_y": 200.0,
            "axes_y": 140.0,
            "velocity_y": 1.2,
            "scale_y_rate": 6.0,
        }
    ]

    for t in range(num_frames):
        # Arbeitsbild mit dunkelgrauem Hintergrund als Float32 initialisieren
        img = np.full((H, W), float(bg_val), dtype=np.float32)
        combined_ellipse_mask = np.zeros((H, W), dtype=np.uint8)
        
        # Bewegungsfeld MUSS ZWINGEND Float32 sein
        flow = np.zeros((H, W, 2), dtype=np.float32)

        # -----------------------------------------------------
        # SCHRITT A: MENISKEN ZUERST ZEICHNEN
        # -----------------------------------------------------
        for config in ellipses_config:
            curr_cy = config["center_y"] + t * config["velocity_y"]
            next_cy = config["center_y"] + (t + 1) * config["velocity_y"]

            curr_ay = config["axes_y"] + t * config["scale_y_rate"]
            next_ay = config["axes_y"] + (t + 1) * config["scale_y_rate"]

            # Wandgrenzen berechnen
            x_left_wall = get_wall_x(config["pore_left_wall"], curr_cy) + 0.0*half_wall
            x_right_wall = get_wall_x(config["pore_right_wall"], curr_cy) - 0.0*half_wall
            
            curr_cx = (x_left_wall + x_right_wall) / 2.0
            ax = (x_right_wall - x_left_wall) / 2.0

            # Werte für das nächste Frame (für Optischen Fluss)
            next_x_left = get_wall_x(config["pore_left_wall"], next_cy) + half_wall
            next_x_right = get_wall_x(config["pore_right_wall"], next_cy) - half_wall
            next_cx = (next_x_left + next_x_right) / 2.0
            next_ax = (next_x_right - next_x_left) / 2.0

            full_arc_mask = np.zeros((H, W), dtype=np.uint8)

            # Dicke Bogenwand ohne Lücken zeichnen (von außen nach innen)
            for layer_idx in range(wall_thickness_arc):
                radius_offset = (wall_thickness_arc // 2) - layer_idx
                intensity = layer_intensities[layer_idx]

                r_x = max(1.0, ax + radius_offset)
                r_y = max(1.0, curr_ay + radius_offset)

                layer_mask = np.zeros((H, W), dtype=np.uint8)
                th = 2 if wall_thickness_arc > 15 else 1
                
                cv2.ellipse(
                    layer_mask, 
                    (int(curr_cx), int(curr_cy)), 
                    (int(r_x), int(r_y)), 
                    angle=0, startAngle=0, endAngle=360, 
                    color=255, thickness=th
                )
                layer_mask[:int(curr_cy), :] = 0  # Untere Hälfte abschneiden
                
                img[layer_mask > 0] = float(intensity)
                full_arc_mask = cv2.bitwise_or(full_arc_mask, layer_mask)

            combined_ellipse_mask = cv2.bitwise_or(combined_ellipse_mask, full_arc_mask)

            # Optischen Fluss (Float32) berechnen
            y_indices, x_indices = np.where(full_arc_mask > 0)
            for y, x in zip(y_indices, x_indices):
                rel_x = x - curr_cx
                rel_y = y - curr_cy
                
                scale_factor_x = next_ax / ax if ax > 0 else 1.0
                scale_factor_y = next_ay / curr_ay if curr_ay > 0 else 1.0
                
                x_next = next_cx + rel_x * scale_factor_x
                y_next = next_cy + rel_y * scale_factor_y
                
                flow[y, x, 0] = float(x_next - x)
                flow[y, x, 1] = float(y_next - y)

        # -----------------------------------------------------
        # SCHRITT B: PORENWÄNDE NACHTRÄGLICH DARÜBER ZEICHNEN
        # -----------------------------------------------------
        background_mask = np.zeros((H, W), dtype=np.uint8)
        for pt1, pt2 in poren_waende:
            cv2.line(background_mask, pt1, pt2, color=255, thickness=wall_thickness)

        # 1. Porenwand mit wall_val (200.0) zeichnen -> verdeckt Meniskus sauber
        img[background_mask > 0] = float(wall_val)

        # 2. Bewegungsfeld an der starren Wand auf 0.0 setzen
        flow[background_mask > 0] = 0.0

        # 3. Kantenfeld v_gt (Float32)
        """
        edge_mask = cv2.Canny(combined_ellipse_mask, 100, 200)
        edge_mask[background_mask > 0] = 255
        v_gt = (1.0 - (edge_mask.astype(np.float32) / 255.0)).astype(np.float32)
        """

        total_region = cv2.bitwise_or(combined_ellipse_mask, background_mask)
        v_gt = (total_region.astype(np.float32) / 255.0)

        # -----------------------------------------------------
        # SCHRITT C: POISSON-RAUSCHEN (IN FLOAT32)
        # -----------------------------------------------------
        if noise_level > 0.0:
            if gaussNoise:
                # noise_level steuert die Standardabweichung sigma (in Grauwerten von 0..255)
                # noise_level = 1.0 -> sigma = 10.0 (deutlich sichtbares Rauschen)
                # noise_level = 2.0 -> sigma = 20.0 (starkes Rauschen)
                sigma = noise_level * 10.0  
                
                # Zufaellige Gauss-Werte mit Mittelwert=0 und Standardabweichung=sigma erzeugen
                gaussian_noise = np.random.normal(loc=0.0, scale=sigma, size=(H, W)).astype(np.float32)
                
                # Rauschen direkt auf das saubere Bild addieren
                img_noisy = img + gaussian_noise
                
                # Wertebereich sauber auf [0.0, 255.0] begrenzen
                img_noisy = np.clip(img_noisy, 0.0, 255.0)
            else:
                # Normierung auf [0, 1] für Beer-Lambert
                mu_normalized = img / 255.0
                
                i0_eff = max(10.0, i0 / (1.0 + noise_level * 3.0))
                
                # Poisson-Prozess anwenden
                I_clean = i0_eff * np.exp(-mu_normalized)
                I_noisy = np.random.poisson(I_clean).astype(np.float32)
                I_noisy = np.maximum(I_noisy, 1.0)
                
                # Rekonstruktion & Rückskalierung auf [0.0, 255.0] als Float32
                mu_noisy = -np.log(I_noisy / i0_eff)
                img_noisy = np.clip(mu_noisy * 255.0, 0.0, 255.0).astype(np.float32)
        else:
            img_noisy = np.clip(img, 0.0, 255.0).astype(np.float32)

        img_clean = np.clip(img, 0.0, 255.0).astype(np.float32)

        frames_clean.append(img_clean)
        frames_noisy.append(img_noisy)
        edges_gt.append(v_gt)
        flow_gt.append(flow)

    return (
        np.array(frames_clean, dtype=np.float32), # (num_frames, 2048, 2048) in float32 [0.0..255.0]
        np.array(frames_noisy, dtype=np.float32), # (num_frames, 2048, 2048) in float32 [0.0..255.0]
        np.array(edges_gt, dtype=np.float32),    # (num_frames, 2048, 2048) in float32 [0.0..1.0]
        np.array(flow_gt, dtype=np.float32)      # (num_frames, 2048, 2048, 2) in float32
    )

# ==========================================
# Test der Funktion
# ==========================================
if __name__ == "__main__":
    u_clean, u_noisy, v_gt, w_gt = generate_meniscus_dataset(
    )
    folder_path = "generatedData"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    np.save(folder_path+"/u_clean.npy", u_clean)
    np.save(folder_path+"/u_noisy.npy", u_noisy)
    np.save(folder_path+"/v_gt.npy", v_gt)
    np.save(folder_path+"/w_gt.npy", w_gt)
    inputDimension=(2048, 2048)
    for j in range(len(u_clean)):
        flow_x = w_gt[j, :, :, 0]  # Shape: (2048, 2048)
        flow_y = w_gt[j, :, :, 1]  # Shape: (2048, 2048)
        result_1_v = flow_x.reshape(inputDimension, order="F")
        result_2_v = flow_y.reshape(inputDimension, order="F")

        result_1_u = u_clean[j].reshape(inputDimension, order="F")
        result_1_phase_field = v_gt[j].reshape(inputDimension, order="F")

        result_1_u_noisy = u_noisy[j].reshape(inputDimension, order="F")

        

        #frame_start = j * nPx
        #frame_end = (j + 1) * nPx
        #result_1_u = reconstructed_image[frame_start:frame_end].reshape(inputDimension, order="F")

        magnitude_v = np.sqrt(result_1_v**2 + result_2_v**2)
        vmax1 = np.max(np.abs(result_1_v))
        vmax1_mag = np.percentile(np.abs(result_1_v), 99)
        vmax2 = np.max(np.abs(result_2_v))
        vmax2_mag = np.percentile(np.abs(result_2_v), 99)
        vmax_mag = np.percentile(magnitude_v, 99)
        
        magnitude_u = np.sqrt(result_1_u**2)
        umax1 = np.max(np.abs(result_1_u))
        umax_mag = np.percentile(magnitude_u, 99)

        magnitude_u_noisy = np.sqrt(result_1_u_noisy**2)
        umax1_noisy = np.max(np.abs(result_1_u_noisy))
        umax_mag_noisy = np.percentile(magnitude_u_noisy, 99)

        

        phase_field1 = np.max(np.abs(result_1_phase_field))
        phase_field2 = np.min(np.abs(result_1_phase_field))

        print(f"vmax1 for frame {j} is", vmax1)
        print(f"vmax2 for frame {j} is", vmax2)
        print(f"vmax1_mag for frame {j} is", vmax1_mag)
        print(f"vmax2_mag for frame {j} is", vmax2_mag)
        
        print(f"umax1 for frame {j} is", umax1)    
        print(f"phase_field1 for frame {j} is", phase_field1)
        print(f"max of phase_field1 for frame {j} is", np.max(result_1_phase_field))
        print(f"min phase_field1 for frame {j} is", np.min(result_1_phase_field))          

        plt.figure(figsize=(7,7))
        
        plt.subplot(3,2,1)
        plt.title("magnitude of u")
        plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax1)
        #plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax_mag)
        plt.axis("off")

        plt.subplot(3,2,2)
        plt.title("magnitude of u_noisy")
        plt.imshow(magnitude_u_noisy, cmap="gray", vmin=0, vmax=umax1_noisy)
        #plt.imshow(magnitude_u, cmap="gray", vmin=0, vmax=umax_mag)
        plt.axis("off")
        
        plt.subplot(3,2,3)
        plt.title("flow_x")
        plt.imshow(result_1_v, cmap="seismic", vmin=-vmax1, vmax=vmax1)
        plt.axis("off")

        plt.subplot(3,2,4)
        plt.title("flow_y")
        plt.imshow(result_2_v, cmap="seismic", vmin=-vmax2, vmax=vmax2)
        plt.axis("off")
        
        plt.subplot(3,2,5)
        plt.title("segments")
        plt.imshow(result_1_phase_field, cmap="gray", vmin=0.0, vmax=1.0)
        plt.axis("off")

        
        #plt.savefig(f"result{dataTerm_norm_list[dataTerm_norm]}{reg_u_norm_list[reg_u_norm]}{motionTerm_u_norm_list[motionTerm_u_norm]}{motionTerm_u_list[motionTerm_u]}{motionTerm_v_norm_list[motionTerm_v_norm]}{motionTerm_v_list[motionTerm_v]}{reg_v_norm_list[reg_v_norm]}{i+1},{j+1}.png", dpi=200, bbox_inches="tight")
        file_path = os.path.join(folder_path, f"{j+1}.png")
        plt.savefig(file_path, dpi=200, bbox_inches="tight")
        plt.close()
    
    print("Shape Bild-Sequenz:", u_noisy.shape)   # (5, 256, 256)
    print("Shape Kanten-GT:", v_gt.shape)         # (5, 256, 256)
    print("Shape Bewegungs-GT:", w_gt.shape)      # (5, 256, 256, 2)