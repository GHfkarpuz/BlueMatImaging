import imageio.v2 as imageio
import numpy as np
import os
import cv2

# Pfade zu deinen Ordnern mit den Bildern
image_folder1 = '/data/dust/user/karpuzfa/resultBlueMatMedia34CorrectedNormsForPhaseFieldmoreFsdframessdgf/resultBlueMatMedia34CorrectedNormsForPhaseFieldmoreFsdframessdgfresizeValue0.25epsilon0.0011.0L20.2TVIso1.0L21.5L1IsoUpwindOpticalFlow0.5L21.0L2OpticalFlow0.5L11.0L21.0L2OpticalFlow1.1TVIso1.0TVIso/images/rawImages'
image_folder2 = '/data/dust/user/karpuzfa/resultBlueMatMedia34CorrectedNormsForPhaseFieldmoreFsdframessdgf/resultBlueMatMedia34CorrectedNormsForPhaseFieldmoreFsdframessdgfresizeValue0.25epsilon0.0011.0L20.2TVIso1.0L21.5L1IsoUpwindOpticalFlow0.5L21.0L2OpticalFlow0.5L11.0L21.0L2OpticalFlow1.1TVIso1.0TVIso/images/preDenoisedImages'
image_folder3 = '/data/dust/user/karpuzfa/resultBlueMatMedia34CorrectedNormsForPhaseFieldmoreFsdframessdgf/resultBlueMatMedia34CorrectedNormsForPhaseFieldmoreFsdframessdgfresizeValue0.25epsilon0.0011.0L20.2TVIso1.0L21.5L1IsoUpwindOpticalFlow0.5L21.0L2OpticalFlow0.5L11.0L21.0L2OpticalFlow1.1TVIso1.0TVIso/images/calcImages'
fps = 10  # Bilder pro Sekunde

# Alle PNG-Dateien alphabetisch/numerisch sortiert einlesen
images1 = sorted([f for f in os.listdir(image_folder1) if f.endswith('.png')])
images2 = sorted([f for f in os.listdir(image_folder2) if f.endswith('.png')])
images3 = sorted([f for f in os.listdir(image_folder3) if f.endswith('.png')])

image_list = []



def resize_to_height(img, h):
    scale = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), h))



# Bilder paarweise parallel durchlaufen
for filename1, filename2, filename3 in zip(images1, images2, images3):
    # Vollständige Pfade zu den Einzelbildern erstellen
    path1 = os.path.join(image_folder1, filename1)
    path2 = os.path.join(image_folder2, filename2)
    path3 = os.path.join(image_folder3, filename3)

    
    # Bilder einlesen
    img1 = imageio.imread(path1)
    img2 = imageio.imread(path2)
    img3 = imageio.imread(path3)

    target_h = min(img1.shape[0], img2.shape[0], img3.shape[0])

    img1 = resize_to_height(img1, target_h)
    img2 = resize_to_height(img2, target_h)
    img3 = resize_to_height(img3, target_h)

    # Bilder nebeneinander anordnen (horizontal stapeln)
    combined = np.hstack((img1, img2, img3))
    
    # Zur Liste hinzufügen
    image_list.append(combined)

# Als MP4-Video speichern
imageio.mimsave('media3Long150Frames.mp4', image_list, fps=fps)
print("Video erfolgreich erstellt!")