import sys
sys.path.append("/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/wrappers/python")


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#from wrapper import FlexBoxSolver # python wrapper


image = Image.open("frame10.bmp").convert("L")

image_np = np.array(image).astype(np.float32)

translated_image = np.zeros([387,584])

for i in range(584):
    for j in range(387):
        translated_image[j][i]=image_np[j+1][i]

shortened_image = np.zeros([387,584])

for i in range(584):
    for j in range(387):
        shortened_image[j][i] = image_np[j][i]

np.save("shortened_image.npy",shortened_image)
np.save("translated_image.npy", translated_image)

print(image_np.shape)
print(shortened_image.shape)
print(translated_image.shape)