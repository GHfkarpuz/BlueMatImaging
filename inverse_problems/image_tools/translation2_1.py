import sys
sys.path.append("/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/wrappers/python")


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#from wrapper import FlexBoxSolver # python wrapper


image = Image.open("frame10.bmp").convert("L")

image_np = np.array(image).astype(np.float32)

translated_image = np.zeros([386,583])

for i in range(583):
    for j in range(386):
        translated_image[j][i]=image_np[j+2][i+1]

shortened_image = np.zeros([386,583])

for i in range(583):
    for j in range(386):
        shortened_image[j][i] = image_np[j][i]

np.save("shortened_image2_1.npy",shortened_image)
np.save("translated_image2_1.npy", translated_image)

# create a arrays with the actual velocity fields
v_x = np.zeros([386,583])
v_y = np.zeros([386,583])


for i in range(583):
    for j in range(386):
        v_x[j][i] = 1

for i in range(583):
    for j in range(386):
        v_y[j][i] = 2


np.save("v_x_2_1.npy", v_x)
np.save("v_y_2_1.npy", v_y)

print(image_np.shape)
print(shortened_image.shape)
print(translated_image.shape)