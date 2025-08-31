import numpy as np
from PIL import Image

result_string = 'results/dark_to_light_1.jpg'
start_string = 'imgs/dark.jpg'
target_string = 'imgs/bww.jpg'

start_img = Image.open(start_string)
target_img = Image.open(target_string)

start_array = np.array(start_img)[:,:,0]
target_array = np.array(target_img)[:,:,0]

height = start_array.shape[0]
width = start_array.shape[1]

def pic_as_list(array):
    list_of_pixels = []
    for x in range(height):
        for y in range(width):
            color = array[x,y]
            list_of_pixels.append([x,y,color])
    return sorted(list_of_pixels, key=lambda x: (x[2], x[0], x[1]))
  
def transport_colors(start_array, target_array):
    
    list_start = pic_as_list(start_array)
    hist_target = [(target_array==color).sum() for color in range(256)]
    
    ind = 0
    for color in range(256):
        while hist_target[color] > 0:
            list_start[ind][2] = color
            hist_target[color] = hist_target[color] - 1
            ind = ind + 1
            
    target = np.zeros((height,width))
    for pixel in list_start:
        x, y, color = pixel
        target[x,y] = color   
    
    return target

final_picture = np.zeros((height,width, 3))

final_picture[:,:,0] = transport_colors(start_array, target_array)
final_picture[:,:,1] = transport_colors(start_array, target_array)
final_picture[:,:,2] = transport_colors(start_array, target_array)

final_picture = final_picture.astype(np.uint8)
img = Image.fromarray(final_picture)
img.save(result_string)

import matplotlib.pyplot as plt

start_img = Image.open(start_string).convert("L")
target_img = Image.open(target_string).convert("L")

start_array = np.array(start_img)
target_array = np.array(target_img)

hist_start, _ = np.histogram(start_array.flatten(), bins=256, range=(0, 255))
hist_target, _ = np.histogram(target_array.flatten(), bins=256, range=(0, 255))

ymax = max(hist_start.max(), hist_target.max())

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.bar(range(256), hist_start, color='deepskyblue')
plt.title("Histogram of image Dark")
plt.xlabel("Gray level (0-255)")
plt.ylabel("Pixel count")
plt.ylim(0, ymax)  

plt.subplot(1,2,2)
plt.bar(range(256), hist_target, color='deepskyblue')
plt.title("Histogram of image Light")
plt.xlabel("Gray level (0-255)")
plt.ylabel("Pixel count")
plt.ylim(0, ymax)  

plt.tight_layout()
plt.show()

from PIL import Image
import numpy as np

result_string = 'results/dark_to_light.jpg'
start_string = 'imgs/dark.jpg'
target_string = 'imgs/light.jpg'

start_img = Image.open(start_string).convert('RGB')
target_img = Image.open(target_string).convert('RGB')

target_img = target_img.resize(start_img.size, Image.Resampling.LANCZOS)

start_array = np.array(start_img)[:, :, 0]
target_array = np.array(target_img)[:, :, 0]

height, width = start_array.shape

def pic_as_list(array):
    list_of_pixels = []
    for x in range(height):
        for y in range(width):
            color = array[x, y]
            list_of_pixels.append([x, y, color])
    return sorted(list_of_pixels, key=lambda x: (x[2], x[0], x[1]))

def transport_colors(start_array, target_array):
    list_start = pic_as_list(start_array)
    hist_target = [(target_array == color).sum() for color in range(256)]

    ind = 0
    for color in range(256):
        while hist_target[color] > 0:
            list_start[ind][2] = color
            hist_target[color] -= 1
            ind += 1

    target = np.zeros((height, width))
    for pixel in list_start:
        x, y, color = pixel
        target[x, y] = color
    return target

final_picture = np.zeros((height, width, 3))
for c in range(3):
    final_picture[:, :, c] = transport_colors(start_array, target_array)

final_picture = final_picture.astype(np.uint8)
Image.fromarray(final_picture).save(result_string)

