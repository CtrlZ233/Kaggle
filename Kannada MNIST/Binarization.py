import numpy as np
from PIL import Image
import os

origin_image_path = "./challenge_test/image"
save_path = "./challenge_test/binary_image"
image_list = os.listdir(origin_image_path)
image_list.sort()
print(image_list[:5])
for name in image_list:
    path = origin_image_path + "/" + name
    origin_array = np.array(Image.open(path))
    means = np.mean(origin_array)
    origin_array[origin_array <= means] = 0
    origin_array[origin_array>means] = 255

    im = Image.fromarray(np.uint8(origin_array))

    im.save(save_path + "/" + name)
    array = np.array(Image.open(save_path + "/" + name))
    print(np.logical_and(np.any(array[origin_array < 255]), np.any(array[origin_array > 0])))