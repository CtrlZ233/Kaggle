import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
image_path = './train_data/image/image_3.bmp'
df = pd.read_csv('./train_data/train_label.csv')
# print(df.iloc[0])
indexs = np.array(df.iloc[3])
indexs = indexs[1:]
indexs = indexs.reshape(15,2)
raw_image = Image.open(image_path)
show_image = np.array(raw_image)
print(indexs)
for index in indexs:
    show_image[int(index[1]), int(index[0])] = 0
show_image = Image.fromarray(np.uint8(show_image))
fig = plt.figure()
plt.subplot(211)
plt.imshow(raw_image)
plt.title('raw image')
plt.axis('off')
plt.subplot(212)
plt.imshow(show_image)
plt.title('show image')
plt.axis('off')
plt.show()
