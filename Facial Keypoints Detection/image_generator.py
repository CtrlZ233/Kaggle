import numpy as np
import pandas as pd
from PIL import Image
df = pd.read_csv("./training/training.csv")
image_values = df['Image'].values
for i, image_value in enumerate(image_values):
    image_value  = image_value.split(' ')
    image_value = [eval(x) for x in image_value]
    image_array = np.array(image_value).reshape((96,96))
    image = Image.fromarray(np.uint8(image_array))
    image.save("./train_data/image/image_"+ str(i) + ".bmp")
df.iloc[:, :-1].to_csv("train_data/train_label.csv")