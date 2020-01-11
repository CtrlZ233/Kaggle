import pandas as pd
from PIL import Image
import numpy as np
import csv
def main():
    csv_data = pd.read_csv("./challenge_test/test.csv").values
    image_array = csv_data[:,1:]
    image_id = csv_data[:,0]
    image_num = image_array.shape[0]
    print(image_array.shape)
    image_array = image_array.reshape((image_num, 28, 28))
    image_prefix = "./challenge_test/image/image"
    # image_csv_path = "./challenge_testa/train_label.csv"

    for i in range(image_num):
        im = Image.fromarray(np.uint8(image_array[i]))
        path = image_prefix + str(i) + ".bmp"
        im.save(path)




main()


