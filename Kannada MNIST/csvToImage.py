import pandas as pd
from PIL import Image
import numpy as np
import csv
def main():
    csv_data = pd.read_csv("./train_data/train.csv").values
    ratio = 0.7
    train_num = int(csv_data.shape[0]*0.7)
    test_num = int(csv_data.shape[0]*0.3)
    train_array = csv_data[:train_num,1:]
    train_label = csv_data[:train_num,0]
    test_array = csv_data[train_num:, 1:]
    test_label = csv_data[train_num:, 0]
    print(train_label.shape)
    image_num = train_array.shape[0]
    print(train_array.shape)
    print(test_array.shape)
    train_array = train_array.reshape((train_num, 28, 28))
    test_array = test_array.reshape((test_num,28, 28))
    test_prefix = "./test_data/test_image/image"
    test_csv_path = "./test_data/test_label.csv"
    train_prefix = "./train_data/train_image/image"
    train_csv_path = "./train_data/train_label.csv"

    with open(test_csv_path, "w",encoding='utf8',newline='') as f:
        csv_writer = csv.writer(f)
        for i in range(test_num):
            im = Image.fromarray(np.uint8(test_array[i]))
            path = test_prefix + str(i) + ".bmp"
            im.save(path)
            data_item = ["image" + str(i)+'.bmp', str(test_label[i])]
            csv_writer.writerow(data_item)

    with open(train_csv_path, "w",encoding='utf8',newline='') as f:
        csv_writer = csv.writer(f)

        for i in range(train_num):
            im = Image.fromarray(np.uint8(train_array[i]))
            path = train_prefix + str(i) + ".bmp"
            im.save(path)
            data_item = ["image" + str(i)+'.bmp', str(train_label[i])]
            csv_writer.writerow(data_item)

main()


