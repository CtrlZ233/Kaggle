import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import os

class train_set(Dataset):
    def __init__(self, filename, image_dir, image_shape, repeat=1):
        super(train_set, self).__init__()
        self.image_label_list = self.read_file(filename)
        self.image_dir = image_dir
        self.filename = filename
        self.repeat = repeat
        self.image_shape = image_shape
        self.toTensor = transforms.ToTensor
        self.len = len(self.image_label_list)

    def __getitem__(self, item):
        index = item % self.len
        image, label = self.image_label_list[index]
        image_path = self.image_dir + "/" + image
        im_array = np.array(Image.open(image_path))
        image_data = torch.tensor(im_array)
        return image_data, label

    def read_file(self, filename):
        image_label_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                content = line.rstrip().split(',')
                # print(content)
                name  = content[0]
                name = 'image_' + name + '.bmp'
                label = content[1:]
                label = list(map(np.float64, label))
                label = np.array(label)
                image_label_list.append((name, label))
        return image_label_list

    def __len__(self):
        if self.repeat == None:
            data_len = 60000
        else:
            data_len = self.len * self.repeat
        return data_len



if __name__ == "__main__":

    filename = "./train_data/validate_complete_data.csv"
    image_dir = "./train_data/image"
    image_shape = (96, 96)
    my_dataset = train_set(filename, image_dir, image_shape)

    epoch_num = 1  # 总样本循环次数
    batch_size = 7  # 训练时的一组数据的大小
    train_data_nums=10
    max_iterate = int((train_data_nums + batch_size - 1) / batch_size * epoch_num)  # 总迭代次数
    train_data = train_set(filename=filename, image_dir=image_dir,image_shape=image_shape, repeat=1)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

    for epoch in range(epoch_num):
        for batch_image, batch_label in train_loader:
            image = batch_image[0, :]
            image = image.numpy()  # image=np.array(image)
            print("batch_image.shape:{},batch_label:{}".format(batch_image.shape, batch_label.shape))

