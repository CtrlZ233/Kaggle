import torch
import numpy as np
import pandas as pd
from MyDataLoader import *
from CNN import *
from FC import *
from AlexNet import *
def is_right(prediction, label):
    index  = -1
    max = -1
    pred = prediction.data.numpy()
    for i in range(len(pred[0])):
        if pred[0][i] > max:
            max = pred[0][i]
            index = i
    print(index, label)
    if index == label:
        return True
    return False


if __name__ == "__main__":
    filename = "./train_data/train_label.csv"
    image_dir = "./train_data/binary_image"
    test_csv = "./test_data/test_label.csv"
    test_dir = "./test_data/binary_image"
    image_shape = (28, 28)
    image_label_list = []
    with open(test_csv, 'r') as f:
        lines = f.readlines()
        for line in lines:
            content = line.rstrip().split(',')
            # print(content)
            name = test_dir + "/" + content[0]
            label = content[1]
            image_label_list.append((name, label))
    image_label_list_test = image_label_list

    my_dataset = train_set(filename, image_dir, image_shape)

    exist = False

    epoch_num = 5  # 总样本循环次数
    batch_size = 10  # 训练时的一组数据的大小
    train_data_nums = 10
    # max_iterate = int((train_data_nums + batch_size - 1) / batch_size * epoch_num)  # 总迭代次数
    net = simpleNet()
    if exist:
        net.load_state_dict(torch.load("./model/LeNet/LeNet_model0.9908.pth"))
    optim = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = torch.nn.MSELoss()
    train_data = train_set(filename=filename, image_dir=image_dir, image_shape=image_shape, repeat=1)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    max_score = 0
    for epoch in range(epoch_num):
        print("===============training===============")
        index = 0
        for batch_image, batch_label in train_loader:
            loss = 0
            batch_image = batch_image.float()
            batch_label = batch_label.float()
            for i in range(batch_image.shape[0]):

                prediction = net.forward(batch_image[i].view(1, 784))
                loss = loss + criterion(prediction, batch_label[i])
                index = index + 1
            optim.zero_grad()
            loss.backward()
            optim.step()
            print("train:epoch:{}, data_index:{}/42000, loss:{:.4f}".format(epoch, index, loss))
        print("================test================")
        right_count = 0
        for i in range(len(image_label_list_test)):
            image_data_path = image_label_list_test[i][0]
            image_label = int(image_label_list_test[i][1])
            image_data = np.array(Image.open(image_data_path))
            image_data = torch.tensor(image_data).float()
            image_data = image_data.view(1, 784)
            prediction = net.forward(image_data)
            # print(prediction)
            if is_right(prediction, image_label):
                right_count = right_count + 1
            print("test:index:{}/18000".format(i))
        print("score:{:.4f}".format(right_count/len(image_label_list_test)))
        if right_count/len(image_label_list_test) > max_score:
            max_score = right_count/len(image_label_list_test)
            torch.save(net.state_dict(), "./model/FC_model/FC_model{:.4f}.pth".format(max_score))



