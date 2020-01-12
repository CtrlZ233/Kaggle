import numpy as np
from Network import *
from MyDataLoader import *
if __name__ == "__main__":

    filename = "./train_data/train_complete_data.csv"
    image_dir = "./train_data/image"
    image_shape = (96, 96)
    my_dataset = train_set(filename, image_dir, image_shape)

    epoch_num = 30  # 总样本循环次数
    batch_size = 7  # 训练时的一组数据的大小

    train_data = train_set(filename=filename, image_dir=image_dir, image_shape=image_shape, repeat=1)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    net = CNN(batch_size)
    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.99)
    for epoch in range(epoch_num):
        for batch_image, batch_label in train_loader:
            batch_image = batch_image.float()
            batch_label = batch_label.float()
            prediction = net.forward(batch_image.view(batch_size, 1, 96, 96))
            # print(prediction.shape)
            # print(batch_label.shape)
            optim.zero_grad()
            loss = criterion(prediction, batch_label)
            print(loss.sum())
            loss.backward()
            optim.step()
