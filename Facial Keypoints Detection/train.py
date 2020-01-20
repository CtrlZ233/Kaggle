import numpy as np
from Network import *
from MyDataLoader import *
from torch.autograd import Variable
import itertools
from visdom import Visdom

from numpy import *

train_loss = []
test_loss = []


if __name__ == "__main__":
    viz = Visdom(env="loss")
    x, y = 0, 0
    win1 = viz.line(np.array([y]), np.array([x]))
    win2 = viz.line(np.array([y]), np.array([x]))
    full_train_filename = "./train_data/train_complete_data.csv"
    half_train_filename = './train_data/train_half_data.csv'
    validate_filename = './train_data/validate_complete_data.csv'
    image_dir = "./train_data/image"
    image_shape = (96, 96)
    epoch_num = 100  # 总样本循环次数
    batch_size = 7  # 训练时的一组数据的大小
    exist = False
    full_train_data = train_set(filename=full_train_filename, image_dir=image_dir, image_shape=image_shape, repeat=1)
    full_train_loader = DataLoader(dataset=full_train_data, batch_size=batch_size, shuffle=False)
    half_train_data = train_set(filename=half_train_filename, image_dir=image_dir, image_shape=image_shape, repeat=1)
    half_train_loader = DataLoader(dataset=half_train_data, batch_size = batch_size, shuffle=False)
    validate_data  =train_set(filename=validate_filename, image_dir=image_dir, image_shape=image_shape, repeat=1)
    validate_loader = DataLoader(dataset=validate_data, batch_size = batch_size, shuffle=False)
    level_1_Net_1 = Level1().cuda()
    level_1_Net_2 = Level1().cuda()
    if exist:
        level_1_Net_1.load_state_dict(torch.load("./model/level_1_model/1/Net_1_116.7519.pth"))
        level_1_Net_2.load_state_dict(torch.load("./model/level_1_model/2/Net_2_116.7519.pth"))
    criterion = torch.nn.MSELoss().cuda()
    optim = torch.optim.Adam(itertools.chain(level_1_Net_1.parameters(), level_1_Net_2.parameters()), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optim,step_size=100,gamma=0.1)
    min_loss = 10000
    epoch_list=[]
    for epoch in range(epoch_num):
        print("epoch:{}".format(epoch))
        print('=============training================')
        print("complete data train")
        train_loss1 = []
        for batch_image, batch_label in full_train_loader:
            batch_image = batch_image.float()
            batch_label = batch_label.float()
            batch_image =Variable(batch_image.cuda())
            batch_label = Variable(batch_label.cuda())
            prediction1 = level_1_Net_1.forward1(batch_image.view(batch_image.shape[0], 1, 96, 96))
            prediction2 = level_1_Net_2.forward2(batch_image.view(batch_image.shape[0], 1, 96, 96))
            prediction = prediction1
            prediction[:,[0, 1, 2, 3, 20, 21, 28, 29]] = (prediction1[:,[0, 1, 2, 3, 20, 21, 28, 29]]+prediction2)/2
            optim.zero_grad()
            loss = criterion(prediction, batch_label)
            # print(loss)
            train_loss1.append(float(loss.data))
            loss.backward()
            optim.step()
        print('half data_train')
        for batch_image, batch_label in half_train_loader:
            batch_image = batch_image.float()
            batch_label = batch_label.float()
            batch_image = Variable(batch_image.cuda())
            batch_label = Variable(batch_label.cuda())
            prediction1 = level_1_Net_1.forward1(batch_image.view(batch_image.shape[0], 1, 96, 96))
            prediction2 = level_1_Net_2.forward2(batch_image.view(batch_image.shape[0], 1, 96, 96))
            prediction = (prediction2 + prediction1[:,[0, 1, 2, 3, 20, 21, 28, 29]])/2
            # print(prediction.shape)
            # print(batch_label.shape)
            optim.zero_grad()
            loss = criterion(prediction, batch_label)
            train_loss1.append(float(loss.data))
            loss.backward()
            optim.step()
        train_loss.append(mean(train_loss1))
        print('=============validating================')
        loss_array = []
        for batch_image, batch_label in validate_loader:
            batch_image = batch_image.float()
            batch_label = batch_label.float()
            batch_image = Variable(batch_image.cuda())
            batch_label = Variable(batch_label.cuda())
            # print(batch_image.shape)
            prediction1 = level_1_Net_1.forward1(batch_image.view(batch_image.shape[0], 1, 96, 96))
            prediction2 = level_1_Net_2.forward2(batch_image.view(batch_image.shape[0], 1, 96, 96))
            prediction = prediction1
            prediction[:, [0, 1, 2, 3, 20, 21, 28, 29]] = (prediction1[:,
                                                           [0, 1, 2, 3, 20, 21, 28, 29]] + prediction2) / 2
            loss = criterion(prediction, batch_label)

            loss_array.append(float(loss.data))
        loss_meam = mean(loss_array)
        test_loss.append(loss_meam)
        print("loss_mean:   {:.2f}".format(loss_meam))
        if loss_meam < min_loss:
            min_loss = loss_meam
            torch.save(level_1_Net_1.state_dict(), "./model/level_1_model/1/Net_1_{:.4f}.pth".format(min_loss))
            torch.save(level_1_Net_2.state_dict(), "./model/level_1_model/2/Net_2_{:.4f}.pth".format(min_loss))
        epoch_list.append(epoch)
        viz.line(Y=np.array([float(loss_meam)/100]), X=np.array([epoch]) ,update='append',win=win1)
        viz.line(Y=np.array([float(mean(train_loss1))/100]), X=np.array([epoch]), update='append', win=win2)