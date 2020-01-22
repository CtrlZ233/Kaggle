import numpy as np
from Network import *
from MyDataLoader import *
from torch.autograd import Variable
from scipy import ndimage
import itertools
from visdom import Visdom
from PIL import Image
from numpy import *
if __name__ == "__main__":
    viz = Visdom(env="l2_train")
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
    level_1_Net_1 = Level1().cuda()
    level_1_Net_2 = Level1().cuda()
    level_2_Net = Level2_Net().cuda()
    level_1_Net_1.load_state_dict(torch.load("./model/level_1_model/1/Net_1_17.4894.pth"))
    level_1_Net_2.load_state_dict(torch.load("./model/level_1_model/2/Net_2_17.4894.pth"))
    if exist:
        level_2_Net.load_state_dict(torch.load("./model/level_2_model/Net_272.7216.pth"))

    full_train_data = train_set(filename=full_train_filename, image_dir=image_dir, image_shape=image_shape, repeat=1)
    full_train_loader = DataLoader(dataset=full_train_data, batch_size=batch_size, shuffle=False)
    validate_data = train_set(filename=validate_filename, image_dir=image_dir, image_shape=image_shape, repeat=1)
    validate_loader = DataLoader(dataset=validate_data, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.MSELoss().cuda()
    optim = torch.optim.Adam(level_2_Net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.1)
    min_loss = 10000
    padding = 60
    for epoch in range(epoch_num):
        print("epoch:{}".format(epoch))
        print('=============training================')
        print("complete data train")
        train_loss1 = []
        for batch_image, batch_label in full_train_loader:
            batch_image = batch_image.float()

            batch_label = batch_label.float()
            batch_image = Variable(batch_image.cuda())
            batch_label = Variable(batch_label.cuda())
            prediction1 = level_1_Net_1.forward1(batch_image.view(batch_image.shape[0], 1, 96, 96))
            prediction2 = level_1_Net_2.forward2(batch_image.view(batch_image.shape[0], 1, 96, 96))
            prediction = prediction1
            prediction[:, [0, 1, 2, 3, 20, 21, 28, 29]] = (prediction1[:,
                                                           [0, 1, 2, 3, 20, 21, 28, 29]] + prediction2) / 2

            prediction = prediction.reshape((batch_image.shape[0],15,2))
            # print(prediction)
            level_2_input = torch.randn(batch_image.shape[0], 15, 32, 32)
            new_batch_image = torch.randn(batch_image.shape[0], 96+2*padding, 96+2*padding)

            for i in range(batch_image.shape[0]):
                new_batch_image[i] = torch.from_numpy(np.pad(batch_image[i].cpu().numpy(),
                                                               ((padding, padding),(padding, padding)),
                                                             'constant', constant_values=0))
            for i in range(batch_image.shape[0]):
                for j in range(15):
                    level_2_input[i, j, :, :] = \
                        new_batch_image[i, (int(prediction[i, j, 0]+padding) - 16):(int(prediction[i, j, 0]+padding) + 16),
                        (int(prediction[i, j, 1]+padding) - 16):(int(prediction[i, j, 1]+padding) + 16)]
            level_2_input = Variable(level_2_input.cuda())
            l2_prediction = level_2_Net.forward(level_2_input)
            # print(l2_prediction.shape)
            # print(batch_label.shape)
            optim.zero_grad()

            true_label = batch_label - prediction.reshape(batch_image.shape[0], 30)
            # print(true_label.shape)
            # print(l2_prediction.shape)
            loss = criterion(l2_prediction, true_label)
            train_loss1.append(float(loss.data))
            loss.backward()
            optim.step()
        loss_array = []
        print('=============validating================')
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

            prediction = prediction.reshape((batch_image.shape[0], 15, 2))
            level_2_input = torch.randn(batch_image.shape[0], 15, 32, 32)
            new_batch_image = torch.randn(batch_image.shape[0], 96 + 2 * padding, 96 + 2 * padding)

            for i in range(batch_image.shape[0]):
                new_batch_image[i] = torch.from_numpy(np.pad(batch_image[i].cpu().numpy(),
                                                             ((padding, padding), (padding, padding)),
                                                             'constant', constant_values=0))
            for i in range(batch_image.shape[0]):
                for j in range(15):
                    level_2_input[i, j, :, :] = \
                        new_batch_image[i,
                        (int(prediction[i, j, 0] + padding) - 16):(int(prediction[i, j, 0] + padding) + 16),
                        (int(prediction[i, j, 1] + padding) - 16):(int(prediction[i, j, 1] + padding) + 16)]
            level_2_input = Variable(level_2_input.cuda())
            l2_prediction = level_2_Net.forward(level_2_input)
            true_label = batch_label-prediction.reshape(batch_image.shape[0], 30)

            loss = criterion(l2_prediction, true_label)


            loss_array.append(float(loss.data))

        loss_mean = mean(loss_array)
        print("loss_mean:   {:.2f}".format(loss_mean))
        if loss_mean < min_loss:
            min_loss = loss_mean
            torch.save(level_2_Net.state_dict(), "./model/level_2_model/Net_{:.4f}.pth".format(min_loss))

        viz.line(Y=np.array([float(mean(train_loss1)) / 100]), X=np.array([epoch]), update='append', win=win1)
        viz.line(Y=np.array([float(loss_mean) / 100]), X=np.array([epoch]), update='append', win=win2)










