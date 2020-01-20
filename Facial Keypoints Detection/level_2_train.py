import numpy as np
from Network import *
from MyDataLoader import *
from torch.autograd import Variable
import itertools
from visdom import Visdom

from numpy import *
if __name__ == "__main__":
    # viz = Visdom(env="loss")
    # x, y = 0, 0
    # win1 = viz.line(np.array([y]), np.array([x]))
    # win2 = viz.line(np.array([y]), np.array([x]))
    full_train_filename = "./train_data/train_complete_data.csv"
    half_train_filename = './train_data/train_half_data.csv'
    validate_filename = './train_data/validate_complete_data.csv'
    image_dir = "./train_data/image"
    image_shape = (96, 96)
    epoch_num = 100  # 总样本循环次数
    batch_size = 1  # 训练时的一组数据的大小
    exist = True
    level_1_Net_1 = Level1().cuda()
    level_1_Net_2 = Level1().cuda()
    level_1_Net_1.load_state_dict(torch.load("./model/level_1_model/1/Net_1_17.8331.pth"))
    level_1_Net_2.load_state_dict(torch.load("./model/level_1_model/2/Net_2_17.8331.pth"))
    level_2_Net = Level2_Net()
    full_train_data = train_set(filename=validate_filename, image_dir=image_dir, image_shape=image_shape, repeat=1)
    full_train_loader = DataLoader(dataset=full_train_data, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.MSELoss(reduce=True).cuda()
    optim = torch.optim.Adam(level_2_Net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.1)
    min_loss = 10000
    for epoch in range(epoch_num):
        # print("epoch:{}".format(epoch))
        # print('=============training================')
        # print("complete data train")
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

            prediction = prediction.reshape((batch_size,15,2))
            loss = criterion(prediction,batch_label.reshape((batch_size, 15, 2)))
            # print(loss/30)
            print(loss)
            # print(prediction-batch_label.reshape((batch_size, 15, 2)))



