from torch import nn
import torch

class Level1(nn.Module):
    def __init__(self):
        super(Level1, self).__init__()
        self.CNN1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                            kernel_size=3, out_channels=32,stride=1,padding=0),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, 2))
        self.CNN2 = nn.Sequential(nn.Conv2d(in_channels=32,
                                            kernel_size=3, out_channels=32),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, 2))
        self.CNN3 = nn.Sequential(nn.Conv2d(in_channels=32,
                                            kernel_size=3, out_channels=32),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, 2))
        self.CNN4 = nn.Sequential(nn.Conv2d(in_channels=32,
                                            kernel_size=3, out_channels=64),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, 1))
        self.CNN5 = nn.Sequential(nn.Conv2d(in_channels=64,
                                            kernel_size=3, out_channels=64),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, 1))
        self.FC1 = nn.Sequential(
            nn.Linear(1024, 30),
            nn.ReLU()
        )
        self.FC2 = nn.Sequential(
            nn.Linear(1024, 8),
            nn.ReLU()
        )
    def forward1(self, x):
        # x = x.float()
        x = self.CNN1(x)

        x = self.CNN2(x)
        x = self.CNN3(x)
        x = self.CNN4(x)
        x = self.CNN5(x)
        # print(x.shape)
        x = x.view(x.shape[0], -1)  # 展开
        # print(x.shape)
        # if built_FC:
        #     (b, in_f) = x.shape  # 查看卷积层输出的tensor平铺后的形状
        #     self.FC = nn.Linear(in_f, 10)  # 全链接层

        x = self.FC1(x)
        return x
    def forward2(self, x):
        # x = x.float()
        x = self.CNN1(x)

        x = self.CNN2(x)
        x = self.CNN3(x)
        x = self.CNN4(x)
        x = self.CNN5(x)
        # print(x.shape)
        x = x.view(x.shape[0], -1)  # 展开
        # print(x.shape)
        # if built_FC:
        #     (b, in_f) = x.shape  # 查看卷积层输出的tensor平铺后的形状
        #     self.FC = nn.Linear(in_f, 10)  # 全链接层

        x = self.FC2(x)
        return x

class Level2_unit(nn.Module):
    def __init__(self):
        super(Level2_unit, self).__init__()
        self.CNN1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                            kernel_size=3, out_channels=10,stride=1,padding=0),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2,2))
        self.CNN2 = nn.Sequential(nn.Conv2d(in_channels=10,
                                            kernel_size=3, out_channels=10,stride=1,padding=0),
                                  nn.ReLU())
        self.FC = nn.Sequential(
            nn.Linear(250, 2),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.CNN1(x)
        x = self.CNN2(x)
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.FC(x)
        return x
class Level2_Net(nn.Module):
    def __init__(self):
        super(Level2_Net,self).__init__()
        self.net_list = []
        for i in range(15):
            self.net_list.append(Level2_unit())
        setattr(self, 'forward', nn.Sequential(*self.net_list))

    def forward(self, x):
        predictions =[]
        for i, net_unit in enumerate(self.net_list):
            prediction = net_unit.forward(x[:,i].view(x.shape[0], 1, 16, 16))
            predictions.append(prediction)
            # print(prediction)
        predictions = tuple(predictions)
        predictions = torch.cat(predictions, 1)
        return predictions

# net = Level2_Net()
# data = torch.randn(1, 15, 16, 16)
# data2 = torch.randn(1, 15, 16, 16)
# pred = torch.randn(1,30)
#
# prediction = net.forward(data)
#
# prediction2 = net.forward(data)

#
# # print(prediction-prediction2)
# print(net)







