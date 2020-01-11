from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.CNN1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                            kernel_size=3, out_channels=32,stride=1,padding=0),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, 1))
        self.CNN2 = nn.Sequential(nn.Conv2d(in_channels=32,
                                            kernel_size=3, out_channels=32),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, 1))
        self.CNN3 = nn.Sequential(nn.Conv2d(in_channels=32,
                                            kernel_size=3, out_channels=32),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, 1))
        self.CNN4 = nn.Sequential(nn.Conv2d(in_channels=32,
                                            kernel_size=3, out_channels=64),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, 1))
        self.CNN5 = nn.Sequential(nn.Conv2d(in_channels=64,
                                            kernel_size=3, out_channels=64),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, 1))
        self.FC = nn.Sequential(
            nn.Linear(10816, 10),
            nn.Sigmoid()
        )
    def forward(self, x):
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

        x = self.FC(x)
        return x
