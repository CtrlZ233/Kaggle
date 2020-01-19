from torch import nn
class Level1(nn.Module):
    def __init__(self):
        super(Level1, self).__init__()
        self.CNN1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                            kernel_size=4, out_channels=20,stride=1,padding=0),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, 2))
        self.CNN2 = nn.Sequential(nn.Conv2d(in_channels=20,
                                            kernel_size=3, out_channels=40),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, 2))
        self.CNN3 = nn.Sequential(nn.Conv2d(in_channels=40,
                                            kernel_size=3, out_channels=60),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, 2))
        self.CNN4 = nn.Sequential(nn.Conv2d(in_channels=60,
                                            kernel_size=3, out_channels=80),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, 2))
        self.CNN5 = nn.Sequential(nn.Conv2d(in_channels=80,
                                            kernel_size=3, out_channels=100),
                                  nn.ReLU())
        self.FC1 = nn.Sequential(
            nn.Linear(400, 30),
            nn.ReLU()
        )
        self.FC2 = nn.Sequential(
            nn.Linear(400, 8),
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

# class Level2(nn.Module):


# from torch import nn
# class Level1(nn.Module):
#     def __init__(self):
#         super(Level1, self).__init__()
#         self.CNN1 = nn.Sequential(nn.Conv2d(in_channels=1,
#                                             kernel_size=3, out_channels=32,stride=1,padding=0),
#                                   nn.ReLU(),
#                                   nn.MaxPool2d(2, 2))
#         self.CNN2 = nn.Sequential(nn.Conv2d(in_channels=32,
#                                             kernel_size=3, out_channels=32),
#                                   nn.ReLU(),
#                                   nn.MaxPool2d(2, 2))
#         self.CNN3 = nn.Sequential(nn.Conv2d(in_channels=32,
#                                             kernel_size=3, out_channels=32),
#                                   nn.ReLU(),
#                                   nn.MaxPool2d(2, 2))
#         self.CNN4 = nn.Sequential(nn.Conv2d(in_channels=32,
#                                             kernel_size=3, out_channels=64),
#                                   nn.ReLU(),
#                                   nn.MaxPool2d(2, 1))
#         self.CNN5 = nn.Sequential(nn.Conv2d(in_channels=64,
#                                             kernel_size=3, out_channels=64),
#                                   nn.ReLU(),
#                                   nn.MaxPool2d(2, 1))
#         self.FC1 = nn.Sequential(
#             nn.Linear(1024, 30),
#             nn.ReLU()
#         )
#         self.FC2 = nn.Sequential(
#             nn.Linear(1024, 8),
#             nn.ReLU()
#         )
#     def forward1(self, x):
#         # x = x.float()
#         x = self.CNN1(x)
#
#         x = self.CNN2(x)
#         x = self.CNN3(x)
#         x = self.CNN4(x)
#         x = self.CNN5(x)
#         # print(x.shape)
#         x = x.view(x.shape[0], -1)  # 展开
#         # print(x.shape)
#         # if built_FC:
#         #     (b, in_f) = x.shape  # 查看卷积层输出的tensor平铺后的形状
#         #     self.FC = nn.Linear(in_f, 10)  # 全链接层
#
#         x = self.FC1(x)
#         return x
#     def forward2(self, x):
#         # x = x.float()
#         x = self.CNN1(x)
#
#         x = self.CNN2(x)
#         x = self.CNN3(x)
#         x = self.CNN4(x)
#         x = self.CNN5(x)
#         # print(x.shape)
#         x = x.view(x.shape[0], -1)  # 展开
#         # print(x.shape)
#         # if built_FC:
#         #     (b, in_f) = x.shape  # 查看卷积层输出的tensor平铺后的形状
#         #     self.FC = nn.Linear(in_f, 10)  # 全链接层
#
#         x = self.FC2(x)
#         return x