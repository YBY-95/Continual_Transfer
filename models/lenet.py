import torch
import torch.nn as nn
from torchvision import models

class LeNet(nn.Module):
    def __init__(self, out_dim=4, in_channel=3, img_sz=256):
        super(LeNet, self).__init__()
        feat_map_sz = img_sz//16
        self.out_dim = out_dim
        self.n_feat = 256 * feat_map_sz * feat_map_sz

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 20, 5, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, 5, padding=2),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.linear = nn.Sequential(
            nn.Linear(self.n_feat, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(500, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        x = self.conv(x)
        x = self.linear(x.view(-1, self.n_feat))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

    def output_num(self):
        return self.out_dim



class TransferLeNet(nn.Module):

    def __init__(self, in_channel=3, img_sz=256):
        super(TransferLeNet, self).__init__()
        feat_map_sz = img_sz//4
        self.n_feat = 256 * 7 * 7

        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channel, 6, 5, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(6, 16, 5, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(16, 120, 5, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, 2),
        # )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )


        # model_alexnet = models.alexnet(pretrained=False)
        # self.features = model_alexnet.features
        self.flatten = nn.Flatten()

    def features(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return x

    def output_num(self):
        return self.n_feat


def LeNetC(out_dim=10):  # LeNet with color input
    return LeNet(out_dim=out_dim, in_channel=3, img_sz=256)