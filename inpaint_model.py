import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, required
from torch.autograd import Variable
from torchvision import models

from Inpaint_base_model import *


class Inpaint_generator(nn.Module):
    def __init__(self, input_nc=2):
        super(Inpaint_generator, self).__init__()

        # Initialize Network
        model = [
            PartialConv2d(in_ch=input_nc, out_ch=32, K=3, S=2, P=1),
            PartialConv2d(in_ch=32, out_ch=128, K=3, S=2, P=1)
        ]
        for _ in range(3):
            model += [Partial_ResidualBlock(128)]

        self.model_initial = nn.Sequential(*model)

        model = [PartialConv2d(in_ch=128, out_ch=256, K=3, S=1, P=1),
                 Partial_ResidualBlock(256),
                 nn.Upsample(scale_factor=2, mode='nearest'), PartialConv2d(
                     256, 128),
                 Partial_ResidualBlock(128),
                 nn.Upsample(scale_factor=2,
                             mode='nearest'), PartialConv2d(128, 64),
                 Partial_ResidualBlock(64),
                 PartialConv2d(in_ch=64, out_ch=1, K=7, S=1, P=3),
                 ]
        self.model_F = nn.Sequential(*model)
        self.model_T1 = nn.Sequential(*model)
        self.model_T1c = nn.Sequential(*model)
        self.model_T2 = nn.Sequential(*model)

        # model += [
        #     nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.InstanceNorm2d(64),
        #     nn.LeakyReLU(inplace=True),
        #     nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.InstanceNorm2d(32),
        #     nn.LeakyReLU(inplace=True))]

    def forward(self, x):
        x = self.model_initial(x)
        F, T1, T1c, T2 = self.model_F(x), self.model_T1(
            x), self.model_T1c(x), self.model_T2(x)
        return torch.cat((F, T1, T1c, T2), 1)


class Inpaint_discriminator(nn.Module):
    def __init__(self, input_nc=2):
        super(Inpaint_discriminator, self).__init__()

        model = [SN_Conv(in_ch=input_nc, out_ch=32, K=3, S=2, P=1)]
        model += [SN_Conv(in_ch=32, out_ch=64, K=3, S=2, P=1)]
        model += [SN_Conv(in_ch=64, out_ch=256, K=3, S=1, P=1)]
        for _ in range(3):
            model += [SN_ResidualBlock(256)]
        model += [SN_Conv(in_ch=256, out_ch=64, K=3, S=1, P=1)]
        model += [SN_Conv(in_ch=64, out_ch=1, K=3, S=1,
                          P=1, activation=nn.Sigmoid())]

        self.model = nn.Sequential(*model)
        self.Avg = nn.AvgPool2d(kernel_size=64)

    def forward(self, x):
        x = self.model(x)
        return self.Avg(x).view(x.size()[0], -1)


class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features

# class deblur_generator(nn.Module):
#     def __init__(self, input_nc=2):
#         super(deblur_generator, self).__init__()
#
#         # Down sampling
#         self.model_1 = nn.Sequential(Conv(in_ch=input_nc, out_ch=64, K=9, P=4, activation=nn.LeakyReLU()))
#         model_2 = []
#         for _ in range(16):
#             model_2 += [ResidualBlock(64, activation=nn.LeakyReLU())]
#         self.model_2 = nn.Sequential(*model_2)
#         # Upsampling
#         model_3=[]
#         for _ in range(2):
#             model_3 += [
#                 nn.Conv2d(64, 256, 3, 1, 1),
#                 nn.BatchNorm2d(256),
#                 nn.PixelShuffle(upscale_factor=2),
#                 nn.PReLU(),]
#         self.model_3 = nn.Sequential(*model_3)
#         model_4 = [Conv(in_ch=64, out_ch=32, K=3, S=2, P=1, activation=nn.LeakyReLU())]
#         model_4 += [Conv(in_ch=32, out_ch=2, K=3, S=2, P=1, activation=nn.Tanh())]
#         self.model_4 = nn.Sequential(*model_4)
#
#
#     def forward(self, x):
#         out1 = self.model_1(x)
#         out2 = self.model_2(out1)
#         out = torch.add(out1, out2)
#         out = self.model_3(out)
#         out = self.model_4(out)
#         return out
#
# class deblur_discriminator(nn.Module):
#     def __init__(self, input_nc=2):
#         super(deblur_discriminator, self).__init__()
#
#         model = [SN_Conv(in_ch=input_nc, out_ch=32, K=3, S=2, P=1)]
#         model += [SN_Conv(in_ch=32, out_ch=64, K=3, S=2, P=1)]
#         model += [SN_Conv(in_ch=64, out_ch=128, K=3, S=1, P=1)]
#         for _ in range(6):
#             model += [SN_ResidualBlock(128)]
#         model += [SN_Conv(in_ch=128, out_ch=64, K=3, S=1, P=1)]
#         model += [SN_Conv(in_ch=64, out_ch=1, K=3, S=1, P=1, activation=nn.Sigmoid())]
#
#         self.model = nn.Sequential(*model)
#         self.Avg = nn.AvgPool2d(kernel_size=64)
#     def forward(self, x):
#         x = self.model(x)
#         return self.Avg(x).view(x.size()[0], -1)
#         # return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class coarse_tumor_generator(nn.Module):
    def __init__(self, input_nc=2):
        super(coarse_tumor_generator, self).__init__()

        # Down sampling
        model = [Conv(in_ch=input_nc, out_ch=32, K=5, S=1,
                      P=2, activation=nn.LeakyReLU())]
        model += [Conv(in_ch=32, out_ch=64, K=3, S=2,
                       P=1, activation=nn.LeakyReLU())]
        model += [Conv(in_ch=64, out_ch=128, K=3, S=2,
                       P=1, activation=nn.LeakyReLU())]
        model += [Conv(in_ch=128, out_ch=256, K=3, S=2,
                       P=1, activation=nn.LeakyReLU())]

        for _ in range(3):
            model += [ResidualBlock(256, activation=nn.LeakyReLU())]

        # Upsampling
        model += [nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(inplace=True)]
        model += [nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(64),
                  nn.LeakyReLU(inplace=True)]
        model += [nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(32),
                  nn.LeakyReLU(inplace=True)]

        model += [Conv(in_ch=32, out_ch=1, K=5, S=1,
                       P=2, activation=nn.LeakyReLU())]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x


class coarse_tumor_discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(coarse_tumor_discriminator, self).__init__()

        model = [Conv(in_ch=input_nc, out_ch=32, K=5, S=2, P=2)]
        model += [Conv(in_ch=32, out_ch=64, K=3, S=2, P=1)]
        model += [Conv(in_ch=64, out_ch=128, K=3, S=1, P=1)]
        for _ in range(3):
            model += [ResidualBlock(128, activation=nn.LeakyReLU())]
        model += [Conv(in_ch=128, out_ch=64, K=3, S=1, P=1)]
        model += [Conv(in_ch=64, out_ch=1, K=3, S=1,
                       P=1, activation=nn.Sigmoid())]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class refine_tumor_generator(nn.Module):
    def __init__(self, input_nc=1):
        super(refine_tumor_generator, self).__init__()
        # Down sampling
        model = [Conv(in_ch=input_nc, out_ch=32, K=5, S=1,
                      P=2, activation=nn.LeakyReLU())]
        model += [Conv(in_ch=32, out_ch=64, K=3, S=2,
                       P=1, activation=nn.LeakyReLU())]
        model += [Conv(in_ch=64, out_ch=128, K=3, S=2,
                       P=1, activation=nn.LeakyReLU())]
        model += [Conv(in_ch=128, out_ch=256, K=3, S=2,
                       P=1, activation=nn.LeakyReLU())]
        for _ in range(3):
            model += [SN_ResidualBlock(256, activation=nn.LeakyReLU())]
        # Upsampling
        model += [nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(inplace=True)]
        model += [nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(64),
                  nn.LeakyReLU(inplace=True)]
        model += [nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(32),
                  nn.LeakyReLU(inplace=True)]
        model += [Conv(in_ch=32, out_ch=1, K=5, S=1,
                       P=2, activation=nn.LeakyReLU())]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x


class refine_tumor_discriminator(nn.Module):
    def __init__(self, input_nc=2):
        super(refine_tumor_discriminator, self).__init__()
        model = [Conv(in_ch=input_nc, out_ch=32, K=5, S=2, P=2)]
        model += [Conv(in_ch=32, out_ch=64, K=3, S=2, P=1)]
        model += [Conv(in_ch=64, out_ch=128, K=3, S=1, P=1)]
        for _ in range(3):
            model += [ResidualBlock(128, activation=nn.LeakyReLU())]
        model += [Conv(in_ch=128, out_ch=64, K=3, S=1, P=1)]
        model += [Conv(in_ch=64, out_ch=1, K=3, S=1,
                       P=1, activation=nn.Sigmoid())]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)
