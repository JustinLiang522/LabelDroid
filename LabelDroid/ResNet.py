"""
modified from LabelDroid
"""
import torch
import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-101 and replace top fc layer."""
        super(ResNet, self).__init__()
        self.att_size = 7
        self.embed_size = 4096

        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool7x7 = nn.AdaptiveAvgPool2d((self.att_size, self.att_size))

    def forward(self, images):
        with torch.no_grad():
            x = self.resnet(images)

        att = self.adaptive_pool7x7(x).squeeze()
        if images.size(0) == 1:
            att = att.unsqueeze(0)
        att = att.permute(0, 2, 3, 1)
        att = att.view(images.size(0), -1, att.size(-1))

        return att


