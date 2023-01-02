from dataset import get_input_tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BN_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=True):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels))
    
    def forward(self, x):
        return F.relu(self.seq(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(32)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = out + x
        return F.relu(out)


class GomokuNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.learning_rate = conf['learning_rate']
        # self.momentum = conf['momentum']
        # self.l2 = conf['l2']
        # self.batch_size = conf['batch_size']
        super().__init__()
        self.load_path = config['path']
        self.version = config['version']
        self.device = config['device']
        
        self.conv1 = BN_Conv2d(3, 32, 3, 1, 1)
        self.conv2 = BN_Conv2d(32, 32, 3, 1, 1)
        self.res1 = ResidualBlock(32)
        self.conv3 = BN_Conv2d(32, 32, 3, 1, 1)
        self.res2 = ResidualBlock(32)

        self.policyhead5 = BN_Conv2d(32, 2, 1, 1, 0)
        self.policyhead_fc6 = nn.Linear(450, 225)

        self.valuehead5 = BN_Conv2d(32, 1, 1, 1, 0)
        self.valuehead_fc6 = nn.Linear(225, 32)
        self.valuehead_fc7 = nn.Linear(32, 1)

        if self.version != 0 and self.load_path is not None:
            self.load_state_dict(torch.load(self.load_path + f"/model_{self.version}", map_location=torch.device('cpu')))

        self.to(self.device)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out)
        out = self.conv3(out)
        out = self.res2(out)

        policy = self.policyhead5(out)
        policy = policy.view(policy.size(0), -1)
        policy = self.policyhead_fc6(policy)
        policy = F.softmax(policy, dim=1)

        value = self.valuehead5(out)
        value = value.view(value.size(0), -1)
        value = self.valuehead_fc6(value)
        value = F.relu(value)
        value = self.valuehead_fc7(value)
        value = torch.tanh(value)

        return policy, value

    def predict(self, board, last_move, color):
        net_input = get_input_tensor(board, last_move, color).unsqueeze(0).to(self.device)
        return self(net_input)
