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
        # self.conv1 = BN_Conv2d(in_channels, 32, 3, 1, 1)
        # self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(32)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = out + x
        return F.relu(out)


class GomokuNet(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.learning_rate = conf['learning_rate']
        self.momentum = conf['momentum']
        self.l2 = conf['l2']
        self.batch_size = conf['batch_size']
        self.load_path = conf['path']
        self.start_version = conf['version']
        
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

        if self.start_version != -1 and self.load_path is not None:
            self.load_state_dict(torch.load(self.load_path + f"/version_{self.start_version}.model", map_location=torch.device('cpu')))

        self.to(conf['device'])

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
        device = torch.device('cpu')
        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        layer1 = torch.tensor(board == color, dtype=torch.float).reshape(15, 15)
        layer2 = torch.tensor(board == -color, dtype=torch.float).reshape(15, 15)
        layer3 = torch.zeros((15, 15))
        if last_move is not None:
            layer3[last_move // 15, last_move % 15] = 1
        net_input = torch.stack([layer1, layer2, layer3], dim=0).to(device)
        with torch.no_grad():
            policy, value = self(net_input)
            if device.type != 'cpu':
                policy = policy.cpu()
                value = value.cpu()
            return policy[0,:].numpy(), value[0,0].numpy()


def train_GomokuNet(model, optimizer, train_data, epochs, device, print_every=100):
    model = model.to(device=device)
    loss_sum = 0
    data_size = train_data.size()    # train_data is Dataset object
    for e in range(epochs):
        t = 0
        data_used = 0
        iter_idx = [i for i in range(data_size)]
        np.random.shuffle(iter_idx)
        board_data = np.array(train_data.board)
        last_move_data = np.array(train_data.last_move)
        p_data = np.array(train_data.p)
        z_data = np.array(train_data.z)
        while data_used < data_size - 1:    
            data_num = min(data_size - data_used, model.batch_size)
            board = board_data[iter_idx[data_used : data_used + data_num]]
            last_move = last_move_data[iter_idx[data_used : data_used + data_num]]
            p = torch.Tensor(p_data[iter_idx[data_used : data_used + data_num]])
            z = torch.Tensor(z_data[iter_idx[data_used : data_used + data_num]])
            net_input = get_input_tensor(board, last_move)
            
            net_input = net_input.to(device=device, dtype=torch.float32)
            p = p.to(device=device, dtype=torch.float32)
            z = z.to(device=device, dtype=torch.float32)
            
            policy, value = model(net_input)
            loss = torch.mean((value - z) ** 2) - torch.mean(torch.sum(p * torch.log(policy + 1e-10), dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % print_every == 0:
                print(f'Epoch {e}, Iteration {t}, loss = {loss.item()}')

            t += 1
            data_used += data_num
            loss_sum += loss.item()
    return loss_sum / epochs


def get_input_tensor(board, last_move):
    res = []
    for tempboard, tempmove in zip(board, last_move):
        tempinput = []
        # 1 represent now player
        tempinput.append(np.array(np.array(tempboard) == 1, dtype=np.float).reshape(15, 15))
        tempinput.append(np.array(np.array(tempboard) == -1, dtype=np.float).reshape(15, 15))
        templ = np.zeros((15, 15))
        if tempmove is not None:
            templ[tempmove // 15, tempmove % 15] = 1
        tempinput.append(templ)
        res.append(tempinput)
    return torch.from_numpy(np.array(res, dtype=np.float))