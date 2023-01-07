import os
import sys
import numpy as np
# from tqdm import tqdm
import torch
import torch.utils.data as Data


def get_input_tensor(board, last_move, color):
    input1 = torch.tensor(board == color, dtype=torch.float).reshape(15, 15)
    input2 = torch.tensor(board == -color, dtype=torch.float).reshape(15, 15)
    input3 = torch.zeros((15, 15))
    if last_move is not None and last_move != -1:
        input3[last_move // 15, last_move % 15] = 1
    input_tensor = torch.stack([input1, input2, input3], dim=0)
    return input_tensor


class Dataset(Data.Dataset):
    def __init__(self, path, mode):
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        if mode == "read":
            self.boards = torch.load(path + "/board_record")
            self.last_moves = torch.load(path + "/last_move_record")
            self.ps = torch.load(path + "/p_record")
            self.zs = torch.load(path + "/z_record")
            self.colors = torch.load(path + "/color_record")
        elif mode == "write":
            self.boards, self.last_moves, self.ps, self.zs, self.colors = [], [], [], [], []
        else:
            raise ValueError("mode must be read or write")

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, index):
        return get_input_tensor(self.boards[index], self.last_moves[index], self.colors[index]), self.ps[index], torch.tensor([self.zs[index]])

    def append(self, board, last_move, p, z, color):
        self.boards.append(board)
        self.last_moves.append(last_move)
        self.ps.append(p)
        self.zs.append(z)
        self.colors.append(color)

    def extend(self, boards, last_moves, ps, zs, colors):
        self.boards.extend(boards)
        self.last_moves.extend(last_moves)
        self.ps.extend(ps)
        self.zs.extend(zs)
        self.colors.extend(colors)

    def save(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        torch.save(self.boards, self.path + "/board_record")
        torch.save(self.last_moves, self.path + "/last_move_record")
        torch.save(self.ps, self.path + "/p_record")
        torch.save(self.zs, self.path + "/z_record")
        torch.save(self.colors, self.path + "/color_record")

    def get_loader(self, batch_size):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=True)

    def augment_data(self, i):
        board, last_move, p, z, color = self.boards[i], self.last_moves[i], self.ps[i], self.zs[i], self.colors[i]
        # 3 rotation, 3 flip
        augment_mode = np.random.randint(0, 6)
        x, y = last_move // 15, last_move % 15
        if augment_mode < 3:
            while augment_mode >= 0:
                board = np.rot90(board.reshape(15, 15)).reshape(225)
                last_move = (7 + (7 - y)) * 15 + (7 + (x - 7))
                p = np.rot90(p.reshape(15, 15)).reshape(225)
                augment_mode -= 1
        else:
            augment_mode -= 3
            while augment_mode >= 0:
                if augment_mode % 2 == 0:
                    board = np.flipud(board.reshape(15, 15)).reshape(225)
                    last_move = (7 + (7 - x)) * 15 + y
                    p = np.flipud(p.reshape(15, 15)).reshape(225)
                else:
                    board = np.fliplr(board.reshape(15, 15)).reshape(225)
                    last_move = x * 15 + (7 + (7 - y))
                    p = np.fliplr(p.reshape(15, 15)).reshape(225)
                augment_mode -= 1
        return board, last_move, p, z, color

    def add_augment_data(self, augment_data_pro):
        sample_index = np.random.randint(0, len(self.boards), int(len(self.boards) * augment_data_pro))
        print("Begin to add augment data......")
        for index in range(len(sample_index)):
        # for index in tqdm(range(len(sample_index)), file=sys.stdout):
            i = sample_index[index]
            board_aug, last_move_aug, p_aug, z_aug, color = self.augment_data(i)
            self.append(board_aug, last_move_aug, p_aug, z_aug, color)