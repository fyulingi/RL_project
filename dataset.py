import torch


class Dataset:
    def __init__(self):
        self.board = []
        self.last_move = []
        self.p = []
        self.z = []

    def size(self):
        return len(self.board)

    def collect_data(self, path):
        self.board.extend(torch.load(path + '/board_record.hyt'))
        self.last_move.extend(torch.load(path + '/last_move_record.hyt'))
        self.p.extend(torch.load(path + '/p_record.hyt'))
        self.z.extend(torch.load(path + '/z_record.hyt'))
