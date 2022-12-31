import sys
from tqdm import tqdm
import numpy as np
import torch


class Generator:
    def __init__(self, max_noise_stone_num):
        self.max_noise_stone_num = max_noise_stone_num
        self.board_record, self.last_move_record, self.p_record, self.z_record = [], [], [], []

    def _generate_consecutive_line(self, consecutive_num):
        start_pos = np.random.randint(0, 15, 2)
        end_pos = [-1, -1]
        while end_pos[0] < 0 or end_pos[0] > 14 or end_pos[1] < 0 or end_pos[1] > 14:
            dx, dy = list(np.random.randint(-1, 2, 2))
            if dx == 0 and dy == 0:
                continue
            end_pos[0] = start_pos[0] + (consecutive_num - 1) * dx
            end_pos[1] = start_pos[1] + (consecutive_num - 1) * dy
        fix_pos_list = []
        if dx == 0:
            x_list = [start_pos[0]] * consecutive_num
        else:
            x_list = list(range(start_pos[0], end_pos[0] + dx, dx))
        if dy == 0:
            y_list = [start_pos[1]] * consecutive_num
        else:
            y_list = list(range(start_pos[1], end_pos[1] + dy, dy))

        fp_1 = [start_pos[0] - dx, start_pos[1] - dy]
        if fp_1[0] in list(range(0, 15)) and fp_1[1] in list(range(0, 15)):
            fix_pos_list.append(fp_1)
        fp_2 = [end_pos[0] + dx, end_pos[1] + dy]
        if fp_2[0] in list(range(0, 15)) and fp_2[1] in list(range(0, 15)):
            fix_pos_list.append(fp_2)

        pos_list = list(zip(x_list, y_list))
        return pos_list, fix_pos_list

    def fill_board_pos_list_with_color(self, board, pos_list, color):
        for pos in pos_list:
            board[15 * pos[0] + pos[1]] = color

    def add_noise(self, board, fix_pos_list):
        # black_stone_num + black_add_num = white_stone_num + white_add_num + difference_num
        difference_num = np.random.randint(0, 2)
        black_stone_ind = np.where(board == 1)
        white_stone_ind = np.where(board == -1)
        black_stone_num = len(black_stone_ind[0])
        white_stone_num = len(white_stone_ind[0])
        max_stone_num = max(black_stone_num, white_stone_num)
        white_add_num = np.random.randint(0, self.max_noise_stone_num+1) + (max_stone_num - white_stone_num)
        black_add_num = white_add_num + difference_num + (max_stone_num - black_stone_num)

        while black_add_num > 0:
            x, y = np.random.randint(0, 15, 2)
            if board[15 * x + y] == 0 and [x, y] not in fix_pos_list:
                board[15 * x + y] = 1
                black_add_num -= 1

        while white_add_num > 0:
            x, y = np.random.randint(0, 15, 2)
            if board[15 * x + y] == 0 and [x, y] not in fix_pos_list:
                board[15 * x + y] = -1
                white_add_num -= 1

    def sel_last_move(self, board, pos_list, type):
        if type == "attack":
            oppo_pos_list = np.argwhere(board == -1)
            index = np.random.randint(0, len(oppo_pos_list))
            last_move = oppo_pos_list[index][0]
        elif type == "defend":
            index = np.random.randint(0, len(pos_list))
            last_move = 15 * pos_list[index][0] + pos_list[index][1]
        else:
            raise ValueError("type must be attack or defend")
        return last_move

    def insert_record(self, board, last_move, p, z):
        self.board_record.append(board)
        self.last_move_record.append(last_move)
        self.p_record.append(p)
        self.z_record.append(z)

    def generate_3_ooo_attack(self, sample_num):
        for _ in tqdm(range(sample_num), file=sys.stdout):
            board = np.zeros(225)
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=3)
            if len(fix_pos_list) != 2:
                continue
            self.fill_board_pos_list_with_color(board, pos_list, 1)
            pi = np.array([0.0 for _ in range(225)])
            pi[15*fix_pos_list[0][0]+fix_pos_list[0][1]] = pi[15*fix_pos_list[1][0]+fix_pos_list[1][1]] = 0.5
            self.add_noise(board, fix_pos_list)
            last_move = self.sel_last_move(board, pos_list, "attack")
            self.insert_record(board, last_move, pi, 1)

    def generate_3_oo_o_attack(self, sample_num):
        for _ in tqdm(range(sample_num), file=sys.stdout):
            board = np.zeros(225)
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=4)
            if len(fix_pos_list) != 2:
                continue
            fix_pos_list.append(list(pos_list[2]))
            pos_list.pop(2)
            self.fill_board_pos_list_with_color(board, pos_list, 1)
            pi = np.array([0.0 for _ in range(225)])
            pi[15*fix_pos_list[-1][0]+fix_pos_list[-1][1]] = 1
            self.add_noise(board, fix_pos_list)
            last_move = self.sel_last_move(board, pos_list, "attack")
            self.insert_record(board, last_move, pi, 1)

    def gen_3_attack_data(self, sample_num):
        print("Begin to generate 3 oo_o attack data......")
        self.generate_3_oo_o_attack(sample_num)
        print("Begin to generate 3 ooo attack data......")
        self.generate_3_ooo_attack(sample_num)

    def generate_3_ooo_defend(self, sample_num):
        for _ in tqdm(range(sample_num), file=sys.stdout):
            board = np.zeros(225)
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=3)
            if len(fix_pos_list) != 2:
                continue
            self.fill_board_pos_list_with_color(board, pos_list, -1)
            pi = np.array([0.0 for _ in range(225)])
            pi[15*fix_pos_list[0][0]+fix_pos_list[0][1]] = pi[15*fix_pos_list[1][0]+fix_pos_list[1][1]] = 0.5
            self.add_noise(board, fix_pos_list)
            last_move = self.sel_last_move(board, pos_list, "defend")
            self.insert_record(board, last_move, pi, 0)

    def generate_3_oo_o_defend(self, sample_num):
        for _ in tqdm(range(sample_num), file=sys.stdout):
            board = np.zeros(225)
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=4)
            if len(fix_pos_list) != 2:
                continue
            fix_pos_list.append(list(pos_list[2]))
            pos_list.pop(2)
            self.fill_board_pos_list_with_color(board, pos_list, -1)
            pi = np.array([0.0 for _ in range(225)])
            ind_1 = 15 * fix_pos_list[0][0] + fix_pos_list[0][1]
            ind_2 = 15 * fix_pos_list[1][0] + fix_pos_list[1][1]
            ind_3 = 15 * fix_pos_list[2][0] + fix_pos_list[2][1]
            pi[ind_1], pi[ind_2], pi[ind_3] = 0.25, 0.25, 0.5
            self.add_noise(board, fix_pos_list)
            last_move = self.sel_last_move(board, pos_list, "defend")
            self.insert_record(board, last_move, pi, 0)

    def gen_3_defend_data(self, sample_num):
        print("Begin to generate 3 oo_o defend data......")
        self.generate_3_oo_o_defend(sample_num)
        print("Begin to generate 3 ooo defend data......")
        self.generate_3_ooo_defend(sample_num)

    def generate_4_oooo_attack(self, sample_num):
        for _ in tqdm(range(sample_num), file=sys.stdout):
            board = np.zeros(225)
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=4)
            if len(fix_pos_list) == 0:
                continue
            self.fill_board_pos_list_with_color(board, pos_list, 1)
            pi = np.array([0.0 for _ in range(225)])
            if len(fix_pos_list) == 2:
                pi[15*fix_pos_list[0][0]+fix_pos_list[0][1]] = pi[15*fix_pos_list[1][0]+fix_pos_list[1][1]] = 0.5
            elif len(fix_pos_list) == 1:
                pi[15*fix_pos_list[0][0]+fix_pos_list[0][1]] = 1
            self.add_noise(board, fix_pos_list)
            last_move = self.sel_last_move(board, pos_list, "attack")
            self.insert_record(board, last_move, pi, 1)

    def generate_4_ooo_o_attack(self, sample_num):
        for _ in tqdm(range(sample_num), file=sys.stdout):
            board = np.zeros(225)
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=5)
            fix_pos_list.append(list(pos_list[3]))
            pos_list.pop(3)
            self.fill_board_pos_list_with_color(board, pos_list, 1)
            pi = np.array([0.0 for _ in range(225)])
            pi[15*fix_pos_list[-1][0]+fix_pos_list[-1][1]] = 1
            self.add_noise(board, fix_pos_list)
            last_move = self.sel_last_move(board, pos_list, "attack")
            self.insert_record(board, last_move, pi, 1)

    def generate_4_oo_oo_attack(self, sample_num):
        for _ in tqdm(range(sample_num), file=sys.stdout):
            board = np.zeros(225)
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=5)
            fix_pos_list.append(list(pos_list[2]))
            pos_list.pop(2)
            self.fill_board_pos_list_with_color(board, pos_list, 1)
            pi = np.array([0.0 for _ in range(225)])
            pi[15*fix_pos_list[-1][0]+fix_pos_list[-1][1]] = 1
            self.add_noise(board, fix_pos_list)
            last_move = self.sel_last_move(board, pos_list, "attack")
            self.insert_record(board, last_move, pi, 1)

    def gen_4_attack_data(self, sample_num):
        print("Begin to generate 4 oooo attack data......")
        self.generate_4_oooo_attack(sample_num)
        print("Begin to generate 4 oo_oo attack data......")
        self.generate_4_oo_oo_attack(sample_num)
        print("Begin to generate 4 ooo_o attack data......")
        self.generate_4_ooo_o_attack(sample_num)

    def generate_4_oooo_defend(self, sample_num):
        for _ in tqdm(range(sample_num), file=sys.stdout):
            board = np.zeros(225)
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=4)
            if len(fix_pos_list) != 2:
                continue
            self.fill_board_pos_list_with_color(board, pos_list, -1)
            board[15 * fix_pos_list[0][0] + fix_pos_list[0][1]] = 1
            pi = np.array([0.0 for _ in range(225)])
            pi[15*fix_pos_list[1][0]+fix_pos_list[1][1]] = 0.5
            self.add_noise(board, fix_pos_list)
            last_move = self.sel_last_move(board, pos_list, "defend")
            self.insert_record(board, last_move, pi, 0)

    def generate_4_ooo_o_defend(self, sample_num):
        for _ in tqdm(range(sample_num), file=sys.stdout):
            board = np.zeros(225)
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=5)
            fix_pos_list.append(list(pos_list[3]))
            pos_list.pop(3)
            self.fill_board_pos_list_with_color(board, pos_list, -1)
            pi = np.array([0.0 for _ in range(225)])
            pi[15*fix_pos_list[-1][0]+fix_pos_list[-1][1]] = 1
            self.add_noise(board, fix_pos_list)
            last_move = self.sel_last_move(board, pos_list, "defend")
            self.insert_record(board, last_move, pi, 0)

    def generate_4_oo_oo_defend(self, sample_num):
        for _ in tqdm(range(sample_num), file=sys.stdout):
            board = np.zeros(225)
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=5)
            fix_pos_list.append(list(pos_list[2]))
            pos_list.pop(2)
            self.fill_board_pos_list_with_color(board, pos_list, -1)
            pi = np.array([0.0 for _ in range(225)])
            pi[15*fix_pos_list[-1][0]+fix_pos_list[-1][1]] = 1
            self.add_noise(board, fix_pos_list)
            last_move = self.sel_last_move(board, pos_list, "defend")
            self.insert_record(board, last_move, pi, 0)

    def gen_4_defend_data(self, sample_num):
        print("Begin to generate 4 oooo defend data......")
        self.generate_4_oooo_defend(sample_num)
        print("Begin to generate 4 oo_oo defend data......")
        self.generate_4_oo_oo_defend(sample_num)
        print("Begin to generate 4 ooo_o defend data......")
        self.generate_4_ooo_o_defend(sample_num)

    def augment_data(self, board, last_move, p, z):
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
                if augment_mode%2 == 0:
                    board = np.flipud(board.reshape(15, 15)).reshape(225)
                    last_move = (7 + (7 - x)) * 15 + y
                    p = np.flipud(p.reshape(15, 15)).reshape(225)
                else:
                    board = np.fliplr(board.reshape(15, 15)).reshape(225)
                    last_move = x * 15 + (7 + (7 - y))
                    p = np.fliplr(p.reshape(15, 15)).reshape(225)
                augment_mode -= 1
        return board, last_move, p, z

    def add_augment_data(self, augment_data_pro):
        sample_index = np.random.randint(0, len(self.board_record), int(len(self.board_record) * augment_data_pro))
        print("Begin to add augment data......")
        for index in tqdm(range(len(sample_index)), file=sys.stdout):
            i = sample_index[index]
            board = self.board_record[i]
            last_move = self.last_move_record[i]
            p = self.p_record[i]
            z = self.z_record[i]
            board_aug, last_move_aug, p_aug, z_aug = self.augment_data(board, last_move, p, z)
            self.insert_record(board_aug, last_move_aug, p_aug, z_aug)

    def generate_train_data(self, sample_num, augment_data_pro):
        self.gen_3_defend_data(sample_num)
        self.gen_3_attack_data(sample_num)
        self.gen_4_defend_data(sample_num)
        self.gen_4_attack_data(sample_num)
        self.add_augment_data(augment_data_pro)

    def save_train_data(self, path):
        torch.save(np.array(self.board_record), path + '/board_record')
        torch.save(np.array(self.last_move_record), path + '/last_move_record')
        torch.save(np.array(self.p_record), path + '/p_record')
        torch.save(np.array(self.z_record), path + '/z_record')


if __name__ == '__main__':
    data_generator = Generator(2)
    data_generator.generate_train_data(5000, 0.2)
    data_generator.save_train_data('./gamedata/enhanced')