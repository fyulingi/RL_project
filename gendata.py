import numpy as np
import torch

class Generator:
    def __init__(self, max_noise_stone_num):
        self._max_noise_stone_num = max_noise_stone_num

    def generate_dead_4_oooo_defend(self, sample_num=10000):
        board_record_black, last_move_record_black, p_record_black, z_record_black = [], [], [], []
        board_record_white, last_move_record_white, p_record_white, z_record_white = [], [], [], []
        i = 0
        while i < sample_num:
            color = np.random.randint(0, 2) * 2 - 1
            board = np.zeros(225)
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=4)
            if len(fix_pos_list) == 0:
                continue

            for x, y in pos_list:
                board[15 * x + y] = color

            pi = np.array([0.0 for _ in range(225)])
            if len(fix_pos_list) == 2:
                ind = 15 * fix_pos_list[0][0] + fix_pos_list[0][1]
                pi[ind] = 1
                fx, fy = fix_pos_list[1][0], fix_pos_list[1][1]
                board[15 * fx + fy] = -color
            if len(fix_pos_list) == 1:
                ind = 15 * fix_pos_list[0][0] + fix_pos_list[0][1]
                pi[ind] = 1

            self._add_noise(board=board, next_player=-color, max_stone_num=self._max_noise_stone_num,
                            fix_pos_list=fix_pos_list)
            if color == -1:
                board_record_black.append(board)
                last_move_record_black.append(15 * pos_list[0][0] + pos_list[0][1])
                p_record_black.append(pi)
                z_record_black.append(0)
            else:
                board_record_white.append(board)
                last_move_record_white.append(15 * pos_list[0][0] + pos_list[0][1])
                p_record_white.append(pi)
                z_record_white.append(0)
            i += 1
        return board_record_black, last_move_record_black, p_record_black, z_record_black, board_record_white, last_move_record_white, p_record_white, z_record_white

    def generate_dead_4_ooo_o_defend(self, sample_num=10000):
        board_record_black, last_move_record_black, p_record_black, z_record_black = [], [], [], []
        board_record_white, last_move_record_white, p_record_white, z_record_white = [], [], [], []
        for _ in range(sample_num):
            color = np.random.randint(0, 2) * 2 - 1
            board = np.zeros(225)
            pos_list, _ = self._generate_consecutive_line(consecutive_num=5)
            fix_pos_list = [pos_list[3]]

            for x, y in pos_list:
                board[15 * x + y] = color
            board[15 * pos_list[3][0] + pos_list[3][1]] = 0

            pi = np.array([0.0 for _ in range(225)])

            ind = 15 * pos_list[3][0] + pos_list[3][1]
            pi[ind] = 1

            self._add_noise(board=board, next_player=-color, max_stone_num=self._max_noise_stone_num,
                            fix_pos_list=fix_pos_list)
            if color == -1:
                board_record_black.append(board)
                last_move_record_black.append(15 * pos_list[1][0] + pos_list[1][1])
                p_record_black.append(pi)
                z_record_black.append(0)
            else:
                board_record_white.append(board)
                last_move_record_white.append(15 * pos_list[1][0] + pos_list[1][1])
                p_record_white.append(pi)
                z_record_white.append(0)
        return board_record_black, last_move_record_black, p_record_black, z_record_black, board_record_white, last_move_record_white, p_record_white, z_record_white

    def generate_dead_4_oo_oo_defend(self, sample_num=10000):
        board_record_black, last_move_record_black, p_record_black, z_record_black = [], [], [], []
        board_record_white, last_move_record_white, p_record_white, z_record_white = [], [], [], []
        for _ in range(sample_num):
            color = np.random.randint(0, 2) * 2 - 1
            board = np.zeros(225)
            pos_list, _ = self._generate_consecutive_line(consecutive_num=5)
            fix_pos_list = [pos_list[2]]

            for x, y in pos_list:
                board[15 * x + y] = color
            board[15 * pos_list[2][0] + pos_list[2][1]] = 0

            pi = np.array([0.0 for _ in range(225)])

            ind = 15 * pos_list[2][0] + pos_list[2][1]
            pi[ind] = 1

            self._add_noise(board=board, next_player=-color, max_stone_num=self._max_noise_stone_num,
                            fix_pos_list=fix_pos_list)
            if color == -1:
                board_record_black.append(board)
                last_move_record_black.append(15 * pos_list[1][0] + pos_list[1][1])
                p_record_black.append(pi)
                z_record_black.append(0)
            else:
                board_record_white.append(board)
                last_move_record_white.append(15 * pos_list[1][0] + pos_list[1][1])
                p_record_white.append(pi)
                z_record_white.append(0)

        return board_record_black, last_move_record_black, p_record_black, z_record_black, board_record_white, last_move_record_white, p_record_white, z_record_white

    def generate_live_3_ooo_attack(self, sample_num=10000):
        board_record_black, last_move_record_black, p_record_black, z_record_black = [], [], [], []
        board_record_white, last_move_record_white, p_record_white, z_record_white = [], [], [], []
        i = 0
        while i < sample_num:
            color = np.random.randint(0, 2) * 2 - 1
            board = np.zeros(225)
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=3)
            if len(fix_pos_list) == 0 or len(fix_pos_list) == 1:
                continue

            for x, y in pos_list:
                board[15 * x + y] = color

            pi = np.array([0.0 for _ in range(225)])
            ind_1 = 15 * fix_pos_list[0][0] + fix_pos_list[0][1]
            ind_2 = 15 * fix_pos_list[1][0] + fix_pos_list[1][1]
            pi[ind_1], pi[ind_2] = 0.5, 0.5

            self._add_noise(board=board, next_player=color, max_stone_num=self._max_noise_stone_num,
                            fix_pos_list=fix_pos_list)

            if color == 1:
                board_record_black.append(board)
                last_move_record_black.append(15 * pos_list[1][0] + pos_list[1][1])
                p_record_black.append(pi)
                z_record_black.append(1)
            else:
                board_record_white.append(board)
                last_move_record_white.append(15 * pos_list[1][0] + pos_list[1][1])
                p_record_white.append(pi)
                z_record_white.append(1)
            i += 1
        return board_record_black, last_move_record_black, p_record_black, z_record_black, board_record_white, last_move_record_white, p_record_white, z_record_white

    def generate_live_3_oo_o_attack(self, sample_num=10000):
        board_record_black, last_move_record_black, p_record_black, z_record_black = [], [], [], []
        board_record_white, last_move_record_white, p_record_white, z_record_white = [], [], [], []
        i = 0
        while i < sample_num:
            color = np.random.randint(0, 2) * 2 - 1
            board = np.zeros(225)
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=4)
            if len(fix_pos_list) == 0 or len(fix_pos_list) == 1:
                continue

            fix_pos_list.append(list(pos_list[2]))

            for x, y in pos_list:
                board[15 * x + y] = color
            board[15 * pos_list[2][0] + pos_list[2][1]] = 0

            pi = np.array([0.0 for _ in range(225)])
            ind = 15 * pos_list[2][0] + pos_list[2][1]
            pi[ind] = 1

            self._add_noise(board=board, next_player=color, max_stone_num=self._max_noise_stone_num,
                            fix_pos_list=fix_pos_list)
            if color == 1:
                board_record_black.append(board)
                last_move_record_black.append(15 * pos_list[1][0] + pos_list[1][1])
                p_record_black.append(pi)
                z_record_black.append(1)
            else:
                board_record_white.append(board)
                last_move_record_white.append(15 * pos_list[1][0] + pos_list[1][1])
                p_record_white.append(pi)
                z_record_white.append(1)

            i += 1
        return board_record_black, last_move_record_black, p_record_black, z_record_black, board_record_white, last_move_record_white, p_record_white, z_record_white

    def generate_live_3_ooo_defend(self, sample_num=10000):
        board_record_black, last_move_record_black, p_record_black, z_record_black = [], [], [], []
        board_record_white, last_move_record_white, p_record_white, z_record_white = [], [], [], []
        i = 0
        while i < sample_num:
            if i%1000 == 0:
                print("generate_live_3_ooo_defend: ", i)
            color = np.random.randint(0, 2) * 2 - 1
            board = np.zeros(225)
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=3)
            if len(fix_pos_list) == 0 or len(fix_pos_list) == 1:
                continue

            for x, y in pos_list:
                board[15 * x + y] = color

            pi = np.array([0.0 for _ in range(225)])
            ind_1 = 15 * fix_pos_list[0][0] + fix_pos_list[0][1]
            ind_2 = 15 * fix_pos_list[1][0] + fix_pos_list[1][1]
            pi[ind_1], pi[ind_2] = 0.5, 0.5
            self._add_noise(board=board, next_player=-color, max_stone_num=self._max_noise_stone_num,
                            fix_pos_list=fix_pos_list)
            if color == -1:
                board_record_black.append(board)
                last_move_record_black.append(15 * pos_list[1][0] + pos_list[1][1])
                p_record_black.append(pi)
                z_record_black.append(0)
            else:
                board_record_white.append(board)
                last_move_record_white.append(15 * pos_list[1][0] + pos_list[1][1])
                p_record_white.append(pi)
                z_record_white.append(0)
            i += 1
        return board_record_black, last_move_record_black, p_record_black, z_record_black, board_record_white, last_move_record_white, p_record_white, z_record_white

    def generate_live_3_oo_o_defend(self, sample_num=10000):
        board_record_black, last_move_record_black, p_record_black, z_record_black = [], [], [], []
        board_record_white, last_move_record_white, p_record_white, z_record_white = [], [], [], []
        i = 0
        while i < sample_num:
            if i%1000 == 0:
                print("generate_live_3_oo_o_defend: ", i)
            color = np.random.randint(0, 2) * 2 - 1
            board = np.zeros(225)
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=4)
            if len(fix_pos_list) == 0 or len(fix_pos_list) == 1:
                continue

            fix_pos_list.append(list(pos_list[2]))

            for x, y in pos_list:
                board[15 * x + y] = color
            board[15 * pos_list[2][0] + pos_list[2][1]] = 0

            pi = np.array([0.0 for _ in range(225)])
            ind_1 = 15 * fix_pos_list[0][0] + fix_pos_list[0][1]
            ind_2 = 15 * fix_pos_list[1][0] + fix_pos_list[1][1]
            ind_3 = 15 * fix_pos_list[2][0] + fix_pos_list[2][1]
            pi[ind_1], pi[ind_2], pi[ind_3] = 0.25, 0.25, 0.5

            self._add_noise(board=board, next_player=-color, max_stone_num=self._max_noise_stone_num,
                            fix_pos_list=fix_pos_list)

            if color == -1:
                board_record_black.append(board)
                last_move_record_black.append(15 * pos_list[1][0] + pos_list[1][1])
                p_record_black.append(pi)
                z_record_black.append(0)
            else:
                board_record_white.append(board)
                last_move_record_white.append(15 * pos_list[1][0] + pos_list[1][1])
                p_record_white.append(pi)
                z_record_white.append(0)
            i += 1
        return board_record_black, last_move_record_black, p_record_black, z_record_black,\
               board_record_white, last_move_record_white, p_record_white, z_record_white

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

    def _add_noise(self, board, next_player, max_stone_num, fix_pos_list):
        stone_num = np.random.randint(0, max_stone_num+1)
        black_stone_ind = np.where(board == 1)
        white_stone_ind = np.where(board == -1)
        black_stone_num = len(black_stone_ind[0])
        white_stone_num = len(white_stone_ind[0])
        black_origin, white_origin = black_stone_num, white_stone_num

        delta = black_stone_num - white_stone_num
        # 假设下一步轮到黑棋走，要放x个黑棋，y个白棋，则x+b=y+w, x+y=stone_num
        # x-y=-delta, 2x=stone_num-delta
        # 假设下一步轮到白棋走，要放x个黑棋，y个白棋，则x+b+1=y+w, x+y=stone_num
        # x-y=-delta-1, 2x=stone_num-delta-1

        if next_player == 1:
            black_stone_num = int((stone_num - delta) / 2)
            white_stone_num = black_stone_num + delta
            if black_stone_num + black_origin > white_stone_num + white_origin:
                white_stone_num += 1
        else:
            black_stone_num = int((stone_num - delta - 1) / 2)
            white_stone_num = black_stone_num + delta
            if black_stone_num + black_origin == white_stone_num + white_origin:
                black_stone_num += 1

        while white_stone_num > 0:
            pos = list(np.random.randint(0, 15, 2))
            if board[15 * pos[0] + pos[1]] == 0 and pos not in fix_pos_list:
                white_stone_num -= 1
                board[15 * pos[0] + pos[1]] = -1

        while black_stone_num > 0:
            pos = list(np.random.randint(0, 15, 2))
            if board[15 * pos[0] + pos[1]] == 0 and pos not in fix_pos_list:
                black_stone_num -= 1
                board[15 * pos[0] + pos[1]] = 1

    def gen_and_save(self, path, sample_num=20000):
        board_record_black, last_move_record_black, p_record_black, z_record_black = [], [], [], []
        board_record_white, last_move_record_white, p_record_white, z_record_white = [], [], [], []
        record = self.generate_live_3_oo_o_attack(sample_num=sample_num)
        board_record_black.extend(record[0])
        last_move_record_black.extend(record[1])
        p_record_black.extend(record[2])
        z_record_black.extend(record[3])
        board_record_white.extend(record[4])
        last_move_record_white.extend(record[5])
        p_record_white.extend(record[6])
        z_record_white.extend(record[7])
        record = self.generate_live_3_oo_o_defend(sample_num=sample_num)
        board_record_black.extend(record[0])
        last_move_record_black.extend(record[1])
        p_record_black.extend(record[2])
        z_record_black.extend(record[3])
        board_record_white.extend(record[4])
        last_move_record_white.extend(record[5])
        p_record_white.extend(record[6])
        z_record_white.extend(record[7])
        record = self.generate_live_3_ooo_attack(sample_num=sample_num)
        board_record_black.extend(record[0])
        last_move_record_black.extend(record[1])
        p_record_black.extend(record[2])
        z_record_black.extend(record[3])
        board_record_white.extend(record[4])
        last_move_record_white.extend(record[5])
        p_record_white.extend(record[6])
        z_record_white.extend(record[7])
        record = self.generate_live_3_ooo_defend(sample_num=sample_num)
        board_record_black.extend(record[0])
        last_move_record_black.extend(record[1])
        p_record_black.extend(record[2])
        z_record_black.extend(record[3])
        board_record_white.extend(record[4])
        last_move_record_white.extend(record[5])
        p_record_white.extend(record[6])
        z_record_white.extend(record[7])
        record = self.generate_live_4_attack(sample_num=sample_num)
        board_record_black.extend(record[0])
        last_move_record_black.extend(record[1])
        p_record_black.extend(record[2])
        z_record_black.extend(record[3])
        board_record_white.extend(record[4])
        last_move_record_white.extend(record[5])
        p_record_white.extend(record[6])
        z_record_white.extend(record[7])
        record = self.generate_live_4_defend(sample_num=sample_num)
        board_record_black.extend(record[0])
        last_move_record_black.extend(record[1])
        p_record_black.extend(record[2])
        z_record_black.extend(record[3])
        board_record_white.extend(record[4])
        last_move_record_white.extend(record[5])
        p_record_white.extend(record[6])
        z_record_white.extend(record[7])
        record = self.generate_dead_4_oo_oo_defend(sample_num=sample_num)
        board_record_black.extend(record[0])
        last_move_record_black.extend(record[1])
        p_record_black.extend(record[2])
        z_record_black.extend(record[3])
        board_record_white.extend(record[4])
        last_move_record_white.extend(record[5])
        p_record_white.extend(record[6])
        z_record_white.extend(record[7])
        record = self.generate_dead_4_ooo_o_defend(sample_num=sample_num)
        board_record_black.extend(record[0])
        last_move_record_black.extend(record[1])
        p_record_black.extend(record[2])
        z_record_black.extend(record[3])
        board_record_white.extend(record[4])
        last_move_record_white.extend(record[5])
        p_record_white.extend(record[6])
        z_record_white.extend(record[7])
        torch.save(np.array(board_record_black), path + '/black/board_record.hyt')
        torch.save(np.array(last_move_record_black), path + '/black/last_move_record.hyt')
        torch.save(np.array(p_record_black), path + '/black/p_record.hyt')
        torch.save(np.array(z_record_black), path + '/black/z_record.hyt')
        torch.save(np.array(board_record_white), path + '/white/board_record.hyt')
        torch.save(np.array(last_move_record_white), path + '/white/last_move_record.hyt')
        torch.save(np.array(p_record_white), path + '/white/p_record.hyt')
        torch.save(np.array(z_record_white), path + '/white/z_record.hyt')

    def gen_live3_defend_data(self, black_data, white_data,  sample_num=5000):
        board_record_black, last_move_record_black, p_record_black, z_record_black = black_data[0], black_data[1], black_data[2], black_data[3]
        board_record_white, last_move_record_white, p_record_white, z_record_white = white_data[0], white_data[1], white_data[2], white_data[3]
        print("Begin to generate live 3 oo_o defend data......")
        record = self.generate_live_3_oo_o_defend(sample_num=sample_num)
        board_record_black.extend(record[0])
        last_move_record_black.extend(record[1])
        p_record_black.extend(record[2])
        z_record_black.extend(record[3])
        board_record_white.extend(record[4])
        last_move_record_white.extend(record[5])
        p_record_white.extend(record[6])
        z_record_white.extend(record[7])

        print("Begin to generate live 3 ooo defend data......")
        record = self.generate_live_3_ooo_defend(sample_num=sample_num)
        board_record_black.extend(record[0])
        last_move_record_black.extend(record[1])
        p_record_black.extend(record[2])
        z_record_black.extend(record[3])
        board_record_white.extend(record[4])
        last_move_record_white.extend(record[5])
        p_record_white.extend(record[6])
        z_record_white.extend(record[7])
        # torch.save(np.array(board_record_black), path + '/black/board_record.hyt')
        # torch.save(np.array(last_move_record_black), path + '/black/last_move_record.hyt')
        # torch.save(np.array(p_record_black), path + '/black/p_record.hyt')
        # torch.save(np.array(z_record_black), path + '/black/z_record.hyt')
        # torch.save(np.array(board_record_white), path + '/white/board_record.hyt')
        # torch.save(np.array(last_move_record_white), path + '/white/last_move_record.hyt')
        # torch.save(np.array(p_record_white), path + '/white/p_record.hyt')
        # torch.save(np.array(z_record_white), path + '/white/z_record.hyt')

    def gen_live3_attack_data(self, black_data, white_data,  sample_num=5000):
        board_record_black, last_move_record_black, p_record_black, z_record_black = black_data[0], black_data[1], black_data[2], black_data[3]
        board_record_white, last_move_record_white, p_record_white, z_record_white = white_data[0], white_data[1], white_data[2], white_data[3]
        print("Begin to generate live 3 oo_o defend data......")
        record = self.generate_live_3_oo_o_attack(sample_num=sample_num)
        board_record_black.extend(record[0])
        last_move_record_black.extend(record[1])
        p_record_black.extend(record[2])
        z_record_black.extend(record[3])
        board_record_white.extend(record[4])
        last_move_record_white.extend(record[5])
        p_record_white.extend(record[6])
        z_record_white.extend(record[7])

        print("Begin to generate live 3 ooo defend data......")
        record = self.generate_live_3_ooo_attack(sample_num=sample_num)
        board_record_black.extend(record[0])
        last_move_record_black.extend(record[1])
        p_record_black.extend(record[2])
        z_record_black.extend(record[3])
        board_record_white.extend(record[4])
        last_move_record_white.extend(record[5])
        p_record_white.extend(record[6])
        z_record_white.extend(record[7])


    def gen_live4_attack_data(self, black_data, white_data, sample_num=5000):
        board_record_black, last_move_record_black, p_record_black, z_record_black = black_data[0], black_data[1], black_data[2], black_data[3]
        board_record_white, last_move_record_white, p_record_white, z_record_white = white_data[0], white_data[1], white_data[2], white_data[3]
        i = 0
        while i < sample_num:
            color = np.random.randint(0, 2) * 2 - 1
            board = np.zeros(225)
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=4)
            if len(fix_pos_list) == 0:
                continue

            for x, y in pos_list:
                board[15 * x + y] = color

            pi = np.array([0.0 for _ in range(225)])
            if len(fix_pos_list) == 2:
                ind_1 = 15 * fix_pos_list[0][0] + fix_pos_list[0][1]
                ind_2 = 15 * fix_pos_list[1][0] + fix_pos_list[1][1]
                pi[ind_1], pi[ind_2] = 0.5, 0.5
            if len(fix_pos_list) == 1:
                ind = 15 * fix_pos_list[0][0] + fix_pos_list[0][1]
                pi[ind] = 1

            self._add_noise(board=board, next_player=color, max_stone_num=self._max_noise_stone_num,
                            fix_pos_list=fix_pos_list)

            if color == 1:
                board_record_black.append(board)
                last_move_record_black.append(15 * pos_list[0][0] + pos_list[0][1])
                p_record_black.append(pi)
                z_record_black.append(1)
            else:
                board_record_white.append(board)
                last_move_record_white.append(15 * pos_list[0][0] + pos_list[0][1])
                p_record_white.append(pi)
                z_record_white.append(1)
            i += 1
        return

    def gen_live4_defend_data(self, black_data, white_data, sample_num=5000):
        board_record_black, last_move_record_black, p_record_black, z_record_black = black_data[0], black_data[1], black_data[2], black_data[3]
        board_record_white, last_move_record_white, p_record_white, z_record_white = white_data[0], white_data[1], white_data[2], white_data[3]
        i = 0
        while i < sample_num:
            color = np.random.randint(0, 2) * 2 - 1
            board = np.zeros(225)
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=4)
            if len(fix_pos_list) == 0:
                continue

            for x, y in pos_list:
                board[15 * x + y] = color

            pi = np.array([0.0 for _ in range(225)])
            if len(fix_pos_list) == 2:
                ind_1 = 15 * fix_pos_list[0][0] + fix_pos_list[0][1]
                ind_2 = 15 * fix_pos_list[1][0] + fix_pos_list[1][1]
                pi[ind_1], pi[ind_2] = 0.5, 0.5
            if len(fix_pos_list) == 1:
                ind = 15 * fix_pos_list[0][0] + fix_pos_list[0][1]
                pi[ind] = 1

            self._add_noise(board=board, next_player=-color, max_stone_num=self._max_noise_stone_num,
                            fix_pos_list=fix_pos_list)

            if color == -1:
                board_record_black.append(board)
                last_move_record_black.append(15 * pos_list[0][0] + pos_list[0][1])
                p_record_black.append(pi)
                z_record_black.append(-1)
            else:
                board_record_white.append(board)
                last_move_record_white.append(15 * pos_list[0][0] + pos_list[0][1])
                p_record_white.append(pi)
                z_record_white.append(-1)
            i += 1
        return

    def generate_train_data(self, path, sample_num = 5000):
        board_record_black, last_move_record_black, p_record_black, z_record_black = [], [], [], []
        board_record_white, last_move_record_white, p_record_white, z_record_white = [], [], [], []
        black_data = [board_record_black, last_move_record_black, p_record_black, z_record_black]
        white_data = [board_record_white, last_move_record_white, p_record_white, z_record_white]

        self.gen_live3_defend_data(black_data, white_data, sample_num)
        self.gen_live3_attack_data(black_data, white_data, sample_num)
        self.gen_live4_defend_data(black_data, white_data, sample_num)
        self.gen_live4_attack_data(black_data, white_data, sample_num)
        print(len(black_data[0]), len(black_data[1]), len(black_data[2]), len(black_data[3]))
        print(len(white_data[0]), len(white_data[1]), len(white_data[2]), len(white_data[3]))

        torch.save(np.array(board_record_black), path + '/black/board_record.hyt')
        torch.save(np.array(last_move_record_black), path + '/black/last_move_record.hyt')
        torch.save(np.array(p_record_black), path + '/black/p_record.hyt')
        torch.save(np.array(z_record_black), path + '/black/z_record.hyt')
        torch.save(np.array(board_record_white), path + '/white/board_record.hyt')
        torch.save(np.array(last_move_record_white), path + '/white/last_move_record.hyt')
        torch.save(np.array(p_record_white), path + '/white/p_record.hyt')
        torch.save(np.array(z_record_white), path + '/white/z_record.hyt')

if __name__ == '__main__':
    data_generator = Generator(2)
    data_generator.generate_train_data('./gamedata/enhanced', sample_num = 5000)