import numpy as np


class State():
    def __init__(self, value):
        self.value = value

    def n_5_2(self, weight=1):
        return self.value[0] * weight

    def n_4_2(self, weight=1):  # 2 for own stones
        return self.value[1] * weight

    def n_33_2(self, weight=1):
        return self.value[2] * weight

    def n_3_2(self, weight=1):
        return self.value[3] * weight

    def n_2_2(self, weight=1):
        return self.value[4] * weight

    def n_1_2(self, weight=1):
        return self.value[5] * weight

    def n_5_1(self, weight=1):
        return self.value[6] * weight

    def n_4_1(self, weight=1):
        return self.value[7] * weight

    def n_33_1(self, weight=1):
        return self.value[8] * weight

    def n_3_1(self, weight=1):
        return self.value[9] * weight

    def n_2_1(self, weight=1):
        return self.value[10] * weight

    def n_1_1(self, weight=1):
        return self.value[11] * weight

    def __add__(self, another):
        return State([self.value[i] + another.value[i] for i in range(12)])

    def __sub__(self, another):
        return State([self.value[i] - another.value[i] for i in range(12)])

    def switch(self):
        return State(self.value[6:] + self.value[:6])

    def score(self):
        if self.n_5_1() >= 1:
            return -100000
        if self.n_4_2() >= 1:
            return 100000
        if self.n_4_1() >= 2:
            return -100000
        if self.n_4_1() >= 1 and self.n_33_1() >= 1:
            return -50000
        if self.n_33_2() >= 2:
            return 10000
        if self.n_33_2() == 0 and self.n_33_1() >= 2:
            return -10000
        return self.n_33_2(1000) - self.n_4_1(1000) - self.n_33_1(800) + self.n_3_2(100) - self.n_3_1(80) + self.n_2_2(
            10) - self.n_2_1(8) + self.n_1_2(1) - self.n_1_1(1)

    def score_prune(self):
        return self.n_5_2(100000) + 50000 * int(
            self.n_4_2() >= 2 or (self.n_4_2() >= 1 and self.n_33_2() >= 1)) + 10000 * int(
            self.n_33_2() >= 2) + self.n_33_2(1000) + self.n_4_2(1000) + self.n_3_2(100) + self.n_2_2(10) + self.n_1_2(
            1)


class ImMentorai():

    def __init__(self, color, board):
        self.color = color
        self.board = board

    def get_line_stat(self, linstr):
        length = len(linstr)
        counter = [0] * 6
        flag = [0] * length
        zeroidx = []
        l3_keep = 0  # 3/5 length
        zero3_keep = None  # 3/5 head node
        l = 0  # left
        num2 = 0
        for i in range(length):
            # print(f"i={i},flag={flag},zero3_keep={zero3_keep},l3_keep={l3_keep}")
            if linstr[i] == '0':
                zeroidx.append(i)
            elif linstr[i] == '1':
                l = i + 1
                if l >= length - 4:
                    break
                num2 = 0
                continue
            else:
                num2 += 1

            if i - l < 4:
                continue
            if i - l > 4:
                raise RuntimeError
            if num2 == 1:
                counter[5] += 1
            elif num2 == 2:
                counter[4] += 1
            elif num2 == 3:
                node1, node2 = zeroidx[-2], zeroidx[-1]
                if zero3_keep == node1:
                    l3_keep += 1
                    flag[node2] = l3_keep
                    zero3_keep = node2
                elif zero3_keep != node2:
                    if l3_keep > 0:
                        if l3_keep == 2:
                            counter[3] += 1
                        elif 3 <= l3_keep < 6:
                            counter[2] += 1
                        elif l3_keep >= 6:
                            counter[2] += 2
                        l3_keep = 0
                        zero3_keep = None
                    if flag[node1] != -1:
                        l3_keep = 2
                        zero3_keep = node2
                        flag[node2] = 2
            elif num2 == 4:
                node = zeroidx[-1]
                flag[node] = -1
                if zero3_keep == node:
                    l3_keep -= 1
                    if l3_keep > 0:
                        if l3_keep == 2:
                            counter[3] += 1
                        elif 3 <= l3_keep < 6:
                            counter[2] += 1
                        elif l3_keep >= 6:
                            counter[2] += 2
                        l3_keep = 0
                        zero3_keep = None
            elif num2 == 5:
                counter[0] += 1

            if linstr[l] == '2':
                num2 -= 1
            l += 1
            if l >= length - 4:
                break

        if l3_keep > 0:
            if l3_keep == 2:
                counter[3] += 1
            elif 3 <= l3_keep < 6:
                counter[2] += 1
            elif l3_keep >= 6:
                counter[2] += 2

        for i in flag:
            if i == -1:
                counter[1] += 1
        # print(flag)
        return counter

    def get_line_state(self, linstr):
        str2 = ""
        for i in range(len(linstr)):
            if linstr[i] == '0':
                str2 += '0'
            elif linstr[i] == '1':
                str2 += '2'
            else:
                str2 += '1'
        return State(self.get_line_stat(linstr) + self.get_line_stat(str2))

    def get_linstr(self, board, i, j, dir, color):
        res = ''
        if dir == 0:  # horizontal
            for y in range(15):
                stone = board[i * 15 + y]
                if stone == 0:
                    res += '0'
                elif stone == color:
                    res += '2'
                else:
                    res += '1'
        elif dir == 1:  # vertical
            for x in range(15):
                stone = board[x * 15 + j]
                if stone == 0:
                    res += '0'
                elif stone == color:
                    res += '2'
                else:
                    res += '1'
        elif dir == 2:  # diagonal
            if j >= i:
                for x in range(15 + i - j):
                    stone = board[x * 15 + x + j - i]
                    if stone == 0:
                        res += '0'
                    elif stone == color:
                        res += '2'
                    else:
                        res += '1'
            else:
                for x in range(i - j, 15):
                    stone = board[x * 15 + x + j - i]
                    if stone == 0:
                        res += '0'
                    elif stone == color:
                        res += '2'
                    else:
                        res += '1'
        elif dir == 3:  # counter diagonal
            if i + j <= 14:
                for x in range(i + j + 1):
                    stone = board[x * 15 + i + j - x]
                    if stone == 0:
                        res += '0'
                    elif stone == color:
                        res += '2'
                    else:
                        res += '1'
            else:
                for x in range(i + j - 14, 15):
                    stone = board[x * 15 + i + j - x]
                    if stone == 0:
                        res += '0'
                    elif stone == color:
                        res += '2'
                    else:
                        res += '1'
        return res

    def get_search_field(self, board):
        way = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        res = []
        for pos in range(225):
            if board[pos] == 0:
                continue
            i, j = pos // 15, pos % 15
            for d in way:
                temp = 15 * (i + d[0]) + j + d[1]
                if 0 <= i + d[0] < 15 and 0 <= j + d[1] < 15 and board[temp] == 0:
                    if temp not in res:
                        res.append(temp)
        return res

    def get_board_state(self, board, color):
        state = State([0] * 12)
        for i in range(15):
            state = state + self.get_line_state(self.get_linstr(board, i, 0, 0, color))
        for j in range(15):
            state = state + self.get_line_state(self.get_linstr(board, 0, j, 1, color))
        for i in range(11):
            state = state + self.get_line_state(self.get_linstr(board, i, 0, 2, color))
        for j in range(1, 11):
            state = state + self.get_line_state(self.get_linstr(board, 0, j, 2, color))
        for i in range(4, 15):
            state = state + self.get_line_state(self.get_linstr(board, i, 0, 3, color))
        for j in range(1, 11):
            state = state + self.get_line_state(self.get_linstr(board, 14, j, 3, color))
        return state

    def get_state_diff(self, board, i, j, color1, color2):
        tempboard = board.copy()
        tempboard[i * 15 + j] = color2
        state = State([0] * 12)
        for dir in range(4):
            state = state - self.get_line_state(self.get_linstr(board, i, j, dir, color1)) + self.get_line_state(
                self.get_linstr(tempboard, i, j, dir, color1))
        return state

    def get_prune_score(self, board, i, j, color, tempstate=None):
        if tempstate is None:
            tempstate = self.get_board_state(board, color)
        return (tempstate + self.get_state_diff(board, i, j, color, color)).score_prune() + 0.8 * (
                    tempstate + self.get_state_diff(board, i, j, color, -color)).switch().score_prune()

    def evaluate(self, board, color, depth, Maxdepth, alpha, beta, tempstate=None, expand_num=3):
        if tempstate is None:
            tempstate = self.get_board_state(board, color)
        if tempstate.n_5_1() > 0:
            return 100000, None
        if depth == Maxdepth:
            return -tempstate.score(), None
        else:
            best_move = None
            search_list = self.get_search_field(board)
            if len(search_list) > expand_num:
                prune_scores = []
                for action in search_list:
                    prune_scores.append(self.get_prune_score(board, action // 15, action % 15, color, tempstate))
                expand_idx = np.argsort(np.array(prune_scores))[::-1]
                for i in range(expand_num):
                    if alpha >= beta:
                        break
                    child_board = board.copy()
                    action = search_list[expand_idx[i]]
                    child_board[action] = color
                    child_state = (self.get_state_diff(board, action // 15, action % 15, color,
                                                       color) + tempstate).switch()
                    child_val, _ = self.evaluate(child_board, -color, depth + 1, Maxdepth, -80000, -1.25 * alpha,
                                                 child_state, expand_num)
                    child_val *= 0.8
                    if child_val > alpha:
                        best_move = action
                        alpha = child_val
                return -alpha, best_move
            else:
                for action in search_list:
                    child_board = board.copy()
                    child_board[action] = color
                    child_state = (self.get_state_diff(board, action // 15, action % 15, color,
                                                       color) + tempstate).switch()
                    child_val, _ = self.evaluate(child_board, -color, depth + 1, Maxdepth, -80000, -1.25 * alpha,
                                                 child_state, expand_num)
                    child_val *= 0.8
                    if child_val > alpha:
                        best_move = action
                        alpha = child_val
                return -alpha, best_move

    def get_action(self, depth, expand_num=3):
        if np.sum(np.abs(self.board)) == 0:
            return 112
        _, best_move = self.evaluate(self.board, self.color, 0, depth, -80000, 80000, None, expand_num)
        return best_move