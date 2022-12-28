import numpy as np
from utils import board2str


def mode_counter(lines, modes):
    count = 0
    for line in lines:
        for mode in modes:
            if line.find(mode) != -1:
                count += 1
                break
    return count


class Mentorai():
    def __init__(self, color, board):
        self.color = color
        self.board = board

    # comment: 不是邻居的点可能效果好
    def get_search_field(self):
        way = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        res = []
        for pos in range(225):
            if self.board[pos] == 0:
                continue
            i, j = pos // 15, pos % 15
            for d in way:
                temp = 15 * (i + d[0]) + j + d[1]
                if 0 <= i + d[0] < 15 and 0 <= j + d[1] < 15 and self.board[temp] == 0:
                    if temp not in res:
                        res.append(temp)
        return res

    def compute_line_score(self, lines):
        score = 0
        
        if mode_counter(lines, ["22222"]) >= 1:
            score += 1000000
        if mode_counter(lines, ["022220"]) >= 1:
            score += 50000
        if mode_counter(lines, ["022221", "122220", "20222", "22202", "22022"]) >= 2:
            score += 10000
        if mode_counter(lines, ["022221", "122220", "20222", "22202", "22022"]) >= 1 and mode_counter(lines, ["02220", "020220", "022020"]) >= 1:
            score += 10000
        if mode_counter(lines, ["02220", "020220", "022020"]) >= 2:
            score += 10000
        if mode_counter(lines, ["02220", "020220", "022020"]) >= 1 and mode_counter(lines, ["002221", "122200", "020221", "122020", "022021", "120220", "20022", "22002", "20202", "1022201"]) >= 1:
            score += 1000
        score += 100 * mode_counter(lines, ["02220", "020220", "022020"])
        if mode_counter(lines, ["002200", "02020", "02002", "20020"]) >= 2:
            score += 1000
        score += 50 * mode_counter(lines, ["002221", "122200", "020221", "122020", "022021", "120220", "20022", "22002", "20202", "1022201"])
        if mode_counter(lines, ["002200", "02020", "02002", "20020"]) >= 1 and mode_counter(lines, ["000221", "122000", "002021", "120200", "020021", "120020", "20002"]) >= 1:
            score += 10
        score += 5 * mode_counter(lines, ["002200", "02020", "02002", "20020"])
        score += 3 * mode_counter(lines, ["000221", "122000", "002021", "120200", "020021", "120020", "20002"])

        score -= 5 * mode_counter(lines, ["122221"])
        score -= 5 * mode_counter(lines, ["12221"])
        score -= 5 * mode_counter(lines, ["1221"])

        return score

    def compute_score(self, pos):
        temp = self.board.copy()
        temp[pos] = self.color
        score = self.compute_line_score(board2str(temp, self.color, pos))
        temp[pos] = -self.color
        score += 0.8 * self.compute_line_score(board2str(temp, -self.color, pos))
        return score

    def action(self, degree=3):
        if np.sum(np.abs(self.board)) == 0:
            return 112//15, 112%15
        search_field = np.array(self.get_search_field())
        scores = np.array([self.compute_score(pos) for pos in search_field])
        best_score = np.max(scores)
        good_moves = np.sum(scores >= 0.8 * best_score)
        best_idx = np.argsort(scores)[::-1]
        degree = max(min(degree, good_moves, len(scores)),1)
        next_id = np.random.choice(search_field[best_idx[:degree]])
        return int(next_id//15), int(next_id%15)