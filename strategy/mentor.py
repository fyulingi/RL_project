import numpy as np
from utils import board2str

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
    
    def mode_counter(self, lines, modes):
        count = 0
        for line in lines:
            for mode in modes:
                if line.find(mode) != -1:
                    count += 1
                    break
        return count


    def compute_line_score(self, lines):

        score = 0
        
        if self.mode_counter(lines, ["22222"]) >= 1:
            score += 1000000

        if self.mode_counter(lines, ["022220"]) >= 1:
            score += 50000

        if self.mode_counter(lines, ["022221", "122220", "20222", "22202", "22022"]) >= 2:
            score += 10000

        if self.mode_counter(lines, ["022221", "122220", "20222", "22202", "22022"]) >= 1 and self.mode_counter(lines, ["02220", "020220", "022020"]) >= 1:
            score += 10000

        if self.mode_counter(lines, ["02220", "020220", "022020"]) >= 2:
            score += 10000
        
        if self.mode_counter(lines, ["02220", "020220", "022020"]) >= 1 and self.mode_counter(lines, ["002221", "122200", "020221", "122020", "022021", "120220", "20022", "22002", "20202", "1022201"]) >= 1:
            score += 1000
        
        score += 100 * self.mode_counter(lines, ["02220", "020220", "022020"])

        if self.mode_counter(lines, ["002200", "02020", "02002", "20020"]) >= 2:
            score += 1000

        score += 50 * self.mode_counter(lines, ["002221", "122200", "020221", "122020", "022021", "120220", "20022", "22002", "20202", "1022201"])

        if self.mode_counter(lines, ["002200", "02020", "02002", "20020"]) >= 1 and self.mode_counter(lines, ["000221", "122000", "002021", "120200", "020021", "120020", "20002"]) >= 1:
            score += 10

        score += 5 * self.mode_counter(lines, ["002200", "02020", "02002", "20020"])

        score += 3 * self.mode_counter(lines, ["000221", "122000", "002021", "120200", "020021", "120020", "20002"])

        score -= 5 * self.mode_counter(lines, ["122221"])

        score -= 5 * self.mode_counter(lines, ["12221"])

        score -= 5 * self.mode_counter(lines, ["1221"])

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

# if __name__ == '__main__':
#
#     board = np.zeros(225)
#     agent = Mentorai(1)
#     # board[112] = 1
#     # search_field = np.array(agent.get_search_field(board))
#     # pos = search_field[0]
#     # board[pos] = 1
#     # print(board2str(board, 1, pos))
#     # print(agent.compute_line_score(board2str(board, 1, pos)))
#     stones = 0
#     color = 1
#     while True:
#         stones += 1
#         action = agent.action(board, color, 2)
#         print(f"{stones}: ({action // 15},{action % 15})")
#         # search_field = np.array(agent.get_search_field(board))
#         # scores = np.array([agent.compute_score(board, pos) for pos in search_field])
#         # print(search_field)
#         # print(scores)
#         board[action] = color
#         color = -color
#         if np.sum(np.abs(board)) == 225:
#             break