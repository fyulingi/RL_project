import numpy as np


def get_search_field(board):
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
            temp = 15 * (i + 2 * d[0]) + j + 2 * d[1]
            if 0 <= i + 2 * d[0] < 15 and 0 <= j + 2 * d[1] < 15 and board[temp] == 0:
                if temp not in res:
                    res.append(temp)
    return res


def count_dir_num(board, color, x, y, x_dir, y_dir):
    res = 0
    while 0 <= x < 15 and 0 <= y < 15:
        if board[x * 15 + y] == color:
            x += x_dir
            y += y_dir
            res += 1
        else:
            break
    return res


def check_result(board, last_x, last_y):
    if last_x < 0 or last_y < 0:
        return "unfinished"
    color = board[last_x*15+last_y]
    win_mess = "blackwin" if color == 1 else "whitewin"
    # left, right, up, down, leftup, rightdown, leftdown, rightup
    x_dir = [0, 0, -1, 1, -1, 1, 1, -1]
    y_dir = [-1, 1, 0, 0, -1, 1, -1, 1]
    for i in range(0, 8, 2):
        a = count_dir_num(board, color, last_x, last_y, x_dir[i], y_dir[i])
        a += count_dir_num(board, color, last_x, last_y, x_dir[i+1], y_dir[i+1])-1
        if a >= 5:
            return win_mess
    return "draw" if np.sum(np.abs(board)) == 225 else "unfinished"

def board2str(board, color, pos):

    def color2str(i, j):
        stone = board[15 * i + j]
        if stone == 0:
            return "0"
        if stone == color:
            return "2"
        return "1"

    i, j = pos // 15, pos % 15
    h, v, d, c = "", "", "", ""

    for k in range(15):
        h += color2str(i, k)
        v += color2str(k, j)
        if 0 <= k + j - i < 15:
            d += color2str(k, k + j - i)
        if 0 <= i + j - k < 15:
            c += color2str(k, i + j - k)

    h = "1" + h + "1"
    v = "1" + v + "1"
    d = "1" + d + "1"
    c = "1" + c + "1"
    
    return h, v, d, c

def printboard(board):
    for i in range(15):
        for j in range(15):
            print(f"{int(board[i * 15 + j]):3d}", end="")
        print()