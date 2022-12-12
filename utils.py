import numpy as np

def check_result(board, last_move):

    """
    Parameters:
    -----------
    board: 225-d array, board info
    last_move: int, position of the last_move, None for empty board

    Returns:
    --------
    game_result: str, "whitewin", "blackwin", "draw", or "unfinished"

    """

    if last_move is None:
        return "unfinished"
    board2D = board.reshape((15, 15))
    loc2D = (last_move // 15, last_move % 15)
    x, y = loc2D
    color = board[last_move]
    assert color != 0
    winning_message = "blackwin" if color == 1 else "whitewin"
    h_num, v_num, d_num, c_num = 1, 1, 1, 1    # num of consecutive stones along horizontal, vertical, diagonal, counterdiagonal directions
    for j in range(1, 5):
        if y + j >= 15 or board2D[x, y + j] != color:
            break
        h_num += 1
    if h_num >= 5:
        return winning_message
    for j in range(1, 5):
        if y - j < 0 or board2D[x, y - j] != color:
            break
        h_num += 1
    if h_num >= 5:
        return winning_message
    for i in range(1, 5):
        if x + i >= 15 or board2D[x + i, y] != color:
            break
        v_num += 1
    if v_num >= 5:
        return winning_message
    for i in range(1, 5):
        if x - i < 0 or board2D[x - i, y] != color:
            break
        v_num += 1
    if v_num >= 5:
        return winning_message
    for i, j in zip(range(1, 5), range(1, 5)):
        if x + i >= 15 or y + j >= 15 or board2D[x + i, y + j] != color:
            break
        d_num += 1
    if d_num >= 5:
        return winning_message
    for i, j in zip(range(1, 5), range(1, 5)):
        if x - i < 0 or y - j < 0 or board2D[x - i, y - j] != color:
            break
        d_num += 1
    if d_num >= 5:
        return winning_message
    for i, j in zip(range(1, 5), range(1, 5)):
        if x + i >= 15 or y - j < 0 or board2D[x + i, y - j] != color:
            break
        c_num += 1
    if c_num >= 5:
        return winning_message
    for i, j in zip(range(1, 5), range(1, 5)):
        if x - i < 0 or y + j >= 15 or board2D[x - i, y + j] != color:
            break
        c_num += 1
    if c_num >= 5:
        return winning_message
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