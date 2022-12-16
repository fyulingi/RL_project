import numpy as np
import agent
# from utils import printboard


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


def play_one_game(board, actor1:agent, actor2):
    game_result = "unfinished"
    turn = 0
    while game_result == "unfinished":
        if turn % 2 == 0:
            x, y = actor1.next_action()
            board[x * 15 + y] = 1
        else:
            x, y = actor2.next_action()
            board[x * 15 + y] = -1
        game_result = check_result(board, x, y)
        turn += 1
    return game_result


def mentor_mentor(plays_num):
    board = np.zeros([225])
    player1 = agent.MentorAgent(1, board)
    player2 = agent.MentorAgent(-1, board)
    print("Mentor black v.s. Mentor white:")
    p1_win = p2_win = 0
    for i in range(plays_num):
        for j in range(225):
            board[j] = 0
        res = play_one_game(board, player1, player2)
        print(f"Game {i}, " + res)
        if res == "blackwin":
            p1_win += 1
        elif res == "whitewin":
            p2_win += 1
        # printboard(board)
    print(f"Mentor black wins {p1_win}, loses {p2_win}.")


def mentor_MCTS(plays_num):
    board = np.zeros([225])
    player1 = agent.MentorAgent(1, board)
    player2 = agent.MCTSAgent(-1, board)
    print("Mentor black v.s. Mentor white:")
    p1_win = p2_win = 0
    for i in range(plays_num):
        for j in range(225):
            board[j] = 0
        res = play_one_game(board, player1, player2)
        print(f"Game {i}, " + res)
        if res == "blackwin":
            p1_win += 1
        elif res == "whitewin":
            p2_win += 1
    print(f"Mentor black wins {p1_win}, loses {p2_win}.")


if __name__ == '__main__':
    mentor_mentor(50)