import os
import numpy as np
import agent
import torch
# from utils import printboard
from utils import check_result


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
    model_path = os.getcwd() + '/./models'
    print("model path: ", model_path)
    mcts_config = {'c_puct': 5, 'version': 0, 'simulation_times': 100, 'device': torch.device('cpu'), 'model_path': model_path,
                   'tau_init': 1, 'tau_decay': 0.8, 'self_play': False, 'gamma': 0.95, 'num_threads': 1, 'stochastic_steps': 0}
    print("Mentor black v.s. MCTS white:")
    player1 = agent.MentorAgent(1, board)
    player2 = agent.MCTSAgent(mcts_config, -1, board)
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
        player2.reset()
    print(f"Mentor black wins {p1_win}, loses {p2_win}, draws {plays_num - p1_win - p2_win}.")
    print("MCTS black v.s. Mentor white:")
    player1 = agent.MCTSAgent(mcts_config, 1, board)
    player2 = agent.MentorAgent(-1, board)
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
        player1.reset()
    print(f"Mcts black wins {p1_win}, loses {p2_win}, draws {plays_num - p1_win - p2_win}.")


if __name__ == '__main__':
    # mentor_mentor(50)
    mentor_MCTS(1)