import os
import numpy as np
import agent
import torch
from utils import printboard
from utils import check_result


def init_state(board, player1, player2):
    for j in range(225):
        board[j] = 0
    player1.reset()
    player2.reset()


def move_one_step(color, x, y, board, actor1, actor2):
    actor1.move_one_step(x, y, color)
    actor2.move_one_step(x, y, color)
    board[x * 15 + y] = color


def play_one_game(board, actor1:agent, actor2):
    game_result = "unfinished"
    turn = 0
    while game_result == "unfinished":
        if turn % 2 == 0:
            x, y = actor1.next_action()
            move_one_step(1, x, y, board, actor1, actor2)
            # print(f"black: {x}, {y}")
        else:
            x, y = actor2.next_action()
            move_one_step(-1, x, y, board, actor1, actor2)
            # print(f"white: {x}, {y}")
        game_result = check_result(board, x, y)
        turn += 1
    return game_result, turn


def two_players_play(board, player1, player2, plays_num):
    print(player1.name + " black v.s. " + player2.name + " white:")
    p1_win = p2_win = 0
    for i in range(plays_num):
        init_state(board, player1, player2)
        res, turns = play_one_game(board, player1, player2)
        print(f"Game {i}, " + res + ", turns: " + str(turns))
        if res == "blackwin":
            p1_win += 1
        elif res == "whitewin":
            p2_win += 1
        # printboard(board)
    print(f"{player1.name} black wins {p1_win}, loses {p2_win}, draws {plays_num - p1_win - p2_win}.")


def mentor_mentor(plays_num):
    board = np.zeros([225])
    player1 = agent.MentorAgent(1, board)
    player2 = agent.MentorAgent(-1, board)
    two_players_play(board, player1, player2, plays_num)


def mentor_MCTS(plays_num):
    board = np.zeros([225])
    model_path = os.getcwd() + '/./models'
    print("model path: ", model_path)
    mcts_config = {'c_puct': 5, 'version': 1, 'simulation_times': 100, 'device': torch.device('cpu'), 'model_path': model_path,
                   'tau_init': 1, 'tau_decay': 0.8, 'self_play': False, 'gamma': 0.95, 'num_threads': 1, 'stochastic_steps': 0}
    player1 = agent.MentorAgent(1, board)
    player2 = agent.MCTSAgent(mcts_config, -1, board)
    two_players_play(board, player1, player2, plays_num)

    player1 = agent.MCTSAgent(mcts_config, 1, board)
    player2 = agent.MentorAgent(-1, board)
    two_players_play(board, player1, player2, plays_num)


def MCTS_MCTS(plays_num):
    board = np.zeros([225])
    model_path = os.getcwd() + '/./models'
    print("model path: ", model_path)
    mcts_config = {'c_puct': 5, 'version': 1, 'simulation_times': 5, 'device': torch.device('cpu'), 'model_path': model_path,
                   'tau_init': 1, 'tau_decay': 0.8, 'self_play': False, 'gamma': 0.95, 'num_threads': 1, 'stochastic_steps': 0}
    player1 = agent.MCTSAgent(mcts_config, 1, board)
    player2 = agent.MCTSAgent(mcts_config, -1, board)
    two_players_play(board, player1, player2, plays_num)

if __name__ == '__main__':
    # mentor_mentor(50)
    # mentor_MCTS(50)
    MCTS_MCTS(50)