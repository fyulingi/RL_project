import json
import os

import numpy as np

from strategy.random_place import randplace
from strategy import mentor
from strategy import MCTS
from agent import MCTSAgent
import torch

SIZE = 15

# 放置棋⼦
def place(board, x, y, color):
    if x >= 0 and y >= 0:
        board[x*SIZE+y] = color


# 处理输⼊，还原棋盘
def restoreBoard():
    fullInput = json.loads(input())
    requests = fullInput["requests"]
    responses = fullInput["responses"]
    board = np.zeros(225)
    # color #  1: black, -1: white
    if requests[0]["x"] == -1 and requests[0]["y"] == -1:
        color = 1
    else:
        color = -1
    turn = len(responses)
    for i in range(turn):
        place(board, requests[i]["x"], requests[i]["y"], -color)
        place(board, responses[i]["x"], responses[i]["y"], color)
    place(board, requests[turn]["x"], requests[turn]["y"], -color)
    return board, color, requests[turn]["x"]*15+requests[turn]["y"]

if __name__ == '__main__':
    board, color, last_move = restoreBoard()

    # random
    # x, y = randplace(board)

    # score-based, not-learning
    # agent = mentor.Mentorai(color, board)
    # x, y = agent.next_action()

    # MCTS
    # todo: if run on botzone, please upload models to "管理存储空间",
    #     and change `model_path` to '/data/models'
    model_path = os.getcwd()+'/./models'
    # print(model_path)
    mcts_config = {'c_puct': 5, 'simulation_times': 1600, 'tau_init': 1, 'tau_decay': 0.8, 'self_play': False, 'gamma': 0.95, 'num_threads': 1}
    ai_mcts = MCTSAgent(mcts_config, model_path, 0, color, board, 6, torch.device('cpu'))
    # agent = MCTS.MCTSAgent({'c_puct': 5, 'simulation_times': 1600, 'tau_init': 1, 'tau_decay': 0.8, 'self_play': False, 'gamma': 0.95, 'num_threads': 8}, './models1', 0, 1, 6, torch.device('cuda'))
    # print(ai_playing(ai_mcts, ai_mentor, True))
    x, y = ai_mcts.next_action(last_move)
    print(json.dumps({"response": {"x": x, "y": y}}))
