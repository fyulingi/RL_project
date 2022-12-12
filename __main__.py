import json
import numpy as np
from strategy.random_place import randplace
from strategy import mentor

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
    return board, color

if __name__ == '__main__':
    board, color = restoreBoard()

    # random
    # x, y = randplace(board)

    # score-based, not-learning
    agent = mentor.Mentorai(color, board)
    x, y = agent.action()

    print(json.dumps({"response": {"x": x, "y": y}}))
