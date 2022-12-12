import json
import numpy
import random

SIZE = 15


# 放置棋⼦
def place(board, x, y):
    if x >= 0 and y >= 0:
        board[x][y] = True


# 随机产⽣决策
def randplace(board):
    empty_grid = []
    for x in range(SIZE):
        for y in range(SIZE):
            if not board[x][y]:
                empty_grid.append((x, y))
    return random.choice(empty_grid)


# 处理输⼊，还原棋盘
def restoreBoard():
    fullInput = json.loads(input())
    requests = fullInput["requests"]
    responses = fullInput["responses"]
    board = numpy.zeros((SIZE, SIZE), dtype=numpy.bool)
    turn = len(responses)
    for i in range(turn):
        place(board, requests[i]["x"], requests[i]["y"])
        place(board, responses[i]["x"], responses[i]["y"])
    place(board, requests[turn]["x"], requests[turn]["y"])
    return board

board = restoreBoard()
x, y = randplace(board)
print(json.dumps({"response": {"x": x, "y": y}}))
