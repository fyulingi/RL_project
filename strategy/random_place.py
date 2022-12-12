import random

# 随机产⽣决策
def randplace(board):
    empty_grid = []
    for id in range(225):
        if board[id] == 0:
            empty_grid.append(id)
    random_id = random.choice(empty_grid)
    return random_id//15, random_id%15