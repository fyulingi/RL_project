from strategy.MCTS import *
from strategy.mentor import *
import torch
from model import *

class Agent():

    def __init__(self, color, board):
        self.color = color
        self.board = board

    # def move(self, x, y, color):
    #     if x >= 0 and y >= 0:
    #         self.board[x * 15 + y] = color

    def next_action(self):
        pass

    def play(self, board, last_move):
        pass

    def self_play(self):
        pass

    # def self_play_and_save(self, path, num_games, verbose=False):
    #     board_record_black, last_move_record_black, p_record_black, z_record_black = [], [], [], []
    #     board_record_white, last_move_record_white, p_record_white, z_record_white = [], [], [], []
    #     for i in range(num_games):
    #         board_record, last_move_record, p_record, z_record = self.self_play()
    #         board_record_black.extend(board_record[::2])
    #         last_move_record_black.extend(last_move_record[::2])
    #         p_record_black.extend(p_record[::2])
    #         z_record_black.extend(z_record[::2])
    #         board_record_white.extend(board_record[1::2])
    #         last_move_record_white.extend(last_move_record[1::2])
    #         p_record_white.extend(p_record[1::2])
    #         z_record_white.extend(z_record[1::2])
    #         if verbose and i % 10 == 9:
    #             print(f"{i + 1} games finished, total steps {len(z_record_black) + len(z_record_white)}.")
    #     torch.save(np.array(board_record_black), path + '/black/board_record.hyt')
    #     torch.save(np.array(last_move_record_black), path + '/black/last_move_record.hyt')
    #     torch.save(np.array(p_record_black), path + '/black/p_record.hyt')
    #     torch.save(np.array(z_record_black), path + '/black/z_record.hyt')
    #     torch.save(np.array(board_record_white), path + '/white/board_record.hyt')
    #     torch.save(np.array(last_move_record_white), path + '/white/last_move_record.hyt')
    #     torch.save(np.array(p_record_white), path + '/white/p_record.hyt')
    #     torch.save(np.array(z_record_white), path + '/white/z_record.hyt')


class MentorAgent(Agent):

    def __init__(self, color, board):
        super().__init__(color, board)
        self.ai = Mentorai(color, board)

    def next_action(self):
        x, y = self.ai.action()
        # self.move(x, y, self.color)
        return x, y

    # def play(self, last_move=None):
    #     return self.ai.action()

    # def self_play(self, degree=3):
    #     game_result = "unfinished"
    #     board = np.zeros(225)
    #     last_move = None
    #     color = 1
    #     board_record = []
    #     last_move_record = []
    #     p_record = []
    #     z_record = []
    #     while game_result == "unfinished":
    #         last_move_record.append(last_move)
    #         action = self.ai.action()
    #         board_record.append(board.copy())
    #         p = [0.0] * 225
    #         p[action] = 1.0
    #         p_record.append(p)
    #         board[action] = color
    #         color = -color
    #         last_move = action
    #         game_result = check_result(board, last_move)
    #     reward = 0 if game_result == "draw" else 1
    #     z_record.append(reward)
    #     while len(z_record) < len(board_record):
    #         z_record.append(-self.ai.gamma * z_record[-1])
    #     z_record = z_record[::-1]
    #     return board_record, last_move_record, p_record, z_record


class MCTSAgent(Agent):

    def __init__(self, config, path, version, color, board, stochastic_steps, device):

        super().__init__(color, board)
        black_net = GomokuNet({'color': color, 'learning_rate': 2e-3, 'momentum': 9e-1, 'l2': 1e-4, 'batch_size': 32, 'path': path + '/black', 'version': version}).to(device=device)
        white_net = GomokuNet({'color': color, 'learning_rate': 2e-3, 'momentum': 9e-1, 'l2': 1e-4, 'batch_size': 32, 'path': path + '/white', 'version': version}).to(device=device)
        self.ai = MCTS(config, black_net, white_net, color, stochastic_steps)

    def play(self, board, last_move):

        action, p = self.ai.action(board, last_move)
        return action

    def self_play(self):

        self.ai.reset()
        self.ai.self_play = True
        game_result = "unfinished"
        board = np.zeros(225)
        last_move = None
        color = 1
        board_record = []
        last_move_record = []
        p_record = []
        z_record = []
        while game_result == "unfinished":
            last_move_record.append(last_move)
            action, p = self.ai.action(board, last_move)
            board_record.append(board.copy())
            p_record.append(p)
            board[action] = color
            color = -color
            last_move = action
            game_result = check_result(board, last_move)
        reward = 0 if game_result == "draw" else 1
        z_record.append(reward)
        while len(z_record) < len(board_record):
            z_record.append(-self.ai.gamma * z_record[-1])
        z_record = z_record[::-1]
        return board_record, last_move_record, p_record, z_record

    

