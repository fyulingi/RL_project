from strategy.MCTS import *
from strategy.mentor import *
from strategy.improved_mentor import *
from strategy.MCTS_Node_model.model import *


class Agent():
    def __init__(self, color, board, name):
        self.color = color
        self.board = board
        self.name = name

    def next_action(self):
        pass

    def move_one_step(self, action, color):
        pass

    def reset(self):
        pass


class MentorAgent(Agent):

    def __init__(self, color, board):
        super().__init__(color, board, 'mentor')
        self.ai = Mentorai(color, board)

    def next_action(self):
        action = self.ai.action()
        # self.move(x, y, self.color)
        return action

    def move_one_step(self, action, color):
        pass

    def reset(self):
        pass


class MCTSAgent(Agent):
    def __init__(self, color, board, mcts_config, model_config, net=None):
        super().__init__(color, board, 'mcts')
        if net is None:
            net = GomokuNet(model_config)
        self.ai = MCTS(mcts_config, net, color, board)

    def next_action(self):
        action, _ = self.ai.action()
        return action

    def move_one_step(self, action, color):
        self.ai.move_one_step(action)

    def play(self):
        action, p = self.ai.action()
        return action, p

    def reset(self):
        self.ai.reset()


class ImMentorAgent(Agent):
    def __init__(self, color, board):
        super().__init__(color, board, 'improved_mentor')
        self.ai = ImMentorai(color,board)

    def next_action(self):
        action = self.ai.get_action(4, 2)
        return action

    def move_one_step(self, action, color):
        pass

    def reset(self):
        pass