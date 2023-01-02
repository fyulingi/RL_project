from math import sqrt
import numpy as np


class Node:
    # p_prior: float, prior probability for its parent to choose this move
    # parent: Node, its parent node
    # color: +1 or -1, which color to move at this moment
    # move: this move, if init state, then move = -1
    def __init__(self, p_prior, parent, color, move):
        # N: visit num
        self.N = 0
        self.Q = 0
        self.W = 0
        self.P = p_prior  # prior probability for its parent to choose this move
        self.parent = parent
        self.children = []
        self.color = color
        self.select_num = 0
        self.move = move
        self.is_end = False

    def compute_UCB(self, c_puct):
        return -self.Q + (c_puct * self.P * sqrt(self.parent.N) / (1 + self.N) if self.parent is not None else 0)

    def select(self, c_puct):
        assert not self.is_end and len(self.children) > 0
        UCB_list = np.array([child.compute_UCB(c_puct) for child in self.children])
        best_action = np.argmax(UCB_list)
        return self.children[best_action], best_action

    def expand(self, p_prior, legal_moves):
        """
        Parameters:
        -----------
        p_prior: 225-d vector, prior probabilities to choose each move
        """
        assert not self.is_end and len(self.children) == 0
        p_sum = 1e-10
        for move in legal_moves:
            p_sum += p_prior[move].item()
        for move in legal_moves:
            self.children.append(Node(p_prior[move].item() / p_sum, self, -self.color, move))