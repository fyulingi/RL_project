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
        assert not self.is_end and self.children
        UCB_list = np.array([child.compute_UCB(c_puct) for child in self.children])
        best_action = np.argmax(UCB_list)
        return self.children[best_action], best_action
        # sorted_idx = np.argsort(UCB_list)
        # for i in reversed(range(len(sorted_idx))):
            # if legal_moves[sorted_idx[i]]:
            #     action = sorted_idx[i]
            #     print("In Node::select, action = ", action)
            #     return self.children[action], action
        # return None, -1

    def expand(self, board, p_prior, legal_moves):
        """
        Parameters:
        -----------
        p_prior: 225-d vector, prior probabilities to choose each move
        """
        assert not self.is_end and not self.children
        for move in range(225):
            if legal_moves[move]:
                self.children.append(Node(p_prior[move], self, -self.color, move))

    def backup(self, value, gamma, use_virtual=False):
        """
        Parameters:
        -----------
        value: float, value for backup (to what extend the temp color temds to win)
        gamma: float, discount factor of the reward
        """
        if use_virtual and self.select_num > 0:
            self.select_num -= 1
            self.N -= 10
            if self.N < 0:
                self.N += 10
        self.N += 1
        self.W += value
        self.Q = self.W / self.N
        if self.parent is not None:
            self.parent.backup(-gamma * value, gamma, use_virtual)  # switch sign because of the switch of color