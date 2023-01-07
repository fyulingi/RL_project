import numpy as np
from strategy.MCTS_Node_model.model import GomokuNet
from utils import check_result, get_search_field
from strategy.MCTS_Node_model.Node import Node



class MCTS:
    def __init__(self, config, net: GomokuNet, color, board):
        """
        Parameters:
        -----------
        color: -1 or +1, the color of the MCTS agent, +1 for black, -1 for white, only counts when not self-playing
        stochastic_steps: int, total stones played by stochastic methods, 0 for deterministic, >0 for stochastic (training only)
        """
        self.root = Node(1.0, None, -color, -1)    # start from empty board
        self.board = board    # board info vector for root node
        self.last_move = None    # update the last move in the root board
        self.c_puct = config['c_puct']
        self.simulation_times = config['simulation_times']
        self.tau_init = config['tau_init']    # initial temperature
        self.tau_decay = config['tau_decay']
        self.gamma = config['gamma']
        self.num_threads = config['num_threads']
        self.net = net
        self.color = color
        self.stochastic_steps = config['stochastic_steps']
        self._expanding_list = []

    def expand_one_node_determined(self, node, action):
        new_node = Node(1, node, -node.color, action)
        node.children.append(new_node)

    def move_one_step(self, action):
        if len(self.root.children) == 0:
            self.expand_one_node_determined(self.root, action)
        for child in self.root.children:
            if child.move == action:
                self.root = child
                self.root.N += 1
                self.last_move = action
                return
        raise RuntimeError("Invalid move.")

    def update_root(self, move_x, move_y):
        action = move_x * 15 + move_y
        # todo: if child is empty list
        for child in self.root.children:
            if child.move == action:
                self.root = child
                return
        node = Node(1, None, -self.root.color, action)
        self.root = node
        self.root.parent = None

    def simulation_one_game(self):
        board = np.copy(self.board)
        node = self.root
        while check_result(board, node.move) == "unfinished":
            if node.children != []:
                node, _ = node.select(self.c_puct)
                board[node.move] = node.color
            else:
                search_field = get_search_field(board)
                p_prior, v = self.get_p_prior_v(board, -node.color, node.move)
                node.expand(p_prior, search_field)
                node, _ = node.select(self.c_puct)
                board[node.move] = node.color
        game_result = check_result(board, node.move)
        if game_result == "draw":
            self.backup(node, 0, self.gamma)
        elif (game_result == "blackwin" and self.color == 1) or (game_result == "whitewin" and self.color == -1):
            self.backup(node, 1, self.gamma)
        else:  # the color of an ended node must have lost the game
            self.backup(node, -1, self.gamma)

    def get_p_prior_v(self, board, color, last_move):
        p_prior, v = self.net.predict(board, last_move, color)
        return p_prior[0].cpu(), v[0][0].cpu()

    def simulate(self, num_steps):
        if self.num_threads == 1:
            for step in range(num_steps):
                self.simulation_one_game()

    def get_move_pi(self, move_list, N_list):
        pi = np.zeros(225)
        for move, N in zip(move_list, N_list):
            pi[move] = N
        pi = pi / np.sum(pi)
        return pi

    def action(self):
        stage = np.sum(np.abs(self.board)) + 1
        if stage == 1:
            pi = np.zeros(225)
            pi[112] = 1
            return 112, pi
        self.simulate(self.simulation_times)    # simulate
        N_list = np.array([child.N * 1.0 for child in self.root.children])
        move_list = [child.move for child in self.root.children]
        if stage > self.stochastic_steps:    # determininstic policy
            action = self.root.children[np.argmax(N_list)].move
        else:    # stochastic policy
            tau = max(self.tau_init * (self.tau_decay ** (stage // 2)), 0.04)
            pi = N_list ** (1 / tau)
            pi /= np.sum(pi)
            action = np.random.choice(move_list, p=pi)
        N_list /= np.sum(N_list)    # for training use
        pi = self.get_move_pi(move_list, N_list)
        return action, pi

    def update(self, action):
        self.root = self.root.children[action]
        self.root.N += 1
        self.last_move = action

    def reset(self):
        self.root = Node(1.0, None, -self.color, -1)  # start from empty board
        self.last_move = None  # update the last move in the root board

    def backup(self, node, value, gamma):
        while node != self.root.parent:
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            value = -value * gamma
            node = node.parent
