from math import sqrt
import numpy as np
from utils import check_result
import time
import threading

class Node:
    def __init__(self, p_prior, parent, color):
        """
        Parameters:
        -----------
        p_prior: float, prior probability for its parent to choose this move
        parent: Node, its parent node
        color: +1 or -1, which color to move at this moment
        """

        self.N = 0
        self.Q = 0
        self.W = 0
        self.P = p_prior    # prior probability for its parent to choose this move
        self.parent = parent
        self.children = []
        self.is_end = False
        self.color = color
        self.select_num = 0

    def compute_UCB(self, c_puct):
        """
        Parameters:
        -----------
        c_puct: float, a coefficient to compute U in the paper

        Returns:
        --------
        UCB value of the edge linked between self and its parents
        """

        return self.Q + (c_puct * self.P * sqrt(self.parent.N) / (1 + self.N) if self.parent is not None else 0)

    def select(self, c_puct, legal_moves):
        """
        Parameters:
        -----------
        c_puct: float, a coefficient to compute U in the paper
        legal_moves: 225-d bool vector, where legal moves are denoted by True, otherwise False.
        
        Returns:
        --------
        child node selected
        position to place stone (action no.)

        """

        assert not self.is_end and self.children
        UCB_list = np.array([child.compute_UCB(c_puct) for child in self.children])
        sorted_idx = np.argsort(UCB_list)
        for i in reversed(range(len(sorted_idx))):
            if legal_moves[sorted_idx[i]]:
                action = sorted_idx[i]
                return self.children[action], action
    
    def expand(self, p_prior):
        """
        Parameters:
        -----------
        p_prior: 225-d vector, prior probabilities to choose each move
        """

        assert not self.is_end and not self.children
        self.children = [Node(p_prior[i], self, -self.color) for i in range(225)]

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
            self.parent.backup(-gamma * value, gamma, use_virtual)    # switch sign because of the switch of color

class MCTS:

    def __init__(self, config, black_net, white_net, color, stochastic_steps):

        """
        Parameters:
        -----------
        config: dict, config info
        black_net: CNN model, evaluation model for black player
        white_net: CNN model, evaluation model for white player
        color: -1 or +1, the color of the MCTS agent, +1 for black, -1 for white, only counts when not self-playing
        stochastic_steps: int, total stones played by stochastic methods, 0 for deterministic, >0 for stochastic (training only)
        
        """

        self.root = Node(1.0, None, 1)    # start from empty board
        self.board = np.zeros(225)    # board info vector for root node
        self.last_move = None    # update the last move in the root board
        self.c_puct = config['c_puct']
        self.simulation_times = config['simulation_times']
        self.tau_init = config['tau_init']    # initial temperature
        self.tau_decay = config['tau_decay']
        self.self_play = config['self_play']    # whether play with itself or another agent
        self.gamma = config['gamma']
        self.num_threads = config['num_threads']
        self.black_net = black_net
        self.white_net = white_net
        self.color = color
        self.stochastic_steps = stochastic_steps
        self._expanding_list = []

    def _simulate_one_step(self):
        
        legal_moves = self.board == 0
        node = self.root
        color = node.color    # the color of the root node may be different when the board is empty
        board = np.copy(self.board)
        action = self.last_move
        while node.children:    # select
            node, action = node.select(self.c_puct, legal_moves)
            legal_moves[action] = False
            board[action] = color
            color = -color
        if node.is_end:    # evaluated end node
            node.backup(node.Q, self.gamma)
            return
        if action is not None:    # if action is None, the node is not an end node
            game_result = check_result(board, action)    #### a function judging the game results, returning "blackwin", "whitewin", "draw", or "unfinished"
            if game_result != "unfinished":
                node.is_end = True
                if game_result == "draw":
                    node.backup(0, self.gamma)
                else:
                    node.backup(-1, self.gamma)    # the color of an ended node must have lost the game
                return
        net = self.black_net if color == 1 else self.white_net    # evaluate for nonfinished node
        p_prior, v = net.predict(    #### may change according to the net
            board=board,
            last_move=action
        )
        node.expand(p_prior)    # expand
        node.backup(v, self.gamma)    # backup

    def simulate(self, num_steps):

        """
        Parameters:
        -----------
        num_steps: int, number of simulation steps

        """

        # print("simulating:")
        if self.num_threads == 1:
            for _ in range(num_steps):
                self._simulate_one_step()
                # print(f"{_ + 1} step finished.")
        elif self.num_threads > 1:
            self._simulate(num_steps)
        else:
            raise RuntimeError(f"Invalid thread number: {self.num_threads}.")

    def action(self, board, last_move):

        """
        Parameters:
        -----------
        board: 225-d array, temp board state (may differ from the history info if not self-play)
        last_move: int, last move index in board

        Returns:
        --------
        action: int, where to place stone
        p: 225-d array, frequency of each choice, for training use

        """

        stage = np.sum(np.abs(board)) + 1
        # We need to rebase our root node to the board first, because of new stones being placed
        if not self.root.children and last_move is not None:    # the root is a leaf node, and the board is not empty
            self.simulate(self.simulation_times)
        if last_move is not None:    # we only need to rebase when the board is not empty
            self.root, self.board, self.last_move = self.root.children[last_move], board, last_move    # rebase
        self.simulate(self.simulation_times)    # simulate
        N_list = np.array([child.N * 1.0 for child in self.root.children])
        if stage > self.stochastic_steps:    # determininstic policy
            action = np.argmax(N_list)
        else:    # stochastic policy
            tau = max(self.tau_init * (self.tau_decay ** (stage // 2)), 0.04)
            pi = N_list ** (1 / tau)
            pi /= np.sum(pi)
            action = np.random.choice([i for i in range(225)], p=pi)
        N_list /= np.sum(N_list)    # for training use
        if not self.self_play:    # while not self-playing, change root node to the opposite color
            self.board[action] = self.root.color
            self.root = self.root.children[action]
            self.last_move = action
        return action, N_list
    
    def reset(self):

        self.root = Node(1.0, None, 1)
        self.board = np.zeros(225)
        self.last_move = None

    def _get_simulate_thread_target(self, num_steps, num_threads):

        def _simulate_thread():
            
            avg_steps = int(num_steps / num_threads)
            for _ in range(avg_steps):
                legal_moves = self.board == 0
                node = self.root
                color = node.color    # the color of the root node may be different when the board is empty
                board = np.copy(self.board)
                action = self.last_move
                while node.children:    # select
                    node, action = node.select(self.c_puct, legal_moves)
                    node.select_num += 1
                    node.N += 10    # virtual loss
                    legal_moves[action] = False
                    board[action] = color
                    color = -color
                    while node in self._expanding_list:
                        time.sleep(1e-4)
                if node.is_end:    # evaluated end node
                    node.backup(node.Q, self.gamma, True)
                    continue
                if node not in self._expanding_list:
                    self._expanding_list.append(node)
                else:
                    continue
                if action is not None:    # if action is None, the node is not an end node
                    game_result = check_result(board, action)    #### a function judging the game results, returning "blackwin", "whitewin", "draw", or "unfinished"
                    if game_result != "unfinished":
                        node.is_end = True
                        if game_result == "draw":
                            node.backup(0, self.gamma, True)
                        else:
                            node.backup(-1, self.gamma, True)    # the color of an ended node must have lost the game
                        self._expanding_list.remove(node)
                        continue
                net = self.black_net if color == 1 else self.white_net    # evaluate for nonfinished node
                p_prior, v = net.predict(    #### may change according to the net
                    board=board,
                    last_move=action
                )
                node.expand(p_prior)    # expand
                node.backup(v, self.gamma, True)    # backup
                self._expanding_list.remove(node)
            
        return _simulate_thread
    
    def _simulate(self, num_steps):

        target = self._get_simulate_thread_target(num_steps, self.num_threads)
        thread_list = []
        for i in range(self.num_threads):
            thr = threading.Thread(target=target, name=f"thread_{i + 1}")
            thr.start()
            thread_list.append(thr)
            time.sleep(1e-3)
        for thr in thread_list:
            thr.join()
