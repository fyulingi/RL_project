import time
import threading
import numpy as np
from utils import check_result
from strategy.Node import Node


class MCTS:
    def __init__(self, config, black_net, white_net, color, board):
        """
        Parameters:
        -----------
        black_net: CNN model, evaluation model for black player
        white_net: CNN model, evaluation model for white player
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
        self.self_play = config['self_play']    # whether play with itself or another agent
        self.gamma = config['gamma']
        self.num_threads = config['num_threads']
        self.black_net = black_net
        self.white_net = white_net
        self.color = color
        self.stochastic_steps = config['stochastic_steps']
        self._expanding_list = []

    def expand_one_node_determined(self, node, x, y):
        new_node = Node(1, node, -node.color, x*15+y)
        node.children.append(new_node)

    def move_one_step(self, x, y, color):
        if len(self.root.children) == 0:
            self.expand_one_node_determined(self.root, x, y)
        for child in self.root.children:
            if child.move == x*15+y:
                self.root = child
                self.root.N += 1
                self.last_move = x*15+y
                return
        raise RuntimeError("Invalid move.")

    def update_root(self, move_x, move_y):
        action = move_x * 15 + move_y
        # todo: if child is empty list
        for child in self.root.children:
            if child.move == action:
                self.root = child
                self.root.N += 1
                return
        node = Node(1, None, -self.root.color, action)
        self.root = node
        self.root.parent = None
        self.root.N += 1

    def simulation_one_game(self):
        board = np.copy(self.board)
        legal_moves = (board==0)
        node = self.root
        while node.children != [] and check_result(board, node.move//15, node.move%15) == "unfinished":
            node, _ = node.select(self.c_puct)
            node.N += 1
            board[node.move] = node.color
            legal_moves[node.move] = False
        if check_result(board, node.move//15, node.move%15) != "unfinished":
            node.is_end = True
            node.backup(node.Q, self.gamma)
            return
        game_result = check_result(board, node.move//15, node.move%15)
        if game_result == "draw":
            node.backup(0, self.gamma)
        elif (game_result == "blackwin" and self.color == 1) or (game_result == "whitewin" and self.color == -1):
            node.backup(1, self.gamma)
        elif game_result == "unfinished":
            p_prior, v = self.get_p_prior_v(board, -node.color, node.move)
            node.backup(v, self.gamma)
            node.expand(p_prior, legal_moves)
        else:  # the color of an ended node must have lost the game
            node.backup(-1, self.gamma)

    def get_p_prior_v(self, board, color, last_move):
        net = self.black_net if color == 1 else self.white_net
        p_prior, v = net.predict(board, last_move)
        return p_prior, v

    def simulate(self, num_steps):
        if self.num_threads == 1:
            for step in range(num_steps):
                self.simulation_one_game()
                # self.simulate_one_step()
                # print(f"{_ + 1} step finished.")
        # elif self.num_threads > 1:
        #     self.simulate_multi_threads(num_steps)
        # else:
        #     raise RuntimeError(f"Invalid thread number: {self.num_threads}.")

    def action(self):
        stage = np.sum(np.abs(self.board)) + 1
        if stage == 1:
            return 7, 7, None
        self.simulate(self.simulation_times)    # simulate
        N_list = np.array([child.N * 1.0 for child in self.root.children])
        if stage > self.stochastic_steps:    # determininstic policy
            action = self.root.children[np.argmax(N_list)].move
        else:    # stochastic policy
            tau = max(self.tau_init * (self.tau_decay ** (stage // 2)), 0.04)
            pi = N_list ** (1 / tau)
            pi /= np.sum(pi)
            action = self.root.children[np.random.choice([i for i in range(len(self.root.children))], p=pi)].move
        N_list /= np.sum(N_list)    # for training use
        return action//15, action%15, N_list

    def update(self, action):
        self.root = self.root.children[action]
        self.root.N += 1
        self.last_move = action

    def reset(self):
        self.root = Node(1.0, None, -self.color, -1)  # start from empty board
        self.last_move = None  # update the last move in the root board

    #
    # def _get_simulate_thread_target(self, num_steps, num_threads):
    #     def _simulate_thread():
    #         avg_steps = int(num_steps / num_threads)
    #         for _ in range(avg_steps):
    #             legal_moves = self.board == 0
    #             node = self.root
    #             color = node.color    # the color of the root node may be different when the board is empty
    #             board = np.copy(self.board)
    #             action = self.last_move
    #             while node.children:    # select
    #                 node, action = node.select(self.c_puct, legal_moves)
    #                 node.select_num += 1
    #                 node.N += 10    # virtual loss
    #                 legal_moves[action] = False
    #                 board[action] = color
    #                 color = -color
    #                 while node in self._expanding_list:
    #                     time.sleep(1e-4)
    #             if node.is_end:    # evaluated end node
    #                 node.backup(node.Q, self.gamma, True)
    #                 continue
    #             if node not in self._expanding_list:
    #                 self._expanding_list.append(node)
    #             else:
    #                 continue
    #             if action is not None:    # if action is None, the node is not an end node
    #                 game_result = check_result(board, action//15, action%15)    #### a function judging the game results, returning "blackwin", "whitewin", "draw", or "unfinished"
    #                 if game_result != "unfinished":
    #                     node.is_end = True
    #                     if game_result == "draw":
    #                         node.backup(0, self.gamma, True)
    #                     else:
    #                         node.backup(-1, self.gamma, True)    # the color of an ended node must have lost the game
    #                     self._expanding_list.remove(node)
    #                     continue
    #             net = self.black_net if color == 1 else self.white_net    # evaluate for nonfinished node
    #             p_prior, v = net.predict(    #### may change according to the net
    #                 board=board,
    #                 last_move=action
    #             )
    #             node.expand(p_prior)    # expand
    #             node.backup(v, self.gamma, True)    # backup
    #             self._expanding_list.remove(node)
    #
    #     return _simulate_thread
    #
    # def simulate_multi_threads(self, num_steps):
    #     target = self._get_simulate_thread_target(num_steps, self.num_threads)
    #     thread_list = []
    #     for i in range(self.num_threads):
    #         thr = threading.Thread(target=target, name=f"thread_{i + 1}")
    #         thr.start()
    #         thread_list.append(thr)
    #         time.sleep(1e-3)
    #     for thr in thread_list:
    #         thr.join()
