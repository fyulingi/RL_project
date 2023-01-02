import os
import torch
import torch.optim as optim
import numpy as np

import config
from dataset import *
from agent import *
from compete import move_one_step


def get_loss(model, net_input, p, z):
    # print(f"net_input shape: {net_input.shape}")
    p_hat, z_hat = model(net_input)
    # print("p hat shape: ", p_hat.shape)
    # print("z hat shape: ", z_hat.shape)
    # print("p shape: ", p.shape)
    # print("z shape: ", z.shape)
    loss = -torch.mean(p_hat.mul(torch.log(p+1e-10))) + torch.mean((z_hat - z) ** 2)
    # print(f"loss: {loss}")
    # print("loss shape: ", loss.shape)
    return loss


def train_on_data(model, optimizer, train_data_loader, epochs, device):
    for epoch in range(epochs):
        for i, (net_input, p, z) in enumerate(train_data_loader):
            net_input = net_input.to(device)
            p = p.to(device)
            z = z.to(device)
            optimizer.zero_grad()
            loss = get_loss(model, net_input, p, z)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"epoch: {epoch}, iter: {i}, loss: {loss}")


def save_model(model, version, type):
    if not os.path.exists("./models/" + type):
        os.makedirs("./models/" + type)
    torch.save(model.state_dict(), "./models/" + type + "/model_" + str(version))


def train_on_gen_data(batch_size, epochs, device):
    model_config = config.get_model_config("train", "./models/gen_data", 0)
    model = GomokuNet(model_config)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_data_loader = Dataset("./game_data/gen_data", "read").get_loader(batch_size)
    train_on_data(model, optimizer, train_data_loader, epochs, device)
    save_model(model, 1, "gen_data")


def get_black_white_round_num(rounds):
    black_num = np.random.randint(1, rounds)
    white_num = rounds - black_num
    return black_num, white_num


def get_reward_by_result(game_result, color):
    if game_result == "draw":
        return 0
    elif (game_result == "black_win" and color == 1) or (game_result == "white_win" and color == -1):
        return 1
    else:
        return -1


def gen_mentor_mcts_data(model, rounds, mu, version):
    black_num, white_num = get_black_white_round_num(rounds)
    mcts_config = config.get_mcts_config("train")
    dataset = Dataset(f"./game_data/with_mentor/{version}", "write")
    for black_round in range(black_num):
        board = np.zeros(225)
        actor1 = MCTSAgent(1, board, mcts_config, None, model)
        actor2 = MentorAgent(-1, board)
        boards, last_moves, ps, zs, colors = [], [], [], [], []
        last_move = -1
        turn = 0
        while check_result(board, last_move) == "unfinished":
            if turn % 2 == 0:
                boards.append(board.copy())
                last_moves.append(last_move)
                last_move, prob = actor1.play()
                ps.append(prob)
                colors.append(1)
                move_one_step(1, last_move, board, actor1, actor2)
            else:
                last_move = actor2.next_action()
                move_one_step(-1, last_move, board, actor1, actor2)
            turn += 1
        reward = get_reward_by_result(check_result(board, last_move), 1)
        zs.append(reward)
        while len(zs) < len(boards):
            zs.append(mu * zs[-1])
        zs = zs[::-1]
        dataset.extend(boards, last_moves, ps, zs, colors)
    for white_round in range(white_num):
        board = np.zeros(225)
        actor1 = MentorAgent(1, board)
        actor2 = MCTSAgent(-1, board, mcts_config, None, model)
        boards, last_moves, ps, zs, colors = [], [], [], [], []
        last_move = -1
        turn = 0
        while check_result(board, last_move) != "unfinished":
            if turn % 2 == 0:
                last_move = actor1.next_action()
                move_one_step(1, last_move, board, actor1, actor2)
                # print(f"black: {x}, {y}")
            else:
                boards.append(board.copy())
                last_moves.append(last_move)
                last_move, prob = actor2.play()
                ps.append(prob)
                colors.append(-1)
                move_one_step(-1, last_move, board, actor1, actor2)
                # print(f"white: {x}, {y}")
        reward = get_reward_by_result(check_result(board, last_move), -1)
        zs.append(reward)
        while len(zs) < len(boards):
            zs.append(mu * zs[-1])
        zs = zs[::-1]
        dataset.extend(boards, last_moves, ps, zs, colors)
    dataset.save()
    return dataset


def train_on_mentor_play(version, rounds, batch_size, epochs, device):
    model_config = config.get_model_config("train", "./models/with_mentor" if version > 0 else "./models/gen_data",
                                           version if version > 0 else 1)
    model = GomokuNet(model_config)
    with torch.no_grad():
        train_data_loader = gen_mentor_mcts_data(model, rounds, 0.9, version).get_loader(batch_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_on_data(model, optimizer, train_data_loader, epochs, device)
    save_model(model, version + 1, "with_mentor")


def gen_self_play_data(model, rounds, mu, version):
    black_num, white_num = get_black_white_round_num(rounds)
    mcts_config = config.get_mcts_config('train')
    dataset = Dataset(f"./game_data/self_play/{version}", "write")
    for black_round in range(black_num):
        board = np.zeros(225)
        actor1 = MCTSAgent(1, board, mcts_config, None, model)
        actor2 = MCTSAgent(-1, board, mcts_config, None, model)
        boards, last_moves, ps, zs, colors = [], [], [], [], []
        last_move = -1
        turn = 0
        while check_result(board, last_move) == "unfinished":
            if turn % 2 == 0:
                boards.append(board.copy())
                last_moves.append(last_move)
                last_move, prob = actor1.play()
                ps.append(prob)
                colors.append(1)
                move_one_step(1, last_move, board, actor1, actor2)
                # print(f"black: {x}, {y}")
            else:
                last_move = actor2.next_action()
                move_one_step(-1, last_move, board, actor1, actor2)
                # print(f"white: {x}, {y}")
            turn += 1
        reward = get_reward_by_result(check_result(board, last_move), 1)
        zs.append(reward)
        while len(zs) < len(boards):
            zs.append(mu * zs[-1])
        zs = zs[::-1]
        dataset.extend(boards, last_moves, ps, zs, colors)
    for white_round in range(white_num):
        board = np.zeros(225)
        actor1 = MCTSAgent(1, board, mcts_config, None, model)
        actor2 = MCTSAgent(-1, board, mcts_config, None, model)
        boards, last_moves, ps, zs, colors = [], [], [], [], []
        last_move = -1
        turn = 0
        while check_result(board, last_move) != "unfinished":
            if turn % 2 == 0:
                last_move = actor1.next_action()
                move_one_step(1, last_move, board, actor1, actor2)
                # print(f"black: {x}, {y}")
            else:
                boards.append(board.copy())
                last_moves.append(last_move)
                last_move, prob = actor2.play()
                ps.append(prob)
                colors.append(-1)
                move_one_step(-1, last_move, board, actor1, actor2)
                # print(f"white: {x}, {y}")
        reward = get_reward_by_result(check_result(board, last_move), -1)
        zs.append(reward)
        while len(zs) < len(boards):
            zs.append(mu * zs[-1])
        zs = zs[::-1]
        dataset.extend(boards, last_moves, ps, zs, colors)
    dataset.save()
    return dataset


def train_on_self_play(version, rounds, batch_size, epochs, device):
    model_config = config.get_model_config("train", "./models/self_play" if version > 0 else "./models/with_mentor",
                                           version if version > 0 else 1)
    model = GomokuNet(model_config)
    with torch.no_grad():
        train_data_loader = gen_self_play_data(model, rounds, 0.9, version).get_loader(batch_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_on_data(model, optimizer, train_data_loader, epochs, device)
    save_model(model, version + 1, "self_play")


def train(mentor_play_round, self_play_rounds):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("begin training on gen data......")
    train_on_gen_data(64, 120, device)
    for round in range(mentor_play_round):
        print(f"begin training on mentor play data, round {round + 1}......")
        train_on_mentor_play(round, 64, 64, 100, device)
    for round in range(self_play_rounds):
        print(f"begin training on self play data, round {round + 1}......")
        train_on_self_play(round, 64, 64, 100, device)


if __name__ == "__main__":
    train(1000, 1000)
