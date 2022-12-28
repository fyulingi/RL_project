from model import *
from dataset import *
from agent import *
import torch
import torch.optim as optim
from utils import printboard


def train_net(color, netpath, version, datapath, epochs, device, savepath):
    config = {'color': color, 'learning_rate': 1e-2, 'momentum': 9e-1, 'l2': 1e-5, 'batch_size': 32, 'path': netpath, 'version': version}
    model = GomokuNet(config)
    # optimizer = optim.SGD(model.parameters(), lr=model.learning_rate, momentum=model.momentum, weight_decay=model.l2, nesterov=True)
    optimizer = optim.Adam(model.parameters(), lr=model.learning_rate, weight_decay=model.l2)
    train_data = Dataset()
    train_data.collect_data(datapath)
    train_data.board = train_data.board[:25600]
    train_data.last_move = train_data.last_move[:25600]
    train_data.p = train_data.p[:25600]
    train_data.z = train_data.z[:25600]
    print(f"loss: {train_GomokuNet(model, optimizer, train_data, epochs, device)}")
    torch.save(model.state_dict(), savepath + f'/version_{version + 1}.model')


def MCTS_vs_mentor(path, version):
    ai_mcts = MCTSAgent({'c_puct': 5, 'simulation_times': 1600, 'tau_init': 1, 'tau_decay': 0.8, 'self_play': False, 'gamma': 0.95}, path, version, 1, 6)
    ai_mentor = MentorAgent(1)

    mcts_win = mentor_win = 0

    print("MCTS black v.s. Mentor white:")
    for i in range(50):
        ai_mcts.ai.reset()
        res = ai_playing(ai_mcts, ai_mentor)
        print(f"Game {i}, " + res)
        if res == "blackwin":
            mcts_win += 1
        elif res == "whitewin":
            mentor_win += 1
    print("Mentor black v.s. MCTS white:")
    for i in range(50):
        ai_mcts.reset()
        res = ai_playing(ai_mentor, ai_mcts)
        print(f"Game {i}, " + res)
        if res == "blackwin":
            mentor_win += 1
        elif res == "whitewin":
            mcts_win += 1
    print(f"MCTS wins {mcts_win}, loses {mentor_win}.")

    
def ai_playing(ai1, ai2, verbose=False):

    board = np.zeros(225)
    game_result = "unfinished"
    last_move = None
    color = 1
    while game_result == "unfinished":
        if color == 1:
            action = ai1.play(board, last_move)
        else:
            action = ai2.play(board, last_move)
        if verbose:
            print(("black " if color == 1 else "white ") + f"action: ({action // 15}, {action % 15})")
        board[action] = color
        last_move = action
        game_result = check_result(board, last_move)
        printboard(board)
        color = -color
    return game_result


def train_on_mentor_selfplay(start_version, rounds=1):
    for i in range(rounds):
        version = start_version + i
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        train_net(1, "./models1/black", version, "./gamedata/mentor/black", 50, device, "./models1/black")
        train_net(-1, "./models1/white", version, "./gamedata/mentor/white", 50, device, "./models1/white")


if __name__ == "__main__":

    # mentorai_selfplay("./gamedata/mentor", 1000)

    # A = torch.load("./gamedata/mentor/white/z_record.hyt")
    # print((A > 0).sum(), (A == 0).sum(), (A < 0).sum())

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_net(1, "./models/black", -1, "./gamedata/enhanced/black", 50, device, "./models/black")
    train_net(-1, "./models/white", -1, "./gamedata/enhanced/white", 50, device, "./models/white")

    # MCTS_vs_mentor("./models", 0)

    # ai_mcts = MCTSAgent({'c_puct': 5, 'simulation_times': 1600, 'tau_init': 1, 'tau_decay': 0.8, 'self_play': False, 'gamma': 0.95, 'num_threads': 8}, './models1', 0, 1, 6, torch.device('cuda'))
    # ai_mentor = MentorAgent(1)
    # print(ai_playing(ai_mcts, ai_mentor, True))

    # ai_mcts = MCTSAgent({'c_puct': 5, 'simulation_times': 1600, 'tau_init': 1, 'tau_decay': 0.8, 'self_play': False, 'gamma': 0.95, 'num_threads': 8}, './models1', 0, -1, 6, torch.device('cuda'))
    # ai_mentor = MentorAgent(1)
    # print(ai_playing(ai_mentor, ai_mcts, True))

    # train_on_mentor_selfplay(0, 5)
