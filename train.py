from dataset import *
from agent import *
import torch
import torch.optim as optim


def train_net(color, netpath, version, datapath, epochs, device, savepath):
    config = {'color': color, 'learning_rate': 2e-3, 'momentum': 9e-1, 'l2': 1e-5, 'batch_size': 64, 'path': netpath, 'version': version}
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


def train_on_mentor_selfplay(start_version, rounds=1):
    for i in range(rounds):
        version = start_version + i
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        train_net(1, "./models1/black", version, "./gamedata/mentor/black", 50, device, "./models1/black")
        train_net(-1, "./models1/white", version, "./gamedata/mentor/white", 50, device, "./models1/white")


def train_mcts_mentor(start_version, rounds=1):
    for i in range(rounds):
        version = start_version + i
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        train_net(1, "./models1/black", version, "./gamedata/mctc/mentor/black", 50, device, "./models1/black")
        train_net(-1, "./models1/white", version, "./gamedata/mctc/mentor/white", 50, device, "./models1/white")


def train_mcts_mcts(start_version, rounds=1):
    for i in range(rounds):
        version = start_version + i
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        train_net(1, "./models1/black", version, "./gamedata/mcts/mcts/black", 50, device, "./models1/black")
        train_net(-1, "./models1/white", version, "./gamedata/mcts/mcts/white", 50, device, "./models1/white")


if __name__ == "__main__":

    # mentorai_selfplay("./gamedata/mentor", 1000)

    # A = torch.load("./gamedata/mentor/white/z_record.hyt")
    # print((A > 0).sum(), (A == 0).sum(), (A < 0).sum())

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_net(1, "./models/black", 0, "./gamedata/enhanced/black", 50, device, "./models/black")
    train_net(-1, "./models/white", 0, "./gamedata/enhanced/white", 50, device, "./models/white")

    # MCTS_vs_mentor("./models", 0)

    # ai_mcts = MCTSAgent({'c_puct': 5, 'simulation_times': 1600, 'tau_init': 1, 'tau_decay': 0.8, 'self_play': False, 'gamma': 0.95, 'num_threads': 8}, './models1', 0, 1, 6, torch.device('cuda'))
    # ai_mentor = MentorAgent(1)
    # print(ai_playing(ai_mcts, ai_mentor, True))

    # ai_mcts = MCTSAgent({'c_puct': 5, 'simulation_times': 1600, 'tau_init': 1, 'tau_decay': 0.8, 'self_play': False, 'gamma': 0.95, 'num_threads': 8}, './models1', 0, -1, 6, torch.device('cuda'))
    # ai_mentor = MentorAgent(1)
    # print(ai_playing(ai_mentor, ai_mcts, True))

    # train_on_mentor_selfplay(0, 5)
