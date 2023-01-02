
def get_mcts_config(mode):
    mcts_config = {'c_puct': 5, 'tau_init': 1, 'tau_decay': 0.8,
                   'gamma': 0.95, 'num_threads': 1, 'stochastic_steps': 0}
    if mode == 'train':
        mcts_config['simulation_times'] = 1000
    elif mode == 'test':
        mcts_config['simulation_times'] = 200
    else:
        raise ValueError('mode must be train or test')
    return mcts_config


def get_model_config(mode, path, version):
    model_config = {'path': path, 'version': version}
    if mode == 'train':
        model_config['device'] = 'cuda'
    elif mode == 'test':
        model_config['device'] = 'cuda'
    elif mode == 'compete':
        model_config['device'] = 'cpu'
    else:
        raise ValueError('mode must be train, test or compete')
    return model_config
