PATH = {
    'DIR_PATH': './data/groove/'
}

h_params = {
    'lr': 1e-3,
    'lr_schedule': 0.99999,
    'batch_size': 64,
    'max_seq_len': 64,
    'z_size': 512,
    'enc_rnn_size': [2048],
    'dec_rnn_size': [2048, 2048, 2048],
    'free_bits': 48,
    'max_beta': 1.2,
    'sampling_schedule': 'inverse_sigmoid',
    'sampling_rate': 1000
}