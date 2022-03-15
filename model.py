from torch import nn, Tensor, cat


class BidirectionalLstmEncoder(nn.Module):
    """
    input sequence data를 Bi-LSTM을 이용해 hidden state vector를 추출
    hidden state vector를 결합시켜 두 개의 latent vector (mu, sigma)를 학습
    """
    def __init__(self,
                 input_size: int = 9,
                 hidden_size: int = 2048,
                 latent_size: int = 512,
                 num_layers: int = 2):
        super(BidirectionalLstmEncoder, self).__init__()

        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.latent_size: int = latent_size
        self.num_layers: int = num_layers

        # Bi-LSTM
        self.bilstm = nn.LSTM(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              bidirectional=True,
                              batch_first=True).cuda()

        # two Latent Vector
        self.mu = nn.Linear(in_features=hidden_size * 2,
                            out_features=latent_size).cuda()

        self.sigma = nn.Linear(in_features=hidden_size * 2,
                               out_features=latent_size).cuda()

        self.softplus = nn.Softplus().cuda()

    def forward(self, input_enc: Tensor):
        batch_size = input_enc.shape[0]
        # Get Final State Vector in Bi-LSTM hidden
        _, (h_n, c_n) = self.bilstm(input_enc)
        h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_size)
        h_t = cat((h_n[:, 0], h_n[:, 1]), dim=-1)
        mu = self.mu(h_t)
        sigma = self.softplus(self.sigma(h_t))
        return mu, sigma
