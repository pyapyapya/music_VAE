from typing import Tuple

from torch import Tensor, cat, nn, rand_like


class BidirectionalLstmEncoder(nn.Module):
    """
    input sequence data를 Bi-LSTM을 이용해 hidden state vector를 추출
    hidden state vector를 결합시켜 두 개의 latent vector (mu, sigma)를 학습
    """

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 2048,
        latent_size: int = 512,
        num_layers: int = 2,
    ):
        super(BidirectionalLstmEncoder, self).__init__()

        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.latent_size: int = latent_size
        self.num_layers: int = num_layers

        # Bi-LSTM
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        ).cuda()

        # two Latent Vector
        self.mu = nn.Linear(
            in_features=hidden_size * 2, out_features=latent_size
        ).cuda()

        self.sigma = nn.Linear(
            in_features=hidden_size * 2, out_features=latent_size
        ).cuda()

        self.softplus = nn.Softplus().cuda()

    def forward(self, input_enc: Tensor) -> Tuple[Tensor, Tensor]:
        # input shape [batch_size, seq_len, input_size]
        batch_size = input_enc.shape[0]

        # Get Final State Vector in Bi-LSTM hidden
        # h_n shape [num_layers*bi-direction, batch_size, hidden_size]
        _, (h_n, c_n) = self.bilstm(input_enc)

        # h_n view shape [num_layer, num_direction, batch_size, hidden_num]
        h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_size)

        # concat h_t [num_layer, batch_size, hidden_size * num_direction]
        h_t = cat((h_n[:, 0], h_n[:, 1]), dim=-1)

        # latent vector shape [num_layer, batch_size, hidden_size]
        mu = self.mu(h_t)
        sigma = self.softplus(self.sigma(h_t))
        return mu, sigma


class Conductor(nn.Module):
    def __init__(self, latent_size: int = 512,
                 hidden_size: int = 1024,
                 output_size: int = 512):
        super(Conductor, self).__init__()
        self.latent_embedding = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.Tanh()
        )

    def forward(self, latent_vector):
        z = self.latent_embedding(latent_vector)
        self.init_hidden(z)
        pass

    def init_hidden(self, z):
        pass


class StackLSTMDecoder(nn.Module):
    """
    Stack Decoder
    Latent Vector를 Input으로 한 Stack Decoder 구현

    논문에서는 2-bar 크기의 sequence는 Flat-Decoder RNN으로 구현함
    16-bar 크기의 sequence는 Hierarchical-Decoder RNN으로 구현함
    """

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 1024,
        output_size: int = 512,
        num_layers: int = 2,
    ):
        super(StackLSTMDecoder, self).__init__()

    def forward(self, x, teacher_forcing_ratio: float = 0.5):
        pass


class MusicVAE(nn.Module):
    """
    Encoder -> Latent Vector -> Conductor -> Decoder -> Output 프로세스를 수행
    Input Data에 따른 최종 Output 모델

    [TODO] Implement Conductor
    """
    def __init__(self):
        super(MusicVAE, self).__init__()

        self.encoder = BidirectionalLstmEncoder()
        self.conductor = Conductor()

        pass

    def forward(self, input_data):
        mu, sigma = self.encoder(input_data)
        eps = rand_like(sigma, requires_grad=False)
        latent_vector = mu + (sigma * eps)
        pass
