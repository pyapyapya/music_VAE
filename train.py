from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from config import h_params
from data_pipeline import DataPipeline
from datasets import get_dataloader
from model import BidirectionalLstmEncoder
from preprocessor import CSVParser


def train(data_loader):
    """
    [TODO] Implement Loss (ELBO, KL-Divergence, Inverse-Sigmoid)
    [TODO] Implement Validation
    [TODO] Implement TensorBoard
    [TODO] Implement Training Process

    :param data_loader: input_data [batch_size, channel, time_step]
    :return:
    """

    epochs = 100
    bilstm = BidirectionalLstmEncoder().cuda()
    optimizer = Adam(params=bilstm.parameters(),
                     lr=h_params['lr'])
    lr_scheduler = ExponentialLR(optimizer,
                                 gamma=h_params['lr_schedule'])

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            batch = batch.cuda()
            batch = batch.permute(0, 2, 1)
            output, _ = bilstm(batch)
            return


def main():
    train_data, val_data, test_data = CSVParser().parse()

    train_data = DataPipeline(train_data).process()
    val_data = DataPipeline(val_data).process()
    test_data = DataPipeline(test_data).process()

    train_dataloader = get_dataloader(train_data,
                                      batch_size=h_params['batch_size'],
                                      shuffle=True,
                                      pin_memory=True,
                                      num_worker=0)

    val_dataloader = get_dataloader(val_data,
                                    batch_size=1,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_worker=0)

    train(data_loader=train_dataloader)


if __name__ == '__main__':
    main()
