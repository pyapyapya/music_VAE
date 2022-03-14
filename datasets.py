import os
from pickle import load
from typing import Union

from numpy import array, load
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from config import PATH


class MIDIDataset(Dataset):
    """
    Pytorch framework를 이용하여 Model을 훈련하기 위한 Dataset 구축하는 클래스
    """
    def __init__(self, sequence_bar: Union[array, array, array]):
        self.dir_path = PATH['DIR_PATH']
        self.sequence_bar: Union[array, array, array] = sequence_bar

    def __len__(self) -> int:
        return self.sequence_bar.shape[0]

    def __getitem__(self, idx: int) -> Tensor:
        sequence: Tensor = Tensor(self.sequence_bar[idx])
        return sequence

    def load_dataset(self) -> Union[array, array, array]:
        file_name = 'subsequence_record.npy'
        sequence_bar: Union[array, array, array] = load(os.path.join(self.dir_path, file_name))
        return sequence_bar


def get_dataloader(data, batch_size=1, shuffle=False, pin_memory=True, num_worker=4) -> DataLoader:
    dataset: MIDIDataset = MIDIDataset(data)
    data_loader: DataLoader = DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         pin_memory=pin_memory,
                                         num_workers=num_worker)
    return data_loader
