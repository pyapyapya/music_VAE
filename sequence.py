import os
from pickle import dump, load
from typing import List, Union

from numpy import array
from tqdm import tqdm

from config import PATH


class SplitSequence:
    """
    MIDI Note Sequence를 Sub_Sequence로 나누는 작업
    slice_bar는 자를 마디의 개수를 의미 Default: 4
    ticks는 마디 내에서 기준으로 할 박자를 무엇으로 할지를 의미 Default: 8

    Max_Sequence는 slice_bar와 ticks를 곱함으로써, 총 Notes Sequence를 가짐

    1. sequence 생성
    2. sub_sequence로 배치 크기를 맞추기 위해 last index drop
    3. 3d-array 배치를 쌓기 위해 list -> numpy로 변환

    """

    def __init__(self, slice_bar: int = 4, ticks: int = 8):
        self.slice_bar: int = slice_bar
        self.ticks: int = ticks
        self.max_sequence: int = self.slice_bar * self.ticks

        self.dir_path = PATH["DIR_PATH"]

        self.record: Union[array, array, array] = self.load_midi_pkl()
        self.sub_sequence_bar: array = self.make_subsequence_bar()

    def make_subsequence_bar(self) -> array:
        sub_sequence_bar: List[array, array] = []
        for midi_score in tqdm(self.record):
            sequence = midi_score.shape[1]
            if sequence > self.max_sequence:
                midi_score = self._cut_sequence(midi_score, sequence)
                split_midi_score = self.split_subsequence(midi_score)

                for batch_idx in range(len(split_midi_score)):
                    sub_sequence_bar.append(array(split_midi_score[batch_idx]))

        return array(sub_sequence_bar)

    def split_subsequence(self, midi_score: array) -> List[Union[array, array, array]]:
        """
        2d array -> 3d array로 배치형태로 구성. 이후 list로 변환
        :param midi_score: [9 channel, midi_max_sequence]로 구성된 array
        :return: 3d array -> 3d list
        """
        midi_score: List[Union[array, array, array]] = midi_score.reshape(
            -1, 9, self.max_sequence
        ).tolist()
        return midi_score

    def _cut_sequence(self, midi_score: array, sequence: int) -> array:
        """
        sub_sequence로 배치 크기를 맞추기 위해 last index drop
        :param midi_score:
        :param sequence: 각 midi_score의 sequence 길이
        :return: 마지막 sequence 부분이 없는 midi_score
        """
        seq_mod = sequence % self.max_sequence
        if seq_mod != 0:
            midi_score = midi_score[:, :-seq_mod]
        return midi_score

    def load_midi_pkl(self) -> List[Union[array, array]]:
        file_name = "record.pkl"
        file_path = os.path.join(self.dir_path, file_name)
        with open(file_path, "rb") as pkl:
            record: List[array, array] = load(pkl)
        return record

    def save_subsequence_pkl(self):
        file_name = "subsequence_record.pkl"
        file_path = os.path.join(self.dir_path, file_name)

        with open(file_path, "wb") as pkl:
            dump(self.sub_sequence_bar, pkl)
