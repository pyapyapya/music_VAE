import os
from typing import Dict, List, Union

from numpy import array, load

from preprocessor import MIDIParser
from sequence import SplitSequence
from config import PATH


class DataPipeline:
    """
    데이터 전처리 과정들을 통합시켜 처리하는 클래스
    """
    def __init__(
        self,
        data_info: List[Dict[str, str]],
        time_numerate: int = 4,
        time_denominator: int = 4,
        slice_bar: int = 4,
        ticks: int = 8,
    ):

        self.data_info = data_info
        self.time_numerate: int = time_numerate
        self.time_denominator: int = time_denominator
        self.slice_bar: int = slice_bar
        self.ticks: int = ticks

    def process(self) -> Union[array, array, array]:
        file_name = 'subsequence_record.npy'
        sub_sequence_file = os.path.join(PATH['DIR_PATH'], file_name)
        if os.path.exists(sub_sequence_file):
            return load(sub_sequence_file)

        record: List[array, array] = self.get_parse_midi()
        sub_sequence: Union[array, array, array] = self.get_split_sequence(record)
        return sub_sequence

    def get_parse_midi(self) -> List[Union[array, array]]:
        """
        Input: Excel Parsing 한 결과
        Action: MIDIParse 와 MIDIPreprocessing 을 통해 MIDI 파일 전처리
        Output: Piano_roll 형태의 matrix (record) 반환 [9 channels(num_instrument) , slice_bar x ticks]
        :return: record List[Union[array, array]]
        """
        midi_parser: MIDIParser = MIDIParser(
            numerator=self.time_numerate,
            denominator=self.time_denominator,
            data_info=self.data_info,
        )
        record: List[array, array] = midi_parser.parse()
        midi_parser.midi2pkl()
        return record

    def get_split_sequence(self, record: List[Union[array, array]]) -> Union[array, array, array]:
        """
        Action: Piano_roll 형태의 matrix 에서 [9 channels(num_instrument) , slice_bar * ticks]
        1. 같은 마디의 배치 형태로 만들기 위해 slice_bar * ticks 이하는 버림
        2. [9 channels , slice_bar * ticks] (2-dim) to
         [batch, 9, slice_bar * ticks] (3-dim) 형태로 만듬
        :param record: Piano_roll 형태의 matrix List[Union[array, array]]
        :return:
        """
        split_sequence: SplitSequence = SplitSequence(
            record=record, slice_bar=self.slice_bar, ticks=self.ticks
        )
        sub_sequence: Union[array, array, array] = split_sequence.make_subsequence_bar()
        split_sequence.save_subsequence_npy()
        return sub_sequence
