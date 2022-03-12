"""
1. Parsing Excel file
    (bpm, beat_type, midi_filename, duration, split)

2. Parsing MIDI File
    (start_time, end_time, pitch(to drum sets), velocity)
"""

import os
from csv import DictReader
from typing import List, Dict

from config import PATH


class CSVParser:
    """
    Groove-MIDI Dataset 파싱하는 클래스
    """
    def __init__(self):
        self.dir_path = PATH['DIR_PATH']

        self.midi_file_name: List = []
        self.bpm: List = []
        self.duration: List = []

        self.train_info: List[Dict[str, str]] = []
        self.val_info: List[Dict[str, str]] = []
        self.test_info: List[Dict[str, str]] = []

    def load_csv(self, file_name: str = 'info.csv'):
        """
        Groove-MIDI Dataset은 1,150개의 파일로 이루어져 있으므로,
        info.csv에서 파싱한 데이터의 개수도 같아야 함
        :return:
        """

        file_path = os.path.join(self.dir_path, file_name)

        with open(file_path, 'r', encoding='UTF-8') as csv_file:
            reader: DictReader = DictReader(csv_file)
            file_info: Dict[str, str] = {}
            for csv_line in reader:

                file_info['midi_filename'] = csv_line['midi_filename']
                file_info['bpm'] = csv_line['bpm']
                file_info['duration'] = csv_line['duration']

                split_data = csv_line['split']

                if split_data == 'train':
                    self.train_info.append(file_info)

                elif split_data == 'validation':
                    self.val_info.append(file_info)

                elif split_data == 'test':
                    self.test_info.append(file_info)

        total_files: int = len(self.train_info) + len(self.val_info) + len(self.test_info)

        assert total_files == 1150, "파싱한 데이터 셋이 Groove Dataset 개수(1,150)개와 같지 않음"
