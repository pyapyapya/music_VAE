"""
1. Parsing Excel file
    (bpm, beat_type, midi_filename, duration, split)

2. Parsing MIDI File
    (start_time, end_time, pitch(to drum sets), velocity)
"""

import os
from copy import deepcopy
from csv import DictReader
from typing import List, Dict, Tuple

import pretty_midi
from config import PATH


class CSVParser:
    """
    Groove-MIDI Dataset 파싱하는 클래스
    """

    def __init__(self):
        self.dir_path = PATH["DIR_PATH"]

        self.train_info: List[Dict[str, str]] = []
        self.val_info: List[Dict[str, str]] = []
        self.test_info: List[Dict[str, str]] = []

    def parse(
        self, file_name: str = "info.csv"
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Groove-MIDI Dataset은 1,150개의 파일로 이루어져 있으므로,
        info.csv에서 파싱한 데이터의 개수도 같아야 함
        :param: file_name
        :return: Tuple[List[Dict[str, str], Dict[str, str], Dict[str, str]]
        """

        file_path = os.path.join(self.dir_path, file_name)

        with open(file_path, "r", encoding="UTF-8") as csv_file:
            reader: DictReader = DictReader(csv_file)
            for csv_line in reader:
                file_info: Dict[str, str] = {}
                file_info["midi_filename"] = csv_line["midi_filename"]
                file_info["bpm"] = csv_line["bpm"]
                file_info["duration"] = csv_line["duration"]

                split_data = csv_line["split"]

                if split_data == "train":
                    self.train_info.append(file_info)

                elif split_data == "validation":
                    self.val_info.append(file_info)

                elif split_data == "test":
                    self.test_info.append(file_info)

        total_files: int = (
            len(self.train_info) + len(self.val_info) + len(self.test_info)
        )

        assert total_files == 1150, "파싱한 데이터 셋이 Groove Dataset 개수(1,150)개와 같지 않음"
        print(self.train_info)
        return self.train_info, self.val_info, self.test_info


class MIDIPreprocessor:
    """
    [TODO]
    1. Make [9 channel x timestamp] rolls
    2. Make Tap (how about 3/4, 6/8 ... )
    3. Quantize Time
    """
    def __init__(self):
        self.dir_path = PATH['DIR_PATH']
        csv_parser = CSVParser()
        self.train_info, self.val_info, self.test_info = csv_parser.parse()

    def load_midi(self):
        """
        MIDI 데이터 전처리 순서
        1. MIDI Load
        2.
        :return:
        """
        for midi_data in self.train_info[:3]:
            file_path = midi_data['midi_filename']
            midi_path = os.path.join(self.dir_path, file_path)
            pm = pretty_midi.PrettyMIDI(midi_path)

            bpm = midi_data['bpm']
            duration = midi_data['duration']

            tempo_change_times, tempo_change_bpm = pm.get_tempo_changes()
            song_start = 0
            song_end = pm.get_end_time()
            print(file_path)
            print(tempo_change_times, tempo_change_bpm, song_end)
            self.adjust_time(pm)

    def adjust_time(self, pm):
        song_start = pm.instruments[0].notes[0].start
        song_end = pm.get_end_time()

        for instrument in pm.instruments:
            new_notes = []
            for note in instrument.notes:
                if note.start >= song_start and note.end <= song_end:
                    note.start -= song_start
                    note.end -= song_start
                    new_notes.append(note)
            pm.instruments[0].notes = new_notes

    def parse_midi(self):
        pass

    def get_piano_roll(self):
        pass

    def quantize(self):
        pass


midi_processor = MIDIPreprocessor()
midi_processor.load_midi()
