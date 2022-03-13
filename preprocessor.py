"""
1. Parsing Excel file
    (bpm, beat_type, midi_filename, duration, split)

2. Parsing MIDI File
    (start_time, end_time, pitch(to drum sets), velocity)
"""

import os
from pickle import load, dump
from csv import DictReader
from typing import List, Dict, Tuple

from numpy import array, zeros
from pretty_midi import PrettyMIDI
from pretty_midi import Note
from tqdm import tqdm

from config import PATH
from constant import ROLAND_DRUM_PITCH_CLASSES


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
        return self.train_info, self.val_info, self.test_info


class MIDIParser:
    """
    MIDI Data를 읽고, MIDIPreprocessor에 의해 변환된 벡터를 pickle로 쓰는하는 클래스
    """
    def __init__(self):

        self.dir_path = PATH['DIR_PATH']
        self.train_info, self.val_info, self.test_info = CSVParser().parse()
        self.record: List[array, array] = []

    def parse(self):
        for midi_data in tqdm(self.train_info):
            preprocessor: DrumExtractor = DrumExtractor(midi_data)
            preprocessor.adjust_time()
            piano_roll = preprocessor.get_piano_roll()
            self.record.append(piano_roll)

    def midi2pkl(self):
        file_name = os.path.join(self.dir_path, 'record.pkl')
        with open(file_name, 'wb') as pkl:
            dump(self.record, pkl)


class DrumExtractor:
    """
    MIDI Data에서 Drum 데이터를 전처리하여 vector로 변환하는 클래스
    """
    def __init__(self, midi_data: Dict[str, str]):
        self.midi_data: Dict[str, str] = midi_data
        self.bpm: int = int(midi_data['bpm'])
        self.end_time: float = float(midi_data['duration'])

        self.dir_path = PATH['DIR_PATH']

        self.ROLAND_DRUM_PITCH_CLASSES = 9
        self.MINUTE = 60
        self.ticks: float = 1 / (self.bpm / self.MINUTE) / 4

        self.file_path = self.get_filename()
        self.pm: PrettyMIDI = PrettyMIDI(self.file_path)

    def get_piano_roll(self) -> array:
        """
        [TODO] piano_roll 생성할 때, velocity 조절하도록 만들어야 함
        [TODO] 학습 진행할 때는 velocity가 없는 one-hot option도 필요할 듯

        미디 정보를 이용하여 vector로 표현할 수 있는 piano roll로 만드는 함수
        :return: np.array [9 channel, total ticks]
        """

        piano_roll: array = zeros((self.ROLAND_DRUM_PITCH_CLASSES, int(self.end_time//self.ticks)))
        notes = self.pm.instruments[0].notes

        # Extract Only-Drum in MIDI File
        for note in notes:
            drum_class: int = ROLAND_DRUM_PITCH_CLASSES[note.pitch]
            start_bar = int(note.start//self.ticks)
            end_bar = int(note.end//self.ticks)
            piano_roll[drum_class, start_bar:end_bar] = note.velocity
        return piano_roll

    def adjust_time(self):
        """
        start time을 0초로 조정
        :return:
        """
        song_start = self.pm.instruments[0].notes[0].start
        song_end = self.end_time

        # Extract Only-Drum in MIDI File
        notes = self.pm.instruments[0].notes
        for note in notes:
            if note.start >= song_start and note.end <= song_end:
                note.start -= song_start
                note.end -= song_start

    def _time_quantize(self, note: Note):
        """
        re-ascending notes using quantized notes
        :param note:
        :return:
        """
        start = note.start
        end = note.end

        quantize_start = self._quantize_round(start)

        diff = start - quantize_start
        quantize_end = end - diff
        note.start = quantize_start
        note.end = quantize_end

    def _quantize_round(self, value: float) -> float:
        """
        퀀타이즈 Default 1/16
        :param value: flaot start_time
        :return: quantized start time
        """
        ratio = self.ticks
        return (value + ratio / 2) // ratio * ratio

    def get_filename(self) -> str:
        file_path = self.midi_data['midi_filename']
        midi_path = os.path.join(self.dir_path, file_path)
        return midi_path

    def get_time_signature(self) -> Tuple[int, int]:
        time_signature = self.midi_data['midi_filename'].split('/')[-1].split('_')[4]
        time_signature = time_signature.split('.')[0].split('-')
        time_numerator = int(time_signature[0])
        time_denominator = int(time_signature[1])

        return time_numerator, time_denominator
