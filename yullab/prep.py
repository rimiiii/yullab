import os
import pretty_midi
import numpy as np
from yullab.utils import class_map, ROLAND_DRUM_PITCH_CLASSES
import pandas as pd
import math


def generate_midi_df(base_path, sample_filenames, max_velocity=127):
    note_dfs = []
    for sample_filename in sample_filenames:
        sample_filename = os.path.join(base_path, sample_filename)
        midi = pretty_midi.PrettyMIDI(sample_filename)

        ns = midi.instruments[0]
        note_sequences = []
        for i, note in enumerate(ns.notes):
            columns = np.zeros(len(ROLAND_DRUM_PITCH_CLASSES))
            columns[class_map()[note.pitch]] = note.velocity/max_velocity
            note_sequences.append(columns)

        note_seq_df = pd.DataFrame(note_sequences)
        note_seq_df.reset_index(inplace=True)
        note_seq_df['filename'] = os.path.basename(sample_filename).split('.')[0]
        note_seq_df.rename(columns={'index': 'timestep'}, inplace=True)
        note_dfs.append(note_seq_df)
    drum_df = pd.concat(note_dfs, axis=0)
    drum_df.set_index(['filename', 'timestep'], inplace=True)
    return drum_df


class BarTransform():

    def __init__(self, bars=1, note_count=60):
        self.split_size = bars*16
        self.note_count = note_count

    def get_sections(self, sample_length):
        return math.ceil(sample_length/ self.split_size)