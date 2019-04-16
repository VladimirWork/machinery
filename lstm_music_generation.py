from music21 import *
import random
import glob
import numpy as np
from keras.utils import np_utils


# noinspection PyUnusedLocal
def load_music_samples() -> list:
    notes = []
    for file in glob.glob('music_samples/*.mid'):
        _midi = converter.parse(file)
        notes_to_parse = None
        parts = instrument.partitionByInstrument(_midi)
        if parts:  # file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else:  # file has notes in a flat structure
            notes_to_parse = _midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes


def encode_and_map(notes: list) -> (list, list):
    sequence_length = 100
    n_vocab = len(set(notes))
    # get all pitch names
    pitch_names = sorted(set(item for item in notes))
    # map pitches to integers
    note_to_int = dict((_note, number) for number, _note in enumerate(pitch_names))
    network_input = []
    network_output = []
    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
    n_patterns = len(network_input)
    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)
    network_output = np_utils.to_categorical(network_output)
    return network_input, network_output


def test_stream_player() -> None:
    keys_detuned = []
    for i in range(127):
        keys_detuned.append(random.randint(-10, 10))
    sample = corpus.parse('bach/bwv70.7')
    for _note in sample.flat.notes:
        _note.pitch.microtone = keys_detuned[_note.pitch.midi]
        print(_note)
    print(len(sample.flat.notes))
    player = midi.realtime.StreamPlayer(sample)
    player.play()


if __name__ == '__main__':
    pass
