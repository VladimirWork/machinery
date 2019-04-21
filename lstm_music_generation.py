from music21 import *
import random
import glob
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint


# noinspection PyUnusedLocal
def load_music_samples():
    _notes = []
    for file in glob.glob('music_samples_input/*.mid'):
        _midi = converter.parse(file)
        notes_to_parse = None
        parts = instrument.partitionByInstrument(_midi)
        if parts:  # file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else:  # file has notes in a flat structure
            notes_to_parse = _midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                _notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                _notes.append('.'.join(str(n) for n in element.normalOrder))
    return _notes


def encode_and_map(notes):
    sequence_length = 100
    _n_vocab = len(set(notes))
    # get all pitch names
    _pitch_names = sorted(set(item for item in notes))
    # map pitches to integers
    note_to_int = dict((_note, number) for number, _note in enumerate(_pitch_names))
    _network_input = []
    _network_output = []
    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        _network_input.append([note_to_int[char] for char in sequence_in])
        _network_output.append(note_to_int[sequence_out])
    n_patterns = len(_network_input)
    # reshape the input into a format compatible with LSTM layers
    _network_input = np.reshape(_network_input, (n_patterns, sequence_length, 1))
    # normalize input
    _network_input = _network_input / float(_n_vocab)
    _network_output = np_utils.to_categorical(_network_output)
    return _network_input, _network_output, _n_vocab, _pitch_names


def get_model(network_input, n_vocab):
    _model = Sequential()
    _model.add(LSTM(
        256,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    _model.add(Dropout(0.3))
    _model.add(LSTM(512, return_sequences=True))
    _model.add(Dropout(0.3))
    _model.add(LSTM(256))
    _model.add(Dense(256))
    _model.add(Dropout(0.3))
    _model.add(Dense(n_vocab))
    _model.add(Activation('softmax'))
    _model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return _model


def train_model(model, network_input, network_output):
    path = 'music_weights_{epoch:02d}_{loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(
        path,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)
    return model


def generate_notes(model, network_input, n_vocab, pitch_names, amount):
    start = np.random.randint(0, len(network_input) - 1)
    int_to_note = dict((number, _note) for number, _note in enumerate(pitch_names))
    pattern = list(network_input[start])
    _prediction_output = []  # predicted notes
    for note_index in range(amount):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        _prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    return _prediction_output


def process_notes(prediction_output):
    offset = 0
    _output_notes = []
    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            _output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            _output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += 0.5
    return _output_notes


def test_stream_player():
    keys_detuned = []
    for i in range(127):
        keys_detuned.append(random.randint(-10, 10))
    sample = corpus.parse('bach/bwv70.7')
    for _note in sample.flat.notes:
        _note.pitch.microtone = keys_detuned[_note.pitch.midi]
        print(_note)
    print(len(sample.flat.notes))
    _player = midi.realtime.StreamPlayer(sample)
    _player.play()


if __name__ == '__main__':
    notes = load_music_samples()
    network_input, network_output, n_vocab, pitch_names = encode_and_map(notes)
    model = get_model(network_input, n_vocab)
    model.load_weights('weights-improvement-186-0.0711-bigger.hdf5')
    model = train_model(model, network_input, network_output)
    prediction_output = generate_notes(model, network_input, n_vocab, pitch_names, 500)
    output_notes = process_notes(prediction_output)
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='music_samples_output/test_output_21_04.mid')
    player = midi.realtime.StreamPlayer(midi_stream)
    player.play()
