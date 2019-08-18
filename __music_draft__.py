from music21.note import Note, Rest
from music21.stream import Stream
from music21.midi import MidiFile
from music21.midi.realtime import StreamPlayer
from music21.midi.translate import midiFileToStream
from copy import deepcopy
from music21.instrument import *


def get_midi_stream_1():
    part1 = [Rest(), Rest(), Note('E-'), Rest()]
    part2 = [Rest(), Rest(), Note('A-'), Rest()]
    part3 = [Note('B-'), Rest(), Note('E-'), Rest()]
    part4 = [Note('B-'), Rest(), Note('A-'), Rest()]
    part5 = [Note('B-'), Rest(), Rest(), Rest()]
    part6 = [Note('G'), Rest(), Note('C'), Rest()]
    part7 = [Note('D'), Rest(), Note('E-'), Rest()]
    stream_instance = Stream()
    stream_instance.append(deepcopy(part1))
    stream_instance.append(deepcopy(part2))
    stream_instance.append(deepcopy(part3))
    stream_instance.append(deepcopy(part2))
    stream_instance.append(deepcopy(part3))
    stream_instance.append(deepcopy(part2))
    stream_instance.append(deepcopy(part4))
    stream_instance.append(deepcopy(part5))
    stream_instance.append(deepcopy(part1))
    stream_instance.append(deepcopy(part6))
    stream_instance.append(deepcopy(part7))
    stream_instance.append(deepcopy(part6))
    stream_instance.append(deepcopy(part7))
    return stream_instance


def open_midi(midi_path, remove_drums):
    mf = MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    if remove_drums:
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]

    return midiFileToStream(mf)


if __name__ == '__main__':
    # midi_stream = open_midi('music_samples_input/FFIX_Piano.mid', True)
    # midi_stream.plot('pianoroll')
    # midi_stream.plot('3dbars')
    # time_signature = midi_stream.getTimeSignatures()[0]
    # music_analysis = midi_stream.analyze('key')
    # print('Music time signature: {0}/{1}'.format(time_signature.beatCount, time_signature.denominator))
    # print('Expected music key: {0}'.format(music_analysis))
    # print('Music key confidence: {0}'.format(music_analysis.correlationCoefficient))

    # midi_stream = open_midi(
    #     'C:\\Users\\admin\\Downloads\\CREAM_SODA_-_Nikakix_Bolshe_Vecherinok.mp3.mid', False
    # )
    # midi_stream = music21.converter.parse(
    #     # 'C:\\Users\\admin\\Downloads\\CREAM_SODA_-_Nikakix_Bolshe_Vecherinok.mp3.mid'
    #     # 'C:\\Users\\admin\\Downloads\\07_Maroon_5___This_Love.mid'
    # )
    midi_stream = get_midi_stream_1()
    midi_stream.insert(0, ElectricBass())
    player = StreamPlayer(midi_stream)
    player.play()
    midi_stream.plot('pianoroll')
