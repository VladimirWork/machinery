import music21


def get_midi_stream():
    notes = [music21.note.Note(x) for x in ['A3', 'B3', 'C3', 'D3', 'E3', 'F3', 'G3',
                                            'A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2']]
    offset = 0
    for note in notes:
        note.offset = offset
        offset += 0.5
    return music21.stream.Stream(notes)


def open_midi(midi_path, remove_drums):
    mf = music21.midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    if remove_drums:
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]

    return music21.midi.translate.midiFileToStream(mf)


if __name__ == '__main__':
    # midi_stream.insert(0, music21.instrument.ElectricBass())
    # midi_stream.insert(0, music21.instrument.AcousticBass())
    # midi_stream.insert(0, music21.instrument.Bass())
    midi_stream = open_midi('music_samples_input/FFIX_Piano.mid', True)
    midi_stream.plot('pianoroll')
    midi_stream.plot('scatter', 'offset', 'pitchClass')
    time_signature = midi_stream.getTimeSignatures()[0]
    music_analysis = midi_stream.analyze('key')
    print('Music time signature: {0}/{1}'.format(time_signature.beatCount, time_signature.denominator))
    print('Expected music key: {0}'.format(music_analysis))
    print('Music key confidence: {0}'.format(music_analysis.correlationCoefficient))
    player = music21.midi.realtime.StreamPlayer(midi_stream)
    player.play()
