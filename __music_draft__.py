import music21

if __name__ == '__main__':
    notes = [music21.note.Note(x) for x in ['A3', 'B3', 'C3', 'D3', 'E3', 'F3', 'G3',
                                            'A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2']]
    offset = 0
    for note in notes:
        note.offset = offset
        offset += 0.5
    midi_stream = music21.stream.Stream(notes)
    # midi_stream.insert(0, music21.instrument.ElectricBass())
    midi_stream.insert(0, music21.instrument.AcousticBass())
    # midi_stream.insert(0, music21.instrument.Bass())
    midi_stream.show('text')
    player = music21.midi.realtime.StreamPlayer(midi_stream)
    player.play()
