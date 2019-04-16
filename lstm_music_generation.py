from music21 import *
import random


if __name__ == '__main__':
    keysDetuned = []
    for i in range(127):
        keysDetuned.append(random.randint(-10, 10))
    sample = corpus.parse('bach/bwv66.6')
    for note in sample.flat.notes:
        note.pitch.microtone = keysDetuned[note.pitch.midi]
    player = midi.realtime.StreamPlayer(sample)
    player.play()
