from keras import utils, models, layers, optimizers
import numpy as np
import random
import sys
from utils import sample
from vk_connector import main


path = utils.get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('Corpus length: {}'.format(len(text)))

# 60 symbols sequences
max_len = 60
# new sequence after each 3 symbols
step = 3
# store extracted sequences
sentences = []
# store targets (symbols right after sequences)
next_chars = []

for i in range(0, len(text) - max_len, step):
    sentences.append(text[i:i + max_len])
    next_chars.append(text[i + max_len])
print('Number of sequences: {}'.format(len(sentences)))

# unique symbols list
chars = sorted(list(set(text)))
print('Number of unique characters: {}'.format(len(chars)))
# map unique characters to indices in list
char_indices = dict((char, chars.index(char)) for char in chars)

# vectorization
x = np.zeros((len(sentences), max_len, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

# direct symbols encoding into binary arrays
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

model = models.Sequential()
model.add(layers.LSTM(128, input_shape=(max_len, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))
optimizer = optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# text generation loop
for epoch in range(1, 50):
    print('\nEpoch: {}\n'.format(epoch))
    model.fit(x, y, batch_size=2048, epochs=1)
    start_index = random.randint(0, len(text) - max_len - 1)
    generated_text = text[start_index: start_index + max_len]
    print('\nGenerating with seed: "{}"\n'.format(generated_text))
    result = generated_text
    # sys.stdout.write(generated_text)
    for i in range(800):
        sampled = np.zeros((1, max_len, len(chars)))
        for t, char in enumerate(generated_text):
            sampled[0, t, char_indices[char]] = 1.
        preds = model.predict(sampled, verbose=0)[0]
        next_index = sample(preds, 0.5)
        next_char = chars[next_index]
        generated_text += next_char
        generated_text = generated_text[1:]
        result += next_char
        # sys.stdout.write(next_char)

    if epoch > 40:
        main(result, 'C:\\Users\\admin\\Downloads\\forest.jpg')
