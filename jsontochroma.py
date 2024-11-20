from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


tokenizer=Tokenizer(num_words=vocab_length)
tokenizer.fit_on_texts(moves)
sequences=tokenizer.texts_to_sequences(moves)
word_index=tokenizer.word_index
model_inputs=pad_sequences(sequences,maxlen=max_len)

new_sequence = tokenizer.texts_to_sequences(['d4 d5 Bf4 Na6 e3 e6 c3 Nf6 Nf3 Bd7 Nbd2 b5 Bd3 Qc8 e4 b4 e5 Ne4 Nxe4 dxe4 Bxe4 bxc3 Bxa8 Qxa8 bxc3 Ba3 Rb1 c5 Qd3 O-O Qxa6 Bc6 Qxa3 Bxf3 gxf3 Qxf3 Qxa7 Qxh1+ Ke2 Qxb1 Qxc5 Qc2+ Ke3 Qxa2 Qb4 h6 c4 g5 Bg3 Qa8 c5 Rb8 Qc3 f5 f4 Qe4+ Kd2 Qg2+ Kd3 gxf4 Bxf4 Qf3+ Kc4 Qxf4 c6 Qf1+ Kc5 Rb1 Qg3+ Kf7 c7 Rc1+ Kd6 Qa6+ Kd7 Qb5+ Kd8 Qe8#'])

# Pad the sequence to match the input size of your model
padded_sequence = pad_sequences(new_sequence, maxlen=max_len)
print(padded_sequence)

model = load_model('/content/drive/MyDrive/smc_chess_model.h5')

# Make predictions


# Example prediction
prediction = model.predict(padded_sequence)

# Get the index of the maximum value
print(prediction)