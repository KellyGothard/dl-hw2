# Gather data from subreddits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import joblib
import re

from tensorflow.python.client import device_lib

import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN, LSTM, GRU
from keras.callbacks import ModelCheckpoint

def main(batch_size=512, epochs=15, period = 5, char_length=10, vocab_size=256, sequence_length = 10):

    data_path = '~/scratch/dl-hw2/data/The_Donald_20000.csv'
    # data_path = '../data/The_Donald_20000.csv'
    df = pd.read_csv(data_path,nrows=10)
    df = df[df['body'].notna()]
    data = ' '.join(list(df['body']))
    X, y, lines = encode_text(data, sequence_length)
    print('X shape: '+str(X.shape))
    print('y shape: '+str(y.shape))

    checkpoint_paths = glob.glob('*.hdf5')
    print(checkpoint_paths)
    for path in checkpoint_paths:
        print('##################################################################')
        print(path)
        model_type = path.split('-')[0]
        model = Sequential()
        if model_type == 'rnn':
            model.add(SimpleRNN(75, input_shape=(X.shape[1], X.shape[2])))
        if model_type == 'lstm':
            model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
        if model_type == 'gru':
            model.add(GRU(75, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dense(vocab_size, activation='softmax'))

        model.load_weights(path)
        parallel_model = parallelize(model)
        parallel_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        preds = parallel_model.predict(X, batch_size=batch_size)
        char_preds = decode_text(preds)

        for i in range(len(lines)):
            print(lines[i]+'-'+char_preds[i])

def available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def onehot_encode_text(post):
    onehot = np.zeros([len(post),1,256])
    char_count = 0
    for char in post:
        pos = ord(char)
        try:
            onehot[char_count][0][pos] = 1
        except:
            print('non ascii')
        char_count += 1

    return np.array(onehot)

def encode_text(post, sequence_length):
    '''
    :param post:
    :return: one hot encoding of characters in post
    '''
    post = post.encode("ascii", errors="ignore").decode()
    mapping = dict((chr(i), i) for i in range(256))
    vocab_size = len(mapping)
    print('Vocabulary Size: %d' % vocab_size)

    lines = create_sequences(post, sequence_length)

    sequences = []
    for line in lines:
        # integer encode line
        encoded_seq = [mapping[char] for char in line]
        # store
        sequences.append(encoded_seq)

    X, y = x_y_split(sequences, sequence_length)

    sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
    X = np.array(sequences)
    y = to_categorical(y, num_classes=vocab_size)

    return X, y, lines

def create_sequences(post, sequence_length):
    sequences = []
    for i in range(len(post)):
        sequence = post[i:i+sequence_length]
        sequences.append(sequence)
    return sequences

def x_y_split(encoded, sequence_length):
    X = []
    y = []
    for i in range(len(encoded)-1):
        if len(encoded[i]) == sequence_length:
            X.append(encoded[i])
            y.append(encoded[i+1][0])
    X = np.array(X)
    y = np.array(y)

    return X, y

def parallelize(model):

    gpu_count = len(available_gpus())
    if gpu_count > 1:
        print(f"\n\nModel parallelized over {gpu_count} GPUs.\n\n")
        parallel_model = keras.utils.multi_gpu_model(model, gpus=gpu_count)
    else:
        print("\n\nModel not parallelized over GPUs.\n\n")
        parallel_model = model

    return parallel_model


def decode_text(preds):
    preds_str = []
    for pred in preds:
        max_pred = np.where(pred == max(pred))[0]
        pred_str = chr(max_pred)
        preds_str.append(pred_str)

    return preds_str


if __name__ == '__main__':
    main()