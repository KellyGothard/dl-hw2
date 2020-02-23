# Gather data from subreddits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pushshift

from tensorflow.python.client import device_lib

import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.callbacks import ModelCheckpoint

def main(batch_size=256, epochs=50):

    # pushshift.subreddit_posts(subreddit = 'The_Donald', n = 100000, save_csv = True, name = 'The_Donald_100000')

    data_path = '../data/The_Donald_10000.csv'
    data = ' '.join(list(pd.read_csv(data_path,nrows=1000)['body']))
    print('Total number of characters: '+str(len(data)))

    X, y, vocab_size = encode_text(data)
    print('X shape: '+str(X.shape))
    print('y shape: '+str(y.shape))

    model = Sequential()
    model.add(SimpleRNN(75, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())

    gpu_count = len(available_gpus())
    if gpu_count > 1:
        print(f"\n\nModel parallelized over {gpu_count} GPUs.\n\n")
        parallel_model = keras.utils.multi_gpu_model(model, gpus=gpu_count)
    else:
        print("\n\nModel not parallelized over GPUs.\n\n")
        parallel_model = model

    # compile model
    parallel_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    filepath = '../output/simple-rnn-{epoch:02d}.hdf5'

    # Keep only a single checkpoint, the best over test accuracy.
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 period = 10)

    # fit model
    history = parallel_model.fit(X, y, epochs=epochs,
                                 batch_size=batch_size,
                                 validation_split = 0.1,verbose=2)

    plot_acc(history)

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

def encode_text(post):
    '''
    :param post:
    :return: one hot encoding of characters in post
    '''
    post =  post.encode("ascii", errors="ignore").decode()
    chars = sorted(list(set(post)))
    mapping = dict((c, i) for i, c in enumerate(chars))
    vocab_size = len(mapping)
    print('Vocabulary Size: %d' % vocab_size)

    lines = create_sequences(post)

    sequences = []
    for line in lines:
        # integer encode line
        encoded_seq = [mapping[char] for char in line]
        # store
        sequences.append(encoded_seq)

    X, y = x_y_split(sequences)

    sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
    X = np.array(sequences)
    y = to_categorical(y, num_classes=vocab_size)

    return X, y, vocab_size

def create_sequences(post):
    sequences = []
    for i in range(len(post)):
        sequence = post[i:i+10]
        sequences.append(sequence)
    return sequences

def x_y_split(encoded):
    X = []
    y = []
    for i in range(len(encoded)-1):
        if len(encoded[i]) == 10:
            X.append(encoded[i])
            y.append(encoded[i+1][0])
    X = np.array(X)
    y = np.array(y)

    return X, y

def plot_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('../output/simple_rnn_training_acc.png')

if __name__ == '__main__':
    main()
