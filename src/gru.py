# Gather data from subreddits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.python.client import device_lib

import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

def main(batch_size=512, epochs = 25, period = 5, hidden_units = 75, vocab_size=256, sequence_length = 10):

    # pushshift.subreddit_posts(subreddit = 'The_Donald', n = 20000, save_csv = True, name = 'The_Donald_20000')

    data_path = '~/scratch/dl-hw2/data/The_Donald_20000.csv'
    # data_path = '../data/The_Donald_20000.csv'
    df = pd.read_csv(data_path,nrows=12000)
    df = df[df['body'].notna()]
    data = ' '.join(list(df['body']))
    X, y = encode_text(data, sequence_length)

    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.1)

    print('X train shape: '+str(X_train.shape))
    print('y train shape: '+str(y_train.shape))

    model = Sequential()
    model.add(GRU(hidden_units, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())

    parallel_model = parallelize(model)
    parallel_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    filepath = 'rnn-{epoch:02d}.hdf5'
    checkpoint = ModelCheckpoint(filepath, verbose=1, period=period)

    history = parallel_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split = 0.1,verbose=2, callbacks = [checkpoint])
    plot_acc(history)

    preds = parallel_model.predict(X_test, batch_size=batch_size)
    print(preds.shape)
    print(y_test.shape)

    c = confusion_matrix(np.argmax(y_test, axis=-1), np.argmax(preds, axis=-1))

    print(c)

    print(
        classification_report(
            np.argmax(y_test, axis=-1),
            np.argmax(preds, axis=-1),
            labels=[str(x) for x in range(vocab_size)],
            target_names=[str(x) for x in range(vocab_size)],
        )
    )

    plot_confusion_matrix(
        c,
        list(range(vocab_size)),
        model_type='rnn',
        normalize=False)
    plot_confusion_matrix(
        c,
        list(range(vocab_size)),
        model_type='rnn',
        normalize=True)

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
    chars = sorted(list(set(post)))
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

    return X, y

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
            y.append(encoded[i+1][-1])
    X = np.array(X)
    y = np.array(y)

    return X, y

def plot_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.savefig('rnn_training_acc.png')

def parallelize(model):

    gpu_count = len(available_gpus())
    if gpu_count > 1:
        print(f"\n\nModel parallelized over {gpu_count} GPUs.\n\n")
        parallel_model = keras.utils.multi_gpu_model(model, gpus=gpu_count)
    else:
        print("\n\nModel not parallelized over GPUs.\n\n")
        parallel_model = model

    return parallel_model

def plot_confusion_matrix(
        cm,
        classes,
        model_type,
        normalize=False,
        title='Confusion matrix',
        cmap='Blues',
        output_path='.',
):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        tag = '_norm'
        print("Normalized confusion matrix:")
    else:
        tag = ''
        print('Confusion matrix:')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap(cmap))
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'{output_path}/{model_type}_confusion{tag}.png')
    plt.close()


if __name__ == '__main__':
    main(hidden_units=50)
