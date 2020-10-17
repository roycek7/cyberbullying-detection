import os
import pickle
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/pre_process/')

from pre_processing import read_dataset
from config import PICKLE_PATH, MAX_LEN, EPOCH, DIMENSION, FILTERS, KERNEL_SIZE, CORPUS, \
    options, labels, DROPOUT_RATE, POOL_SIZE, BATCH_SIZE, NEURONS

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model


def read(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def CNN(embedding_matrix):
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=DIMENSION, weights=[embedding_matrix],
                        input_length=MAX_LEN, trainable=False))
    model.add(Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(MaxPooling1D(pool_size=POOL_SIZE))
    model.add(Flatten())
    model.add(Dropout(rate=DROPOUT_RATE))
    model.add(Dense(units=NEURONS, activation='relu'))
    model.add(Dense(units=len(labels), activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def prediction_output(y_predicted, choice):
    df = pd.DataFrame(columns=['id'] + labels)
    df['id'] = read_dataset('test.csv')['id'].values
    df[labels] = y_predicted
    df.to_csv(f'{PICKLE_PATH}/{options[choice]}_prediction.csv', index=False)


def plot_accuracy_loss(plot_1, plot_2, title, y, x):
    plt.plot(plot_1)
    plt.plot(plot_2)
    plt.title(title)
    plt.ylabel(y)
    plt.xlabel(x)
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def run(argv):
    try:
        if len(argv) != CORPUS:
            raise Exception
        else:
            if isinstance(int(argv[0]), int):
                choice = int(argv[0])

                # Select corpus or pre-trained embedding matrix for embedding layer based on choice given in program
                # input
                corpus_word_embedding_matrix, word_embedding_matrix, training_cnn_data, test_cnn = \
                    read(f'{PICKLE_PATH}/corpus_word_embedding.pkl'), \
                    read(f'{PICKLE_PATH}/pretrained_word_embedding.pkl'), \
                    read(f'{PICKLE_PATH}/cnn_sequence.pkl'), read(f'{PICKLE_PATH}/test_cnn_sequence.pkl')

                y_train = read_dataset('train.csv')[labels].values

                embedding_matrix = corpus_word_embedding_matrix if choice == CORPUS else word_embedding_matrix
                cnn_model = CNN(embedding_matrix)
                history = cnn_model.fit(training_cnn_data, y_train, epochs=EPOCH, batch_size=BATCH_SIZE,
                                        validation_split=0.1, shuffle=True,
                                        callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

                cnn_model.save(f'{PICKLE_PATH}/{options[choice]}_cyberbullying.h5')

                plot_model(cnn_model, to_file='cnn_model.png', show_shapes=True, show_layer_names=True)

                y_pred = cnn_model.predict(test_cnn)
                prediction_output(y_pred, choice)
                plot_accuracy_loss(history.history['accuracy'], history.history['val_accuracy'],
                                   'model-accuracy', 'accuracy', 'epoch')
                plot_accuracy_loss(history.history['loss'], history.history['val_loss'],
                                   'model-loss', 'loss', 'epoch')
    except Exception as e:
        print(f'{str(e).upper()}\nFollow command:\npython convolutional_neural_network.py <file> --mandatory '
              f'<choice of embedding matrix> --mandatory')
        print(f'{options}')


if __name__ == '__main__':
    run(sys.argv[1:])
