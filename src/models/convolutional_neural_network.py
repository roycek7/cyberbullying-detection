import os
import pickle
import sys

import pandas as pd

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/pre_process/')

from pre_processing import read_dataset
from config import PICKLE_PATH, MAX_LEN, EPOCH, DIMENSION, FILTERS, KERNEL_SIZE, CORPUS, \
    options, labels, DROPOUT_RATE, POOL_SIZE, BATCH_SIZE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping


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
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=len(labels), activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def prediction_output(y_predicted, choice):
    df = pd.DataFrame(columns=['id'] + labels)
    df['id'] = read_dataset('test.csv')['id'].values
    df[labels] = y_predicted
    df.to_csv(f'{PICKLE_PATH}/{options[choice]}_prediction.csv', index=False)


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
                cnn_model.fit(training_cnn_data, y_train, epochs=EPOCH, batch_size=BATCH_SIZE, validation_split=0.1,
                              callbacks=[EarlyStopping(monitor='val_loss', patience=4)])

                cnn_model.save(f'{PICKLE_PATH}/{options[choice]}_cyberbullying.h5')

                y_pred = cnn_model.predict(test_cnn)
                prediction_output(y_pred, choice)
    except Exception as e:
        print(f'{str(e).upper()}\nFollow command:\npython convolutional_neural_network.py <file> --mandatory '
              f'<choice of embedding matrix> --mandatory')
        print(f'{options}')


if __name__ == '__main__':
    run(sys.argv[1:])
