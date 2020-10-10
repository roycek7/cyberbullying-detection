from __future__ import absolute_import

import pickle

import gensim
import numpy as np
from config import DIMENSION, MAX_LEN, PICKLE_PATH, MAX_WORDS, word2vec_path
from pre_processing import read_dataset, file_path, clean_dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def get_word_embedding_data(df, df_test, field, word2vec, train=True, word_embedding_weights=None, cnn_data=None):
    try:
        # requires cleaned text
        # creates a tokenizer, configured to only take into account the total number of dataframe most common words,
        # with lower chars
        tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True, char_level=False)

        # builds the word index
        tokenizer.fit_on_texts(df[field].tolist() + df_test[field].tolist())

        # turns strings into lists of integer indices
        word_sequence = tokenizer.texts_to_sequences(df[field].tolist()) if train else \
            tokenizer.texts_to_sequences(df_test[field].tolist())

        # turns the lists of integers into a 2D integer
        # maxlen() - cuts off the text after this number of words
        cnn_data = pad_sequences(word_sequence, maxlen=MAX_LEN)

        if train:
            # embedding matrix
            word_embedding_weights = np.zeros((len(tokenizer.word_index) + 1, DIMENSION))
            for w, i in tokenizer.word_index.items():
                if w in word2vec:
                    word_embedding_weights[i, :] = word2vec[w]

    except FileNotFoundError as e:
        print(f'Unable to find Google News pre-trained word embeddings file!!!\n{e}')
    finally:
        return word_embedding_weights, cnn_data


def store(file, object):
    with open(f'{file}.pkl', 'wb') as f:
        pickle.dump(object, f)


def read(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


try:
    corpus_embedding_matrix = get_word_embedding_data(clean_dataset(read_dataset('train.csv')),
                                                      clean_dataset(read_dataset('test.csv')),
                                                      'comment_text', read(f'{PICKLE_PATH}/corpus_embedding.pkl'))[0]

    store(f'{PICKLE_PATH}/corpus_word_embedding', corpus_embedding_matrix)

    # NEED PRE-TRAINED WORD EMBEDDINGS FOR EMBEDDING LAYER
    word_embedding_matrix, training_cnn_data = get_word_embedding_data(clean_dataset(read_dataset('train.csv')),
                                                                       clean_dataset(read_dataset('test.csv')),
                                                                       'comment_text',
                                                                       gensim.models.KeyedVectors.load_word2vec_format(
                                                                           word2vec_path, binary=True))

    store(f'{PICKLE_PATH}/pretrained_word_embedding', word_embedding_matrix), store(f'{PICKLE_PATH}/cnn_sequence',
                                                                                    training_cnn_data)

    test_cnn = get_word_embedding_data(clean_dataset(read_dataset('train.csv')),
                                       clean_dataset(read_dataset('test.csv')), 'comment_text', None,
                                       train=False)[1]
    store(f'{PICKLE_PATH}/test_cnn_sequence', test_cnn)
except Exception as e:
    print(f'{str(e).upper()}\nFollow command:\npython word_embeddings.py <file> --mandatory')
    print('Need to Download: https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz\n'
          'and place it in cyberbullying-detection/data/pretrained_word2vec/')
