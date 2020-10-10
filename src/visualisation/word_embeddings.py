import os
import pickle
import sys

import gensim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/pre_process/')
from config import PICKLE_PATH, DIMENSION, options, top_words, n_components, word2vec_path, \
    input_words, color_dict, CORPUS


def read(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


embeddings_vec, similar_words = [], []


def plot_similar_words(word_vec, s_words):
    plt.figure(figsize=(8, 8))
    for label, vec, words in zip(input_words, word_vec, s_words):
        x_axis, y_axis = vec[:, 0], vec[:, 1]
        plt.scatter(x_axis, y_axis, c=color_dict[label], label=label)
        for i, word in enumerate(words):
            plt.annotate(word, xy=(x_axis[i], y_axis[i]), xytext=(3, 3),
                         textcoords='offset points', ha='right', va='bottom')
    plt.legend()
    plt.title('Word Embedding')
    plt.show()


def get_embeddings(word2vec):
    for word in input_words:
        embedding, words = [], []
        for s_w, _ in word2vec.most_similar(word, topn=top_words):
            words.append(s_w), embedding.append(word2vec[s_w])
        embeddings_vec.append(embedding), similar_words.append(words)
    return np.array(embeddings_vec).reshape(len(input_words) * top_words, DIMENSION), similar_words


def run(argv):
    try:
        if len(argv) != CORPUS:
            raise Exception
        else:
            if isinstance(int(argv[0]), int):
                choice = int(argv[0])
                word2vec = read(f'{PICKLE_PATH}/corpus_embedding.pkl') if choice == CORPUS else \
                    gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
                vectors, similar_word = get_embeddings(word2vec)
                plot_similar_words(PCA(n_components=n_components).fit_transform(vectors).reshape(
                    len(input_words), top_words, n_components), similar_word)
    except Exception as e:
        print(f'{str(e).upper()}\nFollow command:\npython word_embeddings.py <file> --mandatory '
              f'<choice of embedding> --mandatory')
        print(f'{options}')


if __name__ == '__main__':
    run(sys.argv[1:])
