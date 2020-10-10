import os
import sys

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/pre_process/')

from pre_processing import file_path

CORPUS = 1
PRE_TRAINED = 2
EPOCH = 100
DIMENSION = 300
MAX_LEN = 200
FILTERS = 64
KERNEL_SIZE = 5
MAX_WORDS = 300000
DROPOUT_RATE = 0.2
POOL_SIZE = 2
BATCH_SIZE = 256

word2vec_path = file_path('data', "pretrained_word2vec", "GoogleNews-vectors-negative300.bin")
top_words = 10
n_components = 2
input_words = ['Python', 'House', 'Dog', 'Australia']
color_dict = {'Python': 'red', 'House': 'blue', 'Dog': 'green', 'Australia': 'purple'}


PICKLE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'data'))

options = {
    1: 'corpus',
    2: 'google_pre_trained'
}

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

