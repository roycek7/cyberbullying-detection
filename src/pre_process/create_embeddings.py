import pickle
import ssl

import gensim
import nltk
import pandas as pd
from config import DIMENSION, PICKLE_PATH
from gensim.models import Word2Vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pre_processing import read_dataset

ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
corpus = []
lematizer = WordNetLemmatizer()


def store(file, object):
    with open(f'{file}.pkl', 'wb') as f:
        pickle.dump(object, f)


def create_embeddings(df):
    for row in df:
        words = [word for word in word_tokenize(row) if word not in stop_words]
        lematized_words = [lematizer.lemmatize(word) for word in words]
        corpus.append(lematized_words)
    return corpus


df_train = read_dataset('train.csv')
df_test = read_dataset('test.csv')
df_combine = pd.concat([df_train['comment_text'], df_test['comment_text']], ignore_index=True)

word2vec = gensim.models.Word2Vec(create_embeddings(df_combine), sg=0, min_count=1, size=DIMENSION, window=50)
store(f'{PICKLE_PATH}/corpus_embedding', word2vec)
