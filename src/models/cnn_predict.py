import os
import re
import sys

from tensorflow.keras.models import load_model

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/pre_process/')
from config import PICKLE_PATH, MAX_LEN, MAX_WORDS, options, PRE_TRAINED
from pre_processing import regex, clean_dataset, read_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def run(argv):
    try:
        if len(argv) != PRE_TRAINED:
            raise Exception
        else:
            if isinstance(int(argv[0]), int):
                choice = int(argv[0])
                model = load_model(f'{PICKLE_PATH}/{options[choice]}_cyberbullying.h5')

                text = argv[1]
                for i in regex:
                    text = re.sub(rf'{i}', ' ', text)
                text = text.lower()
                clean_text = ' '.join(text.split())

                tokenizer = Tokenizer(MAX_WORDS, lower=True, char_level=False)
                tokenizer.fit_on_texts(clean_dataset(read_dataset('test.csv'))['comment_text'].values.tolist() +
                                       clean_dataset(read_dataset('train.csv'))['comment_text'].values.tolist())
                tokenized = tokenizer.texts_to_sequences([clean_text])
                test_sequence = pad_sequences(tokenized, maxlen=MAX_LEN)

                y_pred_test = model.predict(test_sequence)

                print(f'toxic: {y_pred_test[0][0]}, severe_toxic: {y_pred_test[0][1]}, obscene: {y_pred_test[0][2]}, '
                      f'threat: {y_pred_test[0][3]}, insult: {y_pred_test[0][4]}, identity_hate: {y_pred_test[0][5]}')
    except Exception as e:
        print(f'{str(e).upper()}\nFollow command:\npython convolutional_neural_network.py <file> --mandatory '
              f'<choice of prediction> --mandatory <text to predict> --mandatory')
        print(f'options: {options}')


if __name__ == '__main__':
    run(sys.argv[1:])
