import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/pre_process/')
from pre_processing import read_dataset
from config import PICKLE_PATH, labels, CORPUS, options
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix

pretrained = 'google_pre_trained'
corpus = 'corpus'


def run(argv):
    try:
        if len(argv) != CORPUS:
            raise Exception
        else:
            if isinstance(int(argv[0]), int):
                choice = int(argv[0])
                test_labels = read_dataset('test_labels.csv')

                prediction = pretrained if choice == CORPUS else corpus

                y_pred = pd.read_csv(f'{PICKLE_PATH}/{prediction}_prediction.csv')
                y_pred = y_pred.round()

                test_labels = test_labels[
                    (test_labels['toxic'] >= 0) & (test_labels['severe_toxic'] >= 0) & (test_labels['obscene'] >= 0) &
                    (test_labels['threat'] >= 0) & (test_labels['insult'] >= 0) & (test_labels['identity_hate'] >= 0)]

                y_pred = y_pred.drop(y_pred[~y_pred['id'].isin(test_labels['id'])].index)

                y_pred = np.array(y_pred.values)
                test_labels = np.array(test_labels.values)

                y_pred = np.delete(y_pred, np.s_[0], axis=1).astype(int)
                test_labels = np.delete(test_labels, np.s_[0], axis=1).astype(int)

                for i in range(len(labels)):
                    _pred = [item[i] for item in y_pred]
                    _label = [item[i] for item in test_labels]

                    matrix = confusion_matrix(y_true=_label, y_pred=_pred)
                    report = classification_report(y_true=_label, y_pred=_pred,
                                                   target_names=[f'Not {labels[i]}', f'{labels[i]}'])
                    print(f'Confusion Matrix {labels[i]}: \n{matrix} \n\nClassification Report {labels[i]}: \n{report}')

                combined_matrix = multilabel_confusion_matrix(y_true=test_labels, y_pred=y_pred)
                combined_report = classification_report(y_true=test_labels, y_pred=y_pred, target_names=labels,
                                                        zero_division=0)
                print(f'Confusion Matrix {labels}: \n{combined_matrix} \n\nClassification Report: \n{combined_report}')
    except Exception as e:
        print(f'{str(e).upper()}\nFollow command:\npython test_accuracy.py <file> --mandatory '
              f'<choice of prediction> --mandatory')
        print(f'{options}')


if __name__ == '__main__':
    run(sys.argv[1:])
