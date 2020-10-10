import logging
import os
import traceback

import pandas as pd

logger = logging.getLogger(__name__)
logger.info('Pre-processing dataset')

regex = ["Wikipedia:([^\s]+)(?#)", "WP:([^\s]+)(?#)", "http\S+", "@\S+", "@", "[^a-zA-Z]", "<.*?>"]


def file_path(folder, sub_folder, file):
    return os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        '..',
                                        os.path.pardir,
                                        folder,
                                        sub_folder,
                                        file))


def cleaned_text(dataframe, regex, field):
    dataframe[field] = dataframe[field].str.replace(rf'{regex}', ' ')
    return dataframe[field].str.lower()


def remove_whitespace(x):
    return " ".join(x.split())


def clean_dataset(dataframe):
    for reg in regex:
        dataframe['comment_text'] = cleaned_text(dataframe, reg, 'comment_text')

    # remove_whitespace to comment_text
    dataframe.comment_text = dataframe.comment_text.apply(remove_whitespace)
    return dataframe


def read_dataset(dataset):
    try:
        return pd.read_csv(file_path('data', 'interim', dataset))
    except Exception as e:
        logger.warning(traceback.print_exc())
        logger.error('Unable to read CSV: {}'.format(e))
