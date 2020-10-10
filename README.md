# cyberbullying-detection
CYBER-BULLYING DETECTION IN SOCIAL MEDIA 

# Convolutional Neural Network
### Steps
Kaggle: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview

Place train.csv and test.csv in cyberbullying-detection/data/interim/

Download google's pre-trained word embeddings from https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
and place it in cyberbullying-detection/data/pretrained_word2vec/ or create word embedding from the corpus with file src/pre_process/create_embeddings.py

options_available = {1: 'corpus', 2: 'google_pre_trained'} 

To visualise the embeddings run src/visualisations/word_embeddings.py with the numerical option ( 1 | 2 ) as an argument.

Create weights, test and training sequence of the embedding matrix with the program src/pre_process/tokenizer_cnn.py. The output will be stored as pickle files.

To run the deep learning model, run src/models/convolutional_neural_network.py, specify the option for the weights to be trained on in the Embedding layer. The program saves the prediction, model as csv and h5 file respectively.

To predict a line of string, run src/models/cnn_predict.py, specify option and the text as input argument. Outputs prediction on the labels.
