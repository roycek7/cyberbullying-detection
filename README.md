# CYBERBULLYING DETECTION IN SOCIAL MEDIA 

Based on Deep Learning Algorithm for Cyberbullying Detection https://pdfs.semanticscholar.org/d581/7c496cf950fd82ef6e05dfa4eaa6f27c24ec.pdf.

Kaggle: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview

# Convolutional Neural Network
### Steps

Place ```train.csv```,  ```test.csv``` and ```test_labels.csv``` in ```cyberbullying-detection/data/interim/```

Run the command below to install required libraries:
```
pip install -r requirements.txt
```

Download google's pre-trained word embeddings from https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz.
and place it in ```cyberbullying-detection/data/pretrained_word2vec/``` or create own word embedding from the corpus with file ```src/pre_process/create_embeddings.py```.

```
options = {1: 'corpus', 2: 'google_pre_trained'}
```

To visualise the embeddings run ```src/visualisations/word_embeddings.py``` with the numerical option ``` 1 | 2 ``` as an argument.

Create weights, test and training sequence of the embedding matrix with the program ```src/pre_process/tokenizer_cnn.py```. The output will be stored as pickle files.

To run the deep learning model, run ```src/models/convolutional_neural_network.py```, specify the option for the weights to be trained on in the Embedding layer. The program saves the prediction, model as csv and h5 file respectively.

To predict a line of string, run ```src/models/cnn_predict.py```, specify option and the text as input argument. Outputs prediction on the labels.

To get the confusion matrix and classification report run ```src/models/test_accuracy.py```, specify option of the prediction. Output a classification report along with confusion matrix.
