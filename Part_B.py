#
# Part_B.py
#
# __author__= Allen Telson
#
# This file is in conjunction with Part B Word Embedding and Neural Networks of CAP 4770  Fall 2019 - Project
# This code is developed in order to provide different neural networks that can be used with pre-trained GloVe
# or custom word embeddings. It can be used with both full and reduced Amazon data sets. After building, training and
# validating a neural network the program proceeds by providing figures that illustrate histograms associated with
# the model providing details about training accuracy, validating accuracy, training loss, and validation loss.
#
# Links to resources used:
# https://realpython.com/python-keras-text-classification/
# https://keras.io/examples/pretrained_word_embeddings/
# https://nlp.stanford.edu/projects/glove/
# https://towardsdatascience.com/word-embeddings-for-sentiment-analysis-65f42ea5d26e

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import metrics
from keras import Sequential, Model, Input
from keras.initializers import Constant
from keras.layers import Embedding, Masking, LSTM, Dense, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder

plt.style.use('ggplot')


# This function is found in link https://realpython.com/python-keras-text-classification/
# used to plot Neural networks after fitting
def plot_history(artifact):
    acc = artifact.history['categorical_accuracy']
    val_acc = artifact.history['val_categorical_accuracy']
    loss = artifact.history['loss']
    val_loss = artifact.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def create_custom_lstm():
    # create sequential model
    lstm = Sequential()

    # add embedding layer
    lstm.add(Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable=True))

    # add masking will let the model know that some part of the data is padding
    lstm.add(Masking(mask_value=0.0))

    # Recurrent layer
    lstm.add(LSTM(64, return_sequences=False,
                  dropout=0.1, recurrent_dropout=0.1))

    # Fully connected layer
    lstm.add(Dense(64, activation='relu'))

    # Output layer
    lstm.add(Dense(3, activation='softmax'))

    # Compile the model
    lstm.compile(
        optimizer='adam', loss='categorical_crossentropy',
        metrics=[metrics.mae, metrics.categorical_accuracy, metrics.cosine_proximity])

    # print contents of NN
    lstm.summary()
    return lstm


def create_glove_lstm():
    # Code used to build GloVe embedding layer
    # This code can be found via link: https://keras.io/examples/pretrained_word_embeddings/
    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    # print vocabulary size
    print('Found %s word vectors.' % len(embeddings_index))

    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    # create sequential model
    lstm = Sequential()

    # add embedding layer
    lstm.add(embedding_layer)

    # add masking will let the model know that some part of the data is padding
    lstm.add(Masking(mask_value=0.0))

    # Recurrent layer
    lstm.add(LSTM(64, return_sequences=False,
                  dropout=0.1, recurrent_dropout=0.1))

    # Fully connected layer
    lstm.add(Dense(64, activation='sigmoid'))

    # Output layer
    lstm.add(Dense(3, activation='softmax'))

    # Compile the model
    lstm.compile(
        optimizer='rmsprop', loss='categorical_crossentropy',
        metrics=[metrics.mae, metrics.categorical_accuracy, metrics.cosine_proximity])

    # print contents of NN
    lstm.summary()
    return lstm


def create_glove_cnn():
    # Code used to build GloVe embedding layer
    # This code can be found via link: https://keras.io/examples/pretrained_word_embeddings/
    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    # print vocabulary size
    print('Found %s word vectors.' % len(embeddings_index))

    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=1000,
                                trainable=False)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    # build Conv1D layers by stacking
    # add maxpool after every Conv1D layer
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    # Global Maxpooling after final convolution layer
    x = GlobalMaxPooling1D()(x)
    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    # output layer
    preds = Dense(3, activation='softplus')(x)

    cnn = Model(sequence_input, preds)
    cnn.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=[metrics.mae, metrics.categorical_accuracy, metrics.cosine_proximity])

    # print contents of NN
    cnn.summary()
    return cnn


def create_custom_nn():
    # build sequential model
    nn = Sequential()
    # add embedding layer
    nn.add(Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable=True))
    # flattens the inputs
    nn.add(Flatten())
    # add three fully connected layers
    nn.add(Dense(128, activation='tanh'))
    nn.add(Dense(64, activation='tanh'))
    # output layer
    nn.add(Dense(3, activation='softplus'))
    nn.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=[metrics.mae, metrics.categorical_accuracy, metrics.cosine_proximity])
    # print contents of NN
    nn.summary()
    return nn


def create_glove_nn():
    # Code used to build GloVe embedding layer
    # This code can be found via link: https://keras.io/examples/pretrained_word_embeddings/
    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    nn = Sequential()
    nn.add(embedding_layer)
    nn.add(Flatten())
    nn.add(Dense(128, activation='relu'))
    nn.add(Dense(64, activation='relu'))
    nn.add(Dense(3, activation='softmax'))
    nn.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=[metrics.mae, metrics.categorical_accuracy, metrics.cosine_proximity])
    nn.summary()
    return nn


def create_custom_cnn():
    # build embedding layer
    embedding_layer = Embedding(MAX_NUM_WORDS,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    # set size
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    # build Conv1D layers by stacking
    # add maxpool after every Conv1D layer
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(pool_size=5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    # Global Maxpooling after final convolution layer
    x = GlobalMaxPooling1D()(x)
    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    # output layer
    preds = Dense(3, activation='softmax')(x)
    cnn = Model(sequence_input, preds)
    cnn.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=[metrics.mae, metrics.categorical_accuracy, metrics.cosine_proximity])
    # print contents of NN
    cnn.summary()
    return cnn


# Variables used for GloVe
BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove')

# Variables used for Embedding layer and pre-processing data
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
VALIDATION_SPLIT = 0.2

# Pre-process Data
reduced_data = 'reduced_amazon_ff_reviews.csv'
full_data = 'full_amazon_ff_reviews.csv'
amazon_data = pd.read_csv(reduced_data)

# retrieve text from data frame
texts = amazon_data['Text']
ratings = amazon_data['Rating']

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='#$%&()*+-<=>@[\\]^_`{|}~\t\n',
                      lower=False, split=' ')

# fit text into tokenizer
tokenizer.fit_on_texts(texts)

# convert into sequences
sequences = tokenizer.texts_to_sequences(texts)

# word index is the vocabulary for all known words within the corpus.
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# pad data in order to have sequences at fixed length
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Create encoder to convert labels into integers
# This code was found via link: https://towardsdatascience.com/word-embeddings-for-sentiment-analysis-65f42ea5d26e
le = LabelEncoder()
ratings_sequence = le.fit_transform(ratings)

# convert data into categorical data
rating_category = to_categorical(ratings_sequence, num_classes=3)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, rating_category, test_size=VALIDATION_SPLIT,
                                                    random_state=1000)

# create model
model = create_custom_nn()

# train and validate model
history = model.fit(X_train, y_train, batch_size=300, epochs=10, validation_data=(X_test, y_test))

# plot history
plot_history(history)

# Used for cross validation
# neural_network = KerasClassifier(build_fn=create_glove_lstm, epochs=10,
#                                  batch_size=128)

# y_predict = cross_val_score(neural_network, data, rating_category, cv=5)
