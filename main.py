import regex as re
import string
import numpy as np
import random
import pandas as pd



from wordcloud import WordCloud, STOPWORDS


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from tqdm import tqdm
import os
import nltk
import random


from collections import defaultdict
from collections import Counter

import keras

from keras.layers import (LSTM,
                          Embedding,
                          BatchNormalization,
                          Dense,
                          TimeDistributed,
                          Dropout,
                          Bidirectional,
                          Flatten,
                          GlobalMaxPool1D)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from sklearn import metrics
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    accuracy_score
)
from sklearn.pipeline import Pipeline
df = pd.read_csv("\Users\ritik\PycharmProjects\NLP using LSTM\Tweets CSV\tweets.csv", encoding="latin-1")


df = df.dropna(how="any", axis=1)

#finding the text length and creating a new column to save it
df['len_of_text'] = df['text'].apply(lambda x: len(x.split(' ')))

df.head()


# the source for the included function is https://www.kaggle.com/andreshg/nlp-glove-bert-tf-idf-lstm-explained


# This is used to clean the text
# removing URL
def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


# removing emojis
def remove_emoji(text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# removing html text
def remove_html(text):
    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', text)


# removing improper data using regex
'''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''


def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        '',
        text
    )
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)

    text = remove_url(text)
    text = remove_emoji(text)
    text = remove_html(text)

    return text


'''This function makes the data cleaning process more concise as it removes all 
the words which add very little value to the choosing of write sentences 
for example 'can', 'be', 'in' etc. '''

no_value_words = stopwords.words('english')  # a seperate dictionary sort of thing which stores all these words
more_words = ['u', 'im', 'c']  # these words were further observed in the data
stop_words = no_value_words + more_words

# stemmers here are the algorithms which help find the root word involved
stemmer = nltk.SnowballStemmer("english")
def pr_data(text):
    text = clean_text(text)
    text = ' '.join(stemmer.stem(word) for word in text.split(' ') if word not in stop_words)

    return text

'''A new column is created to display results which are derived
after using stemming and lemmization'''
df['clean_text'] = df['text'].apply(pr_data)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(df['target'])

df['target_record_2'] = le.transform(df['target'])


x = df['clean_text']
y = df['target']


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42,test_size = 0.2)
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))

train_tweets = df['clean_text'].values
test_tweets = df['clean_text'].values
train_target = df['target'].values

# Calculating the length of our vocabulary
word_tokenizer = Tokenizer()

word_tokenizer.fit_on_texts(train_tweets)

vocab_length = len(word_tokenizer.word_index) + 1
vocab_length

def embed(corpus):
    return word_tokenizer.texts_to_sequences(corpus)

longest_train = max(train_tweets, key=lambda sentence: len(word_tokenize(sentence)))
length_long_sentence = len(word_tokenize(longest_train))

train_padded_sentences = pad_sequences(
    embed(train_tweets),
    length_long_sentence,
    padding='post'
)
test_padded_sentences = pad_sequences(
    embed(test_tweets),
    length_long_sentence,
    padding='post'
)

train_padded_sentences

em_dictionary = dict()
em_dim = 100

embed_matrix = np.zeros((vocab_length, em_dim))

for word, index in word_tokenizer.word_index.items():
    embed_vector = em_dictionary.get(word)
    if embed_vector is not None:
        embed_matrix[index] = embed_vector

embed_matrix


def glove_lstm():
    model = Sequential()

    model.add(Embedding(
        input_dim=embed_matrix.shape[0],
        output_dim=embed_matrix.shape[1],
        weights=[embed_matrix],
        input_length=length_long_sentence
    ))

    model.add(Bidirectional(LSTM(
        length_long_sentence,
        return_sequences=True,
        recurrent_dropout=0.2
    )))

    model.add(GlobalMaxPool1D())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(length_long_sentence, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(length_long_sentence, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model


m = glove_lstm()
m.summary()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(train_padded_sentences,train_target,test_size=0.20)

# using LSTM with glove to resolve the model and predict results
m = glove_lstm()

checkpoint = ModelCheckpoint(
    'm.h5',
    monitor = 'val_loss',
    verbose = 1,
    save_best_only = True
)
reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.2,
    verbose = 1,
    patience = 5,
    min_lr = 0.001
)
history = m.fit(
    X_train,
    y_train,
    epochs = 7,
    batch_size = 32,
    validation_data = (X_test, y_test),
    verbose = 1,
    callbacks = [reduce_lr, checkpoint]
)
