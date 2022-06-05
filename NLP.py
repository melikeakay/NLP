# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:56:21 2022

@author: melike
"""

import pandas as pd
import numpy as np
import os
import string
import gensim

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.layers import Dense,LSTM,GRU

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences




df = pd.read_csv('movie_data.csv')
print(df.head(3))


review_lines = list()
lines = df['review'].values.tolist()


nltk.download('punkt')
nltk.download('stopwords')
for line in lines:
    tokens = word_tokenize(line)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('','',string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    review_lines.append(words)
    
len(review_lines)

EMBEDDING_DIM = 100

sentences=review_lines
model = gensim.models.Word2Vec(vector_size= EMBEDDING_DIM,window=5,workers=8,min_count=1)

model.build_vocab(sentences)
model.train(sentences, total_examples=model.corpus_count,epochs=model.epochs)   

words=list(model.wv.index_to_key)
print('Vocabulary size: %d' % len(words))
filename='imdb_embedding_word2vec.txt'
model.wv.save_word2vec_format(filename,binary=False)

embeddings_index = {}
f = open(os.path.join('','imdb_embedding_word2vec.txt'), encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_index[word] = coefs
f.close()

tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(review_lines)
sequences = tokenizer_obj.texts_to_sequences(review_lines)


word_index = tokenizer_obj.word_index 
print('Found %s unique tokens. ' % len(word_index))

review_pad = pad_sequences(sequences,maxlen=2678)
sentiment = df['sentiment'].values

num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words,EMBEDDING_DIM))

for word,i in word_index.items():
    if i > num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print(num_words)


print('Build Model...')

model = Sequential()
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer = Constant(embedding_matrix),
                            input_length=2678,
                            trainable=False)
model.add(embedding_layer)
model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1,activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

VALIDATION_SPLIT = 0.2
indices = np.arange(review_pad.shape[0])
np.random.shuffle(indices)
review_pad = review_pad[indices]
sentiment = sentiment[indices]
num_validation_samples = int(VALIDATION_SPLIT * review_pad.shape[0])

X_train_pad = review_pad[:-num_validation_samples]
y_train = sentiment[:-num_validation_samples]
X_test_pad = review_pad[-num_validation_samples:]
y_test = sentiment[-num_validation_samples:]


print('Train...')

model.fit(X_train_pad, y_train,batch_size=300, epochs=1, validation_data=(X_test_pad,y_test),verbose=2)


     