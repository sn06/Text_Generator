# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 10:08:13 2018

@author: sn06
"""

import sqlite3
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils

#read data
conn = sqlite3.connect('database.sqlite')
df = pd.read_sql(con=conn,sql = 'select * from scripts')
conn.close()
text = df['detail']
text.to_csv('text.txt',index=False)
text = (open('text.txt').read())

#character mapping
characters = sorted(list(set(text)))
n_to_char = {n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}

X=[]
y=[]
length = len(text)
seq_length = 100
for i in range(0,length-seq_length):
    sequence = text[i:i+seq_length]
    label = text[i+seq_length]
    X.append([char_to_n[char] for char in sequence])
    y.append(char_to_n[label])
    
X_new = np.reshape(X,(len(X),seq_length,1))
X_new = X_new / float(len(characters))
y_new = np_utils.to_categorical(y)
del(y,text)

#create model
model = Sequential()
model.add(LSTM(400, input_shape=(X_new.shape[1],X_new.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(400))
model.add(Dropout(0.2))
model.add(Dense(y_new.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam')

model.fit(X_new,y_new,epochs=1,batch_size=100)

#text gen
string_mapped = X[555555]
for i in range(seq_length):
    x = np.reshape(string_mapped,(1,len(string_mapped),1))
    x = x / float(len(characters))
    pred_index = model.predict_classes(x,verbose=0)
    seq = [n_to_char[value] for value in string_mapped]
    string_mapped.append(pred_index[0])
    string_mapped = string_mapped[1:len(string_mapped)]
    
print(''.join(seq))
    