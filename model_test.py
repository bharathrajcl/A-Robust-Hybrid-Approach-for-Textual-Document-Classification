# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 13:48:09 2021

@author: Bharathraj C L
"""

from pickle import load
from numpy import array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import util_code
from sklearn.model_selection import train_test_split



trainLines, trainLabels = util_code.load_dataset('train_data.pkl')
#testLines,testLabels = util_code.load_dataset('test_10000.pkl')


tokenizer = util_code.create_tokenizer(trainLines)
length = util_code.max_length(trainLines)
vocab_size = len(tokenizer.word_index)+1

trainX = util_code.encode_text(tokenizer,trainLines,length)
#testX = util_code.encode_text(tokenizer,testLines,length)

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
trainLabels = lb.fit_transform(trainLabels)
#testLabels = lb.fit_transform(testLabels)

trainLabels = to_categorical(trainLabels)
#testLabels = to_categorical(testLabels)

x_train,x_test,y_train,y_test = train_test_split(trainX,trainLabels,test_size=.1,random_state=2000)
model = load_model('model2_data.h5')


loss,acc = model.evaluate([x_train],array(y_train),verbose=0)

loss,acc = model.evaluate([x_test],array(y_test),verbose=0)
