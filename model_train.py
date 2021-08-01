# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 14:08:51 2021

@author: Bharathraj C L
"""
from pickle import load
from numpy import array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import util_code
from sklearn.model_selection import train_test_split
import model_build

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

model1 = model_build.define_model1(length,vocab_size)
model2 = model_build.define_model2(length,vocab_size)
model3 = model_build.define_model3(length,vocab_size)

x_train,x_test,y_train,y_test = train_test_split(trainX,trainLabels,test_size=.1,random_state=2000)
def train_save_model1():
    global x_train
    global x_test
    global y_train
    global y_test
    global model1
    model1.fit([x_train,x_train,x_train],array(y_train),validation_data=([x_test,x_test,x_test],y_test),epochs = 5,batch_size=100)
    model1.save('model1_data.h5')
    

train_save_model1()

def train_save_model2():
    global x_train
    global x_test
    global y_train
    global y_test
    global model2
    model2.fit([x_train],array(y_train),validation_data=(x_test,y_test),epochs = 5,batch_size=100)
    model2.save('model2_data.h5')

train_save_model2()

def train_save_model3():
    global x_train
    global x_test
    global y_train
    global y_test
    global model3
    model3.fit([x_train],array(y_train),validation_data=(x_test,y_test),epochs = 5,batch_size=100)
    model3.save('model3_data.h5')
    
train_save_model3()