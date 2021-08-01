# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 12:57:16 2021

@author: Bharathraj C L
"""

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input,Dense,Flatten,Dropout,Embedding
from tensorflow.keras.layers import Conv1D,MaxPooling1D,concatenate
from tensorflow.keras import layers
import tensorflow.compat.v2 as tf
import tensorflow


def define_model1(length,vocab_size):
    
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size,100)(inputs1)
    conv1 = Conv1D(filters=32,kernel_size=4,activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    
    inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocab_size,100)(inputs2)
    conv2 = Conv1D(filters=32,kernel_size=6,activation='relu')(embedding2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    
    inputs3 = Input(shape=(length,))
    embedding3 = Embedding(vocab_size,100)(inputs3)
    conv3 = Conv1D(filters=32,kernel_size=6,activation='relu')(embedding3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)
    
    merged = concatenate([flat1,flat2,flat3])
    
    dense1 = Dense(10,activation='relu')(merged)
    outputs = Dense(4,activation='softmax')(dense1)
    
    model = Model(inputs=[inputs1,inputs2,inputs3],outputs = outputs)
    
    opt = tensorflow.keras.optimizers.SGD(learning_rate=0.01)
    model.compile(loss='binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
    
    #plot_model(model,show_shapes=True,to_file='multichannel.png')
    
    return model


def define_model2(length,vocab_size):
    embedding_dim = 100
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim,input_length=length))
    model.add(layers.Conv1D(128,5))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Activation('relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(4,activation='softmax'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    return model


def define_model3(length,vocab_size):
    
    
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size,100)(inputs1)
    conv1 = Conv1D(filters=32,kernel_size=4,activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    
   
    conv2 = Conv1D(filters=32,kernel_size=6,activation='relu')(embedding1)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    

    
    merged = concatenate([flat1,flat2])
    
    dense1 = Dense(128,activation='relu')(merged)
    drop3 = Dropout(0.5)(dense1)
    outputs = Dense(4,activation='softmax')(dense1)
    
    model = Model(inputs=[inputs1],outputs = outputs)
    
    opt = tensorflow.keras.optimizers.SGD(learning_rate=0.01)
    model.compile(loss='binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
    
    plot_model(model,show_shapes=True,to_file='multichannel.png')
    
    return model