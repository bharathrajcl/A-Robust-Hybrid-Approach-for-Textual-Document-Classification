# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 14:08:51 2021

@author: Bharathraj C L
"""



from numpy import array
from tensorflow.keras.models import load_model
import util_code
from sklearn.model_selection import train_test_split
import model_build
import json
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

with open('training_config_data.json') as data:
    training_config_data = json.load(data)


trainLines, trainLabels = util_code.load_dataset(training_config_data['preprocessed_dataset_training'])


tokenizer = util_code.create_tokenizer(trainLines)
util_code.save_tokenizer_vector(tokenizer,training_config_data['tokenizer_path'])
length = training_config_data['length']
#training_config_data['length'] = length
vocab_size = len(tokenizer.word_index)+1

trainX = util_code.encode_text(tokenizer,trainLines,length)

lb = LabelEncoder()
trainLabels = lb.fit_transform(trainLabels)
util_code.save_tokenizer_vector(lb,training_config_data['labelencoder_path'])
trainLabels = to_categorical(trainLabels)

if(training_config_data['select_model'] == 1 and training_config_data['use_saved_model1'] == True):
    model = tf.keras.models.load_model(training_config_data['model1_path'])
elif(training_config_data['select_model'] == 1 and training_config_data['use_saved_model1'] == False):
    model = model_build.define_model1(length,vocab_size)
elif(training_config_data['select_model'] == 2 and training_config_data['use_saved_model2'] == True):
    model = tf.keras.models.load_model(training_config_data['model2_path'])
elif(training_config_data['select_model'] == 2 and training_config_data['use_saved_model2'] == False):
    model = model_build.define_model1(length,vocab_size)
elif(training_config_data['select_model'] == 3 and training_config_data['use_saved_model3'] == False):
    model = model_build.define_model1(length,vocab_size)
elif(training_config_data['select_model'] == 3 and training_config_data['use_saved_model3'] == True):
    model = tf.keras.models.load_model(training_config_data['model3_path'])

x_train,x_test,y_train,y_test = train_test_split(trainX,trainLabels,test_size=.1,random_state=2000)
def train_save_model1(training_config_data):
    global x_train
    global x_test
    global y_train
    global y_test
    global model
    model.fit([x_train,x_train,x_train],array(y_train),validation_data=([x_test,x_test,x_test],y_test),epochs = training_config_data['epochs'],batch_size=training_config_data['batch_size'])
    model.save(training_config_data['model1_path'])
    



def train_save_model2(training_config_data):
    global x_train
    global x_test
    global y_train
    global y_test
    global model
    model.fit([x_train,x_train,x_train],array(y_train),validation_data=([x_test,x_test,x_test],y_test),epochs = training_config_data['epochs'],batch_size=training_config_data['batch_size'])
    model.save(training_config_data['model2_path'])


def train_save_model3(training_config_data):
    global x_train
    global x_test
    global y_train
    global y_test
    global model
    model.fit([x_train,x_train,x_train],array(y_train),validation_data=([x_test,x_test,x_test],y_test),epochs = training_config_data['epochs'],batch_size=training_config_data['batch_size'])
    model.save(training_config_data['model3_path'])
    
    
if(training_config_data['select_model'] == 1):
    train_save_model1(training_config_data)
if(training_config_data['select_model'] == 2):
    train_save_model2(training_config_data)
if(training_config_data['select_model'] == 3):
    train_save_model3(training_config_data)
    
