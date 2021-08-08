# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 13:48:09 2021

@author: Bharathraj C L
"""



from numpy import array
import util_code
import json
import tensorflow as tf

with open('training_config_data.json') as data:
    training_config_data = json.load(data)
testLines, testLabels = util_code.load_dataset(training_config_data['preprocessed_dataset_testing'])


tokenizer = util_code.load_tokenizer_vector(training_config_data['tokenizer_path'])
vocab_size = len(tokenizer.word_index)+1
length = training_config_data['length']
testX = util_code.encode_text(tokenizer,testLines,length)

from tensorflow.keras.utils import to_categorical

lb = util_code.load_tokenizer_vector(training_config_data['labelencoder_path'])
testLabels = lb.transform(testLabels)
testLabels = to_categorical(testLabels)


if(training_config_data['select_model'] == 1):
    model = tf.keras.models.load_model(training_config_data['model1_path'])
    loss,acc = model.evaluate([testX,testX,testX],array(testLabels),verbose=0)
elif(training_config_data['select_model'] == 2):
    model = tf.keras.models.load_model(training_config_data['model2_path'])
    loss,acc = model.evaluate([testX],array(testLabels),verbose=0)
elif(training_config_data['select_model'] == 3):
    model = tf.keras.models.load_model(training_config_data['model3_path'])
    loss,acc = model.evaluate([testX],array(testLabels),verbose=0)


