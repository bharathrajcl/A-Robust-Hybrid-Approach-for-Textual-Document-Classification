# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 16:47:53 2021

@author: Bharathraj C L
"""

from os import listdir
from pickle import  dump
import util_code
import ndm_code
import pandas as pd
import json
with open('training_config_data.json') as data:
    training_config_data = json.load(data)

def load_doc(filename):
    file = open(filename,'r',encoding='utf8')
    text = file.read()
    file.close()
    return text

def retrieve_data(directory):
    global training_config_data
    data_list = pd.read_csv(training_config_data['base_file_path'])
    data_list = data_list.values
    documents = list()
    for count,filename in enumerate(listdir(directory)):
        if(count%1000 == 0):
            print(count)
        path = directory+'/'+filename
        doc = load_doc(path)
        review,label = doc,data_list[count][1]
        tokens = util_code.process_docs(review)
        documents.append([tokens,label])    
    return documents

def filter_ndm(documents_act,result_dict):
    
    for each_count,each_data in enumerate(documents_act):
        temp_sent = []
        each_word_list = each_data[0].split(' ')
        for each_word in each_word_list:
            try:
                if(each_word in result_dict[each_word[0]]):
                    temp_sent.append(each_word)
            except:
                print(each_word)
                continue
        each_data[0] = ' '.join(temp_sent)
        if(each_count%1000 == 0):
            print(each_count)
        documents_act[each_count] = each_data
        
    return documents_act

def save_dataset(dataset,filename):
    dump(dataset, open(filename,'wb'))
    
is_ndm_flag = True
train_datapath = training_config_data['data_folder_path']
#test_datapath= './data_10000/test/'
'''
if(is_ndm_flag):
    documents = retrieve_data(train_datapath)
    document_copy = documents.copy()
    result_dict,result_text = ndm_code.ndm_main(document_copy)
    document_copy = documents.copy()
    documents1 = filter_ndm(document_copy, result_dict)
else:
    documents = retrieve_data(train_datapath)
'''
documents = retrieve_data(train_datapath)
document_copy = documents.copy()
result_dict,result_text = ndm_code.ndm_main(document_copy)
document_copy = documents.copy()
documents1 = filter_ndm(document_copy, result_dict)
save_dataset(documents1, 'train_data.pkl')

#documents = retrieve_data(test_datapath)
#save_dataset(documents, 'test_10000.pkl')