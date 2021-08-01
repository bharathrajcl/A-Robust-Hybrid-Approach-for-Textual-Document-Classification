# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 14:21:06 2021

@author: Bharathraj C L
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import time

def dataframe_req(data):
    frequent_list = []
    for each_sent in data:
        each_sent = each_sent[0].lower()
        each_sent_list = each_sent.split(' ')
        frequent_list.append(dict(Counter(each_sent_list)))
        
    lb = LabelEncoder()
    labels = []
    for each_label in data:
        labels.append(each_label[1])
    
    target_out = lb.fit_transform(labels)
    
    return frequent_list,target_out


def get_unique_words(frequent_list):
    words = []
    for i in frequent_list:
        words.extend(i.keys())
    words = list(set(words))
    
    return words


def create_data_required(words,target_out):
    act_word_dict = {}
    for i in words:
        act_word_dict[i] = 0
    uni_target = list(set(target_out))
    act_data_temp = []
    for i in range(0,len(uni_target)):
        act_data_temp.append([act_word_dict.copy(),act_word_dict.copy()])
    
    uni_target_copy = uni_target.copy()
    for count,i in enumerate(uni_target):
        po = []
        ne = []
        po.append(i)
        for j in uni_target_copy:
            if(j not in po):
                ne.append(j)
        pot = {'class_positive_data':po}
        neg = {'class_negative_data':ne}
        act_data_temp[count][0].update(pot)
        act_data_temp[count][1].update(neg)
    
    return act_data_temp

def custom_countvector(target_out,frequent_list,act_data_temp):
    for count,index in enumerate(target_out):
        each_sent = frequent_list[count]
        label = index
        for data_index in act_data_temp:
            for each_word in each_sent:
                if(label in data_index[0]['class_positive_data']):
                    data_index[0][each_word] = data_index[0][each_word]+each_sent[each_word]
                if(label in data_index[1]['class_negative_data']):
                    data_index[1][each_word] = data_index[0][each_word]+each_sent[each_word]
    
    return act_data_temp

def get_label_count(dict_list,labels_count):
    positive,negative = dict_list[0]['class_positive_data'],dict_list[1]['class_negative_data']
    p_count = labels_count[positive[0]]
    n_count = 0
    for i in negative:
        n_count = n_count + labels_count[i]
        
    return p_count,n_count


def ndm_logic(target_out,words,act_data_temp):
    labels_count = dict(Counter(target_out))
    uni_target = list(set(target_out))
    result = np.zeros((len(uni_target),len(words)))
    for count_col,i in enumerate(words):
        pre_nmr = 0
        nmr = 0
        each_word = i
        for count_row,j in enumerate(act_data_temp):
            tp = j[0][each_word]
            fp = j[1][each_word]
            positive_count,negative_count = get_label_count(j, labels_count)
            fn = positive_count - tp
            tn = negative_count - fp
            tpr = tp/(tp+fp)
            fpr = fp/(fp+fn)
            if(min(tpr,fpr) != 0):
                pre_nmr = abs(tpr-fpr)
                nmr = pre_nmr/min(tpr,fpr)
            else:
                nmr = 0.001
            result[count_row][count_col] = nmr
    
    df_result = pd.DataFrame(result,columns = [words])
    df_result.to_csv('ndm_result.csv',index=False)
    
    return df_result

def cleaning_col(df,cleaning_condition,consider_percentage):
    row_count = len(df.index)
    df_col = df.columns
    col_count = {}
    for i in df_col:
        col_data = df[i]
        check = col_data.sum()
        if(check > row_count*0.001*1000):
            col_count[i[0]] = check
        
    col_count = sorted(col_count,key = col_count.get,reverse= True)
    dict_count = len(col_count)
    
    result_text = col_count
    
    return result_text,col_count

def arrange_used(result_text):
    result_dict = {}
    for i in result_text:
        result_dict[i.strip()[0]] = []
    for i in result_text:
        result_dict[i.strip()[0]].append(i)
        
    return result_dict

def ndm_main(data):
    frequent_list,target_out = dataframe_req(data)
    words = get_unique_words(frequent_list)
    act_data_temp = create_data_required(words, target_out)
    act_data_temp = custom_countvector(target_out, frequent_list, act_data_temp)
    df_result = ndm_logic(target_out, words, act_data_temp)
    cleaning_condition = .1
    consider_percentage = .1
    result_text,col_count = cleaning_col(df_result, cleaning_condition, consider_percentage)
    result_dict = arrange_used(result_text)
    return result_dict,result_text