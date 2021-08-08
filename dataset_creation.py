# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 13:50:46 2021

@author: Bharathraj C L
"""

import os
import pandas as pd
from nltk.corpus import stopwords

import json
with open('training_config_data.json') as data:
    training_config_data = json.load(data)
    
path = training_config_data['data_folder_path']
temp_path = training_config_data['raw_dataset_folder_path']
data_list = os.listdir(temp_path)
act_data_list = [x for x in data_list if(x.split('.')[-1] == 'tsv')]

punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''



def remove_punct(each_token):
    no_punct = ""
    for char in each_token:
        if char not in punctuations:
            no_punct = no_punct+char
        else:
            no_punct = no_punct+' '
    return no_punct

stop_words = set(stopwords.words('english'))
def clean_doc(doc):
    tokens = doc.split()
    tokens = [remove_punct(word) for word in tokens]
    tokens = ' '.join(tokens).split(' ')
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [i for i in tokens if not i.isdigit()]
    tokens = [word for word in tokens if len(word) > 2]
    tokens = ' '.join(tokens).lower()
    return tokens

    
def create_text_file(text,label,path,index):
    te = open(path+'/'+str(index)+'_'+label+'.txt','w+',encoding = 'utf-8')
    te.write(text)
    te.close()
    


def main_method(data_list,count_per_label):
    global path
    for count,i in enumerate(data_list):
        print(count,'outer')
        #print(i)
        label_data = i.split('.')[0]
        #print(label_data)
        if label_data not in []:
            df = pd.read_csv(temp_path+i,error_bad_lines=False,sep='\t')
            df = df.dropna()
            r_head = pd.DataFrame(df['review_headline'],columns =['review_headline'])
            r_body = pd.DataFrame(df['review_body'],columns = ['review_body'])
            del df
            all_data = pd.concat([r_head,r_body],axis = 1)
            all_data = all_data.values
            del r_head
            del r_body
            count = 0
            index = 0
            for each_data in all_data:
                count = count+1
                text= each_data[0]+' '+each_data[1]
                pre_text = clean_doc(text).split(' ')
                if(len(pre_text) > 100):
                    pre_text = ' '.join(pre_text)
                    create_text_file(pre_text, label_data, path, index)
                    index = index+1
                if(index > count_per_label):
                    break
                if(count%1000 == 0):
                    print(count, index)
                

            
main_method(act_data_list, training_config_data['number_raw_file_per_labels'])
    

def create_relevant_csv(file_path,output_path):
    data = os.listdir(file_path)
    act_data = []
    
    for c,i in enumerate(data):
        temp = i.split('.')[0]
        name,label = temp,temp.split('_')[-1]
        new_temp = name+';'+label
        act_data.append(new_temp)
        
    act_data = list(set(act_data))
    new_act_data = []
    for i in act_data:
        new_temp = i.split(';')
        new_act_data.append(new_temp)
    df = pd.DataFrame(new_act_data,columns=['file_name','label'])
    df.to_csv(output_path,index = False)

create_relevant_csv(training_config_data['data_folder_path'], training_config_data['base_file_path'])

