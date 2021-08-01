# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 16:27:21 2021

@author: Bharathraj C L
"""

from nltk.corpus import stopwords
from pickle import load
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_dataset(filename):
    document = load(open(filename,'rb'))
    reviews = []
    labels = []
    for i in document:
        reviews.append(i[0])
        labels.append(i[1])
    return reviews,labels

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
    tokens = ' '.join(tokens)
    return tokens

        
        
def process_docs(document):
    document = clean_doc(document)
    return document

def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def max_length(lines):
    return max([len(s.split()) for s in lines])

def encode_text(tokenizer,lines,length):
    encoded = tokenizer.texts_to_sequences(lines)
    padded = pad_sequences(encoded,maxlen=length,padding='post')
    return padded
