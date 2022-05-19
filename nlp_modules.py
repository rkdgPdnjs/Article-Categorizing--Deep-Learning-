# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:28:20 2022

@author: Alfiqmal
"""

# =============================================================================
# MODULES SCRIPTS
# =============================================================================

#%% PACKAGES

import re
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Embedding
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
import pickle

#%%

class ExploratoryDataAnalysis():
    
    def init(self):
        
        pass
    
    def upper_checking(self, data):
        
        for index, word in enumerate(data):
            anyupper = any(i.isupper() for i in data[index])
        print(anyupper) 

    def symbol_cleaning(self, data):
        
        for index, word in enumerate(data):
            data[index] = re.sub(r"[^\w]", " ", word)
            
    def tokenizer(self, data, token_path, num_words = 10000, 
                  oov_token = "<OOV>", prnt = False):
    
        token = Tokenizer(num_words = num_words,
                      oov_token = oov_token)
    
        token.fit_on_texts(data)
    
        token_json_file = token.to_json()
    
        with open(token_path, "w") as json_file:
            json.dump(token_json_file, json_file)
        
        word_index = token.word_index
    
        if prnt == True:
            print(dict(list(word_index.items())[0:10]))
    
        data = token.texts_to_sequences(data)
    
        return data
    
    def pad_sequencing(self, data, maxlen = 400):
        return pad_sequences(data, maxlen = maxlen,
                            padding = "post",
                            truncating = "post")
    
    def onehotencoder(self, data, path):

        ohe = OneHotEncoder(sparse = False)

        data = ohe.fit_transform(np.expand_dims(data, axis = -1))
    
        pickle.dump(ohe, open(path, "wb"))
    
        return data
    
class ModelCreation():

    
    def LSTM_layer(self, category_len, num_words = 10000, embedding_output = 64,
                   nodes = 64, dropout = 0.2):
    
        model = Sequential()
        model.add(Embedding(num_words, embedding_output))
        model.add(Bidirectional(LSTM(nodes, return_sequences = True)))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(nodes)))
        model.add(Dropout(dropout))
        model.add(Dense(category_len, activation = "softmax"))
        
        model.summary()
        
        return model
    
class ModelEvaluation():
    
    def report_metrics(self, y_true,y_pred):
   
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        print(accuracy_score(y_true, y_pred))