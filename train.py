# -*- coding: utf-8 -*-
"""
Created on Thu May 19 14:52:15 2022

@author: Alfiqmal
"""

# =============================================================================
# TRAIN SCRIPT
# =============================================================================

#%% PACKAGES

import os
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
from nlp_modules import ExploratoryDataAnalysis
from nlp_modules import ModelCreation
from nlp_modules import ModelEvaluation


#%% PATHS

TOKENIZER_JSON_PATH = os.path.join(os.getcwd(), "saved_models", 
                                   "tokenizer_data.json")
MODEL_SAVE_PATH = os.path.join(os.getcwd(), "saved_models", "model.h5")
OHE_PATH = os.path.join(os.getcwd(), "saved_models", "ohe.pkl")
LOG_PATH = os.path.join(os.getcwd(),"log")

URL = "https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv"

#%% LOAD DATA

df = pd.read_csv(URL)

text = df["text"]
category = df["category"]

#%% DATA INSPECTING AND CLEANING

# LOWER CASE CLEANING

# =============================================================================
# LOWER CASE ALL TEXTS TO EASE TOKENIZING PROCESS
# =============================================================================

# =============================================================================
# AFTER INSPECTING OUR CSV FILE, WE CAN SEE THAT "TEXT" HAVE ALL LOWER CASE
# WORDS SO, WE HAVE TO CHECK IF THERE IS ANY UPPER CASE LETTERS OR SOME SORT
# =============================================================================

eda = ExploratoryDataAnalysis()

eda.upper_checking(text)

# =============================================================================
# WE GOT FALSE FOR ANYUPPER, SO WE CAN CONCLUDE THAT WE HAVE NO UPPER CASE
# LETTER IN OUR TEXT
# =============================================================================

# SYMBOLS CLEANING

# =============================================================================
# REMOVING SYMBOLS
# =============================================================================

eda.symbol_cleaning(text)

# =============================================================================
# CHECK IF THERE ARE ANY SYMBOLS LEFT AFTER WE REMOVE
# =============================================================================

text[50]
text[1]
text[200]

# =============================================================================
# SEEMS LIKE OUR TEXT FILE IS ALREADY GOOD TO GO!
# =============================================================================

#%% TOKENIZING

text = eda.tokenizer(text, TOKENIZER_JSON_PATH)


#%% PAD SEQUENCING 

# =============================================================================
# WE MIGHT NEED TO SEE THE MAXLEN FOR OUR PAD SEQUENCING
# =============================================================================

temp_maxlen = ([np.shape(i) for i in (text)])
np.mean(temp_maxlen)

# =============================================================================
# WE HAVE OBTAINED THE VALUE OF 393.8620224719101
# SO OUR MAXLEN WOULD BE AROUND 400
# =============================================================================

text = eda.pad_sequencing(text)

#%% ONE HOT ENCODE OUR TARGETS 

category = eda.onehotencoder(category, OHE_PATH)

#%% TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(text,
                                                    category,
                                                    test_size = 0.3,
                                                    random_state = 13)

# =============================================================================
# EXPAND DIMENSION TO FIT INTO MODEL
# =============================================================================

X_train = np.expand_dims(X_train, axis = -1)
X_test = np.expand_dims(X_test, axis = -1)

# =============================================================================
# TO DECLARE THE NUMBER OF UNIQUE CATEGORY FOR OUR TARGET
# =============================================================================

category_len = np.shape(category)[1]

#%% MODEL CREATION

mc = ModelCreation()

lstm = mc.LSTM_layer(category_len)

plot_model(lstm)

lstm.compile(optimizer='adam',
              loss='categorical_crossentropy', 
              metrics='acc')

log_files = os.path.join(LOG_PATH, 
                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tensorboard_callback = TensorBoard(log_dir=log_files, histogram_freq=1)

hist = lstm.fit(X_train, y_train, epochs=5,
                 validation_data = (X_test, y_test), 
                 callbacks = tensorboard_callback)

#%% MODEL EVALUATION

predicted = np.empty([len(X_test), 5])

for index, test in enumerate(X_test):
    predicted[index,:] = lstm.predict(np.expand_dims(test, axis = 0))
    

#%% MODEL ANALYSIS 

y_pred = np.argmax(predicted, axis=1)
y_true = np.argmax(y_test, axis=1)

me = ModelEvaluation()

me.report_metrics(y_true, y_pred)

# =============================================================================
#               precision    recall  f1-score   support
# 
#            0       0.97      0.90      0.93       144
#            1       0.94      0.90      0.92       115
#            2       0.91      0.94      0.92       126
#            3       0.96      0.97      0.96       157
#            4       0.89      0.95      0.92       126
# 
#     accuracy                           0.93       668
#    macro avg       0.93      0.93      0.93       668
# weighted avg       0.93      0.93      0.93       668
# 
# [[129   0   6   2   7]
#  [  1 104   2   2   6]
#  [  1   2 118   3   2]
#  [  0   3   2 152   0]
#  [  2   2   2   0 120]]
# 0.9326347305389222
# =============================================================================

# =============================================================================
# 93.26% ACCURACY SCORE. SHOULD BE GOOD!
# =============================================================================

#%% MODEL SAVE

lstm.save(MODEL_SAVE_PATH)
