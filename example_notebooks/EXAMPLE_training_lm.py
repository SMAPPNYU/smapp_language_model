#!/usr/bin/env python
# coding: utf-8

import os
import sys
import datetime
import pandas as pd
import torch
import torch.nn as nn
sys.path.append('../')
from lr_scheduler import CyclicLR
from training_utils import training_loop, test_loop
from model import RNNLM
from data_utils import (IndexVectorizer, 
                        TextDataset, 
                        SpacyTokenizer,
                        LMDataLoader)
import pickle


####################################################
# Config
####################################################

## Input / output
data_dir = '../../data/imdb'

## Tokenization
TOKENIZE = SpacyTokenizer().tokenize

## Vectorization
MIN_WORD_FREQ = 2
MAX_VOCAB_SIZE = 20000
STAT_END_TOK = True

## Model Architecture
hidden_dim = 1150
embedding_dim = 400
dropout = 0.3
lstm_layers = 3
lstm_bidirection = False
lstm_tie_weights = False

## Training Language Model
batch_size = 50
learning_rate = 1e-3
num_epochs = 100
display_epoch_freq = 10
target_seq_len = 65
max_seq_len = 75
min_seq_len = 5

# GPU setup
use_gpu = torch.cuda.is_available()
device_num = 0
device = torch.device(f"cuda:{device_num}" if use_gpu else "cpu")
device


# IO setup
today = datetime.datetime.now().strftime('%Y-%m-%d')
model_cache_dir = os.path.join(data_dir, 'models')
data_cache = os.path.join(model_cache_dir, 'data_cache.pkl')
vectorizer_cache = os.path.join(model_cache_dir, 'lm_vectorizer.pkl')
os.makedirs(model_cache_dir, exist_ok=True)
model_file_lm = os.path.join(model_cache_dir, f'LM__{today}.json')
model_file_class = os.path.join(model_cache_dir, f'CLASS__{today}.json')

train_file = os.path.join(data_dir, 'train.csv')
valid_file = os.path.join(data_dir, 'valid.csv')





RE_VECTORIZE = False
if RE_VECTORIZE or not os.path.isfile(data_cache):
    train = pd.read_csv(train_file)
    valid = pd.read_csv(valid_file)
    vectorizer = IndexVectorizer(max_words = MAX_VOCAB_SIZE, 
                             min_frequency=MIN_WORD_FREQ,
                             start_end_tokens=STAT_END_TOK, 
                             tokenize=TOKENIZE)
    train_ds = TextDataset(data=train, vectorizer=vectorizer, text_col='text')
    valid_ds = TextDataset(data=valid, vectorizer=vectorizer, text_col='text')
    pickle.dump([train_ds, valid_ds], open(data_cache, 'wb'))
    pickle.dump(vectorizer, open(vectorizer_cache, 'wb'))
else:
    train_ds, valid_ds = pickle.load(open(data_cache, 'rb'))
    vectorizer = pickle.load(open(vectorizer_cache, 'rb'))





print(f'Train size: {len(train_ds)}\nvalid size: {len(valid_ds)}')
print(f"Vocab size: {len(vectorizer.vocabulary)}")

train_dl = LMDataLoader(dataset=train_ds, 
                        target_seq_len=target_seq_len, 
                        shuffle=True, 
                        max_seq_len=max_seq_len, 
                        min_seq_len=min_seq_len, 
                        p_half_seq_len=0.05,
                        batch_size=batch_size)
valid_dl = LMDataLoader(dataset=valid_ds,
                        target_seq_len=target_seq_len, 
                        shuffle=True, 
                        max_seq_len=max_seq_len, 
                        min_seq_len=min_seq_len, 
                        p_half_seq_len=0.05,
                        batch_size=batch_size) 


if use_gpu: torch.cuda.manual_seed(303)
else: torch.manual_seed(303)

# set up Files to save stuff in
runtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
model_file_lm = model_file_lm
    
# Build and initialize the model
lm = RNNLM(device, vectorizer.vocabulary_size, embedding_dim, hidden_dim, batch_size, 
           dropout = dropout, 
           tie_weights = lstm_tie_weights, 
           num_layers = lstm_layers, 
           bidirectional = lstm_bidirection, 
           word2idx = vectorizer.word2idx,
           log_softmax = False)

if use_gpu:
    lm = lm.to(device)
    
# Loss and Optimizer
loss = nn.CrossEntropyLoss()

# Extract pointers to the parameters of the lstms
param_list = [{'params': rnn.parameters(), 'lr': 1e-3} for rnn in lm.rnns]

# If weights are tied between encoder and decoder, we can only optimize 
# parameters in one of those two layers
if not lstm_tie_weights:
    param_list.extend([
            {'params': lm.encoder.parameters(), 'lr':1e-3},
            {'params': lm.decoder.parameters(), 'lr':1e-3},
        ])
else:
    param_list.extend([
        {'params': lm.decoder.parameters(), 'lr':1e-3},
    ])

optimizer = torch.optim.Adam(param_list, lr=0.01)

scheduler = CyclicLR(optimizer,  max_lrs=[0.1, 0.1, 0.1, 0.1, 0.1], 
                     mode='ulmfit', ratio=1.5, cut_frac=0.4, 
                     n_epochs=num_epochs, batchsize=50000/1171, 
                     verbose=False, epoch_length=50000)

history = training_loop(batch_size=batch_size, 
                        num_epochs=num_epochs,
                        display_freq=1, 
                        model=lm, 
                        criterion=loss,
                        optim=optimizer,
                        scheduler=None,
                        device=device,
                        training_set=train_dl,
                        validation_set=valid_dl,
                        best_model_path=model_file_lm,
                        history=None)

