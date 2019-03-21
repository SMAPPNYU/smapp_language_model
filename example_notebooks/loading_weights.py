#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import datetime
import pandas as pd
import torch
import torch.nn as nn
import sys

sys.path.append('../')
from training_utils import training_loop, test_loop
from model import RNNLM
from data_utils import (IndexVectorizer, 
                        TextDataset, 
                        SpacyTokenizer,
                        LMDataLoader)
from lr_scheduler import CyclicLR
# from data_utils import IndexVectorizer, TextDataset, simple_tokenizer


# In[ ]:


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
dropout = 0.3
lstm_tieweights = True

## Training Language Model
batch_size = 50
learning_rate = 1e-5
num_epochs = 100
display_epoch_freq = 10
target_seq_len = 65
max_seq_len = 75
min_seq_len = 5



# GPU variables
use_gpu = torch.cuda.is_available()
device_num = 0
device = torch.device(f"cuda:{device_num}" if False else "cpu")
device


directory_pre_trained_models = '../../data/weights_pretrained/'
new_data_directory = '../../data/imdb/models/'
os.makedirs(directory_pre_trained_models, exist_ok=True)



# ## Loading weights

# In[5]:


encoder_file = os.path.join(directory_pre_trained_models, 'fwd_wt103.h5')
fitos_file = os.path.join(directory_pre_trained_models, 'fitos_wt103.pkl')
vectorizer_file = os.path.join(new_data_directory, 'lm_vectorizer.pkl')


# In[7]:


enc = torch.load(encoder_file, map_location=lambda storage, loc: storage)


# In[59]:


embedding_size = enc['0.encoder.weight'].shape[1]
hidden_size = int(enc['0.rnns.0.module.weight_hh_l0_raw'].shape[0]/4)
num_layers = 3


# In[10]:


new_enc = {}
for k,v in enc.items():
    layer_detail = k.split('.')
    layer_name = layer_detail[-1].replace('_raw', '')
    if len(layer_detail) == num_layers: 
        new_enc[f'{layer_detail[1]}.{layer_name}'] = v
    else:
        new_enc[f'{layer_detail[1]}.{layer_detail[2]}.{layer_name}'] = v
    


# In[11]:


# Remove this odd element as it is the same as encoder.weight
#new_enc['encoder_with_dropout.embed.weight'] == new_enc['encoder.weight']
del new_enc['encoder_with_dropout.embed.weight']


# In[12]:


# Load our vectorizer

## Load the wikitext vocabulary
pretrained_idx2word = pickle.load(open(fitos_file, 'rb'))


# In[13]:


pretrained_word2idx = {k: i for i,k in enumerate(pretrained_idx2word)}


# In[14]:


new_model_vectorizer = pickle.load(open(vectorizer_file, 'rb'))


# In[15]:


pretrained_encoder_weights = enc['0.encoder.weight']


# In[16]:


row_m = pretrained_encoder_weights.mean(dim=0)


# In[17]:


row_m = [x.item() for x in row_m]


# In[18]:


new_vocab_size = len(new_model_vectorizer.word2idx)
new_encoder_weights = torch.tensor([row_m for i in range(new_vocab_size)])


# In[19]:


new_idx2weights = {}
for word, i in new_model_vectorizer.word2idx.items():
    if word in pretrained_word2idx:
        word_idx = pretrained_word2idx[word]
        new_encoder_weights[i] = pretrained_encoder_weights[word_idx]


# In[20]:


import copy
new_enc['encoder.weight'] = new_encoder_weights
new_enc['decoder.weight'] = copy.copy(new_encoder_weights)
new_enc['decoder.bias'] = torch.zeros(new_enc['decoder.weight'].shape[0])


# In[64]:


# example of a model
lm = RNNLM(device=device, vocab_size=new_vocab_size, 
           embedding_size=embedding_size, hidden_size=hidden_size, 
           batch_size=50, num_layers=3, tie_weights=True, word2idx = new_model_vectorizer.word2idx)


# In[65]:


lm.load_state_dict(new_enc)


# In[49]:


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


# In[50]:


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


# In[51]:


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


# In[52]:


if use_gpu: torch.cuda.manual_seed(303)
else: torch.manual_seed(303)


# In[53]:


# set up Files to save stuff in
runtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


# In[66]:



if use_gpu:
    lm = lm.to(device)
    
# Loss and Optimizer
loss = nn.CrossEntropyLoss()

# Extract pointers to the parameters of the lstms
param_list = [{'params': rnn.parameters(), 'lr': learning_rate} for rnn in lm.rnns]

# If weights are tied between encoder and decoder, we can only optimize 
# parameters in one of those two layers
if not lstm_tieweights:
    param_list.extend([
            {'params': lm.encoder.parameters(), 'lr':learning_rate},
            {'params': lm.decoder.parameters(), 'lr':learning_rate},
        ])
else:
    param_list.extend([
        {'params': lm.decoder.parameters(), 'lr':learning_rate},
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




