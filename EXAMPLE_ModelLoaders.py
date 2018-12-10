import numpy as np
import os
import pandas as pd
import spacy
import sys
import torch
import torch.nn as nn
from data_utils import (IndexVectorizer, 
                        SpacyTokenizer, 
                        TextDataset, 
                        LMDataLoader, 
                        CLFDataLoader)

DATA_DIR = '../data/yelp'
INFILE_PATH = os.path.join(DATA_DIR, 'train.csv')

train = pd.read_csv(INFILE_PATH)
# Bogus binary labels for testing
train['label'] = np.random.randint(0, 1, size=train.shape[0])

tokenize = SpacyTokenizer().tokenize
vectorizer = IndexVectorizer(max_words = 10000, min_frequency=2,
                             start_end_tokens=True, tokenize=tokenize)
train_ds = TextDataset(data=train, vectorizer=vectorizer, 
                       text_col='text', label_col='useful')

BATCH_SIZE = 10
clf_dl = CLFDataLoader(dataset=train_ds, batch_size=BATCH_SIZE)
lm_dl = LMDataLoader(dataset=train_ds, target_seq_len=50, 
                     shuffle=True, max_seq_len=70, 
                     min_seq_len=5, p_half_seq_len=0.05,
                     batch_size=BATCH_SIZE)

# Test the iterators
clf_it = iter(clf_dl)
lm_it = iter(lm_dl)

a = next(lm_it)
print(f'Shape of lm x: {a[0].shape}')
print(f'Shape of lm y: {a[1].shape}')
a = next(clf_it)
print(f'Shape of clf x: {a[0].shape}')
print(f'Shape of clf y: {a[1].shape}')


# Test how input to first layer of model would work
encoder = nn.Embedding(len(vectorizer.vocabulary), 5, padding_idx = 0)

x, y = next(lm_it)
emb_batch = encoder(torch.transpose(x, 0, 1))
print(emb_batch)
