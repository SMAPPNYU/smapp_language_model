import numpy as np
import pandas as pd
import spacy
import sys
import torch
import torch.nn as nn
from data_utils import (LanguageModelDataset, 
                        SpacyTokenizer, 
                        LanguageModelDataLoader)

data = {'text': ['hello world', 
                 'the brown FOX jumped over the lazy Dog', 
                 'where is my coffee now', 
                 'the long story of a long... document long soo long longer than the others'],
           'label': [1,1,1,1]}
data = pd.DataFrame(data)
tokenizer = SpacyTokenizer().tokenize
dataset = LanguageModelDataset(data, tokenizer=tokenizer)

lm_loader = LanguageModelDataLoader(
    dataset=dataset, batch_size=2, target_seq_len=3, shuffle=False,
    max_seq_len=5, min_seq_len=2, p_half_seq_len=0.05)

for x,y in lm_loader:
    print(x.shape)
    print(y.shape)

# Test how input to first layer of model would work
encoder = nn.Embedding(30, 5, padding_idx = 0)

it = iter(lm_loader)

x, y = next(it)
emb_batch = encoder(torch.transpose(x, 0, 1))
print(emb_batch)
