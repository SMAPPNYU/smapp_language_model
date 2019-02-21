import torch
import numpy as np
import os
import pandas as pd
import spacy
import sys
import torch.nn as nn
import torch.nn.functional as F
import pickle
#from sklearn.preprocessing import OneHotEncoder
from data_utils import (IndexVectorizer,
                        SpacyTokenizer,
                        TextDataset,
                        LMDataLoader,
                        CLFDataLoader)

class EncoderModel(torch.nn.Module):
    def  __init__(self, vectorizer, hidden_size, embedding_size, bidirectional = False, num_layers=3, tie_weights=False):
        super(EncoderModel, self).__init__()
        self.dropout = nn.Dropout(p = 0.5)
        self.num_directions = 1 if not bidirectional else 2
        self.tie_weights = tie_weights
        self.num_layers = num_layers
        self.encoder = nn.Embedding(vectorizer.vocabulary_size, embedding_size, padding_idx = 0)
        self.rnns = [nn.LSTM(embedding_size if l == 0 else hidden_size * self.num_directions,
                              embedding_size if (l == self.num_layers-1 and self.tie_weights) else hidden_size,
                             num_layers = 1, 
                             bidirectional = bidirectional,
                             batch_first = True) for l in range(num_layers)]
        self.rnns = nn.ModuleList(self.rnns)
        
    def forward(self, input_):

        x = self.encoder(input_)
        #print("Input shape after embeddings: ", x.shape)
        hidden = None
        for l, rnn in enumerate(self.rnns):
            output, hidden = rnn(x if l==0 else self.dropout(output), None if l==0 else hidden)
        return output # just return the hidden states

class ClassifierModel(torch.nn.Module):
    def  __init__(self, lm_hidden_size, hidden_size, output_size):
        super(ClassifierModel, self).__init__()
        self.softmaxProb = nn.Softmax(dim=1)
        self.lm_hidden_size = lm_hidden_size
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(lm_hidden_size*3,hidden_size)
        #self.linear2 = nn.Linear(hidden_size,hidden_size)
        self.linear3 = nn.Linear(hidden_size,output_size)
        self.input_dropout = nn.Dropout(p = 0.2)
        self.hidden_dropout = nn.Dropout(p = 0.5)
    
    def forward(self, input_):
        #print(input_.shape)
        mean_pool = torch.mean(input_[:,:, :self.lm_hidden_size], dim=1)
        max_pool = torch.max(input_[:,:, :self.lm_hidden_size], dim=1)[0]
        last_hidden = input_[:, -1, :self.lm_hidden_size]
        concat = torch.cat([last_hidden, max_pool, mean_pool], dim=1)
        #print(concat.shape)
        h1 = self.linear1(self.input_dropout(concat))
        #h2 = self.linear2(self.hidden_dropout(h1))
        h3 = self.linear3(self.activation(self.hidden_dropout(h1)))
        return self.softmaxProb(h3)
