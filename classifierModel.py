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
from model import _detach

class EncoderModel(torch.nn.Module):
    def  __init__(self, device, vectorizer, hidden_size, embedding_size, bidirectional = False, batch_size = 50, num_layers=3, tie_weights=False):
        super(EncoderModel, self).__init__()
        self.dropout = nn.Dropout(p = 0.5)
        self.num_directions = 1 if not bidirectional else 2
        self.tie_weights = tie_weights
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.encoder = nn.Embedding(vectorizer.vocabulary_size, embedding_size, padding_idx = 0)
        self.rnns = [nn.LSTM(embedding_size if l == 0 else hidden_size * self.num_directions,
                              embedding_size if (l == self.num_layers-1 and self.tie_weights) else hidden_size,
                             num_layers = 1, 
                             bidirectional = bidirectional,
                             batch_first = True) for l in range(num_layers)]
        self.rnns = nn.ModuleList(self.rnns)
        self.hidden = [self._reset_hidden_layer(layer_index=l) for l in range(num_layers)]

    def _reset_hidden_layer(self, layer_index, bsz=None):
        '''
        Resets (or initalizes) the initial hidden (h0) and output (c0) for an LSTM.
        Returns a tuple of tensors!
        '''
        if bsz == None: 
            bsz = self.batch_size
        if layer_index == self.num_layers - 1 and self.tie_weights:
            dim = self.embedding_size
        else:
            dim = self.hidden_size
        h0 = torch.zeros(self.num_directions, bsz, 
                         dim).to(self.device)
        c0 = torch.zeros(self.num_directions, bsz, 
                         dim).to(self.device)
        return (h0, c0)

    def forward(self, input_):

        output = self.encoder(input_)
        #print("Input shape after embeddings: ", x.shape)
        new_hidden = []
        for l, rnn in enumerate(self.rnns):
            #print(l, x.shape)
            #if hidden is not None:
            #    print(hidden.shape)
            output, hidden = rnn(self.dropout(output) if l!= self.num_layers-1 else output, self.hidden[l])
            #print(output.shape, hidden[0].shape)
            new_hidden.append(hidden)
        self.hidden = [_detach(h, cpu=False) for h in new_hidden]
        #self.hidden = [h for h in new_hidden]
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
