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
    def  __init__(self, vectorizer):
        super(EncoderModel, self).__init__()
        self.dropout = nn.Dropout(p = 0.5)
        self.encoder = nn.Embedding(vectorizer.vocabulary_size, 200, padding_idx = 0)
        self.lstm1 = nn.LSTM(200,100, num_layers=1, bidirectional = False, batch_first = True)
        self.lstm2 = nn.LSTM(100,100, num_layers=1, bidirectional = False, batch_first = True)
        self.lstm3 = nn.LSTM(100,100, num_layers=1, bidirectional = False, batch_first = True)

    def forward(self, input_):
        #print(input)
        #print("Input shape received to Encoder: ",input.shape)
        x = self.encoder(input_)
        #print("Input shape after embeddings: ", x.shape)
        lstm1_output, last_cell_state = self.lstm1(x)
        lstm2_output, last_cell_state = self.lstm2(self.dropout(lstm1_output), last_cell_state)
        return self.lstm3(self.dropout(lstm2_output), last_cell_state)
        #return last_hidden_state, lstm3_output

class ClassifierModel(torch.nn.Module):
    def  __init__(self):
        super(ClassifierModel, self).__init__()
        self.softmaxProb = nn.Softmax(dim=1)
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(300,100)
        self.linear2 = nn.Linear(100,3)
    
    def forward(self, input_):
        hidden_states = input_[0]
        mean_pool = torch.mean(hidden_states, dim=1)
        max_pool = torch.max(hidden_states, dim=1)[0]
        last_hidden = hidden_states[:, -1, :]
        concat = torch.cat([last_hidden, max_pool, mean_pool], dim=1)
        h1 = self.linear1(concat)
        h2 = self.linear2(self.activation(h1))
        return self.softmaxProb(h2)