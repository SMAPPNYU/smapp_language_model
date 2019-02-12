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
    def  __init__(self, vectorizer, hidden_size, embedding_size, bidirectional):
        super(EncoderModel, self).__init__()
        self.dropout = nn.Dropout(p = 0.5)
        self.encoder = nn.Embedding(vectorizer.vocabulary_size, embedding_size, padding_idx = 0)
        self.lstm1 = nn.LSTM(embedding_size,hidden_size, num_layers=1, 
                             bidirectional=bidirectional, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size,hidden_size, num_layers=1, 
                             bidirectional=bidirectional, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size,hidden_size, num_layers=1, 
                             bidirectional=bidirectional, batch_first=True)

    def forward(self, input_):

        x = self.encoder(input_)
        #print("Input shape after embeddings: ", x.shape)
        lstm1_output, last_cell_state = self.lstm1(x)
        lstm2_output, last_cell_state = self.lstm2(self.dropout(lstm1_output), last_cell_state)
        lstm3_output = self.lstm3(self.dropout(lstm2_output), last_cell_state)
        return lstm3_output[0] # just return the hidden states

class ClassifierModel(torch.nn.Module):
    def  __init__(self, lm_hidden_size, hidden_size, output_size):
        super(ClassifierModel, self).__init__()
        self.softmaxProb = nn.Softmax(dim=1)
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(lm_hidden_size*3,hidden_size)
        #self.linear2 = nn.Linear(hidden_size,hidden_size)
        self.linear3 = nn.Linear(hidden_size,output_size)
        self.input_dropout = nn.Dropout(p = 0.2)
        self.hidden_dropout = nn.Dropout(p = 0.5)
    
    def forward(self, input_):
        mean_pool = torch.mean(input_, dim=1)
        max_pool = torch.max(input_, dim=1)[0]
        last_hidden = input_[:, -1, :]
        concat = torch.cat([last_hidden, max_pool, mean_pool], dim=1)
        h1 = self.linear1(self.input_dropout(concat))
        #h2 = self.linear2(self.hidden_dropout(h1))
        h3 = self.linear3(self.activation(self.hidden_dropout(h1)))
        return self.softmaxProb(h3)