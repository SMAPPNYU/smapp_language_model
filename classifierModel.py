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

if __name__ == "__main__":
    BATCH_SIZE = 10

    # GPU setup
    #use_gpu = torch.cuda.is_available()
    #device_num = 0
    #device = torch.device(f"cuda:{device_num}" if use_gpu else "cpu")
    #print(device)

    DATA_DIR = '../data/yelp/'
    INFILE_PATH = os.path.join(DATA_DIR, 'train.csv')

    train = pd.read_csv(INFILE_PATH)

    vectorizer = pickle.load(open("lm_vectorizer.pkl", "rb"))
    train_ds = TextDataset(data=train, vectorizer=vectorizer, 
                       text_col='text', label_col='useful')

    clf_dl = CLFDataLoader(dataset=train_ds, batch_size=BATCH_SIZE)
    # Test the iterators
    clf_it = iter(clf_dl)
    x, y = next(clf_it)
    #print(x.shape)
    #print(x[0].shape)
    #print(y.shape)
    #print(y[0])

    d = torch.load(DATA_DIR+"models/LM__2019-01-24.json")
    for i in d:
        print(i)
    del d["decoder.weight"]
    del d["decoder.bias"]

    m = EncoderModel(vectorizer)
    for i in m.state_dict():
        print(i)
    m.load_state_dict(d)
    m.requires_grad = False

    c = ClassifierModel()

    final = nn.Sequential(m,c)
    finalOutput = final.forward(x)
    #print("Shape of final outputs: ",finalOutput.shape)
    #print("Final outputs: ",finalOutput)

    optimizer = torch.optim.SGD(final.parameters(), lr = 0.05)
    criterion = torch.nn.MSELoss()
    for i in range(0, 10):
        final.zero_grad()
        x, y = next(clf_it)
        res = final(x)
        #print(predictions)
        #print(labels)
        y = torch.tensor(y, dtype=torch.float)
        error = criterion(res, y)
        error.backward()
        optimizer.step()
        print("Batch: ", i+1, " Error : ", error)
        
