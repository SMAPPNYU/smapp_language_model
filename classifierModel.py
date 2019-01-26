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

    def forward(self, input):
        #print(input)
        #print("Input shape received to Encoder: ",input.shape)
        x = self.encoder(input)
        #print("Input shape after embeddings: ", x.shape)
        lstm1_output, last_hidden_state = self.lstm1(x)
        lstm2_output, last_hidden_state = self.lstm2(self.dropout(lstm1_output), last_hidden_state)
        lstm3_output, last_hidden_state = self.lstm3(self.dropout(lstm2_output), last_hidden_state)
        return lstm3_output[:, -1]

class ClassifierModel(torch.nn.Module):
    def  __init__(self):
        super(ClassifierModel, self).__init__()
        self.softmaxProb = nn.Softmax(dim=0)
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(100,100)
        self.linear2 = nn.Linear(100,1)
    
    def forward(self, input):
        h1 = self.linear1(input)
        h2 = self.linear2(self.activation(h1))
        return self.softmaxProb(h2)

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
    
#if __name__=="__main__":
#    main()
