import torch
import numpy as np
import os
import pandas as pd
import spacy
import sys
import torch.nn as nn
import torch.nn.functional as F
import pickle
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
        self.lstm1 = nn.LSTM(200,100, num_layers=1, bidirectional = True, batch_first = True)
        self.lstm2 = nn.LSTM(200,100, num_layers=1, bidirectional = True, batch_first = True)

    def maxPool(self, x, batchsize):
        #Same way can do avg pool and concatenate
        F.adaptive_max_pool1d(x.transpose(1,2), (1,)).view(batchsize,-1)

    def forward(self, input):
        #print(input)
        print("Input shape received to Encoder: ",input.shape)
        x = self.encoder(input)
        print("Input shape after embeddings: ", x.shape)
        lstm1_output, last_hidden_state = self.lstm1(x)
        lstm2_output, last_hidden_state = self.lstm2(self.dropout(lstm1_output), last_hidden_state)
        return lstm2_output[:, -1]

class ClassifierModel(torch.nn.Module):
    def  __init__(self):
        super(ClassifierModel, self).__init__()
        self.softmaxProb = nn.Softmax()
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(200,100)
        self.linear2 = nn.Linear(100,2)
    
    def forward(self, input):
        h1 = self.linear1(input)
        h2 = self.linear2(self.activation(h1))
        return self.softmaxProb(h2)

def main():
    BATCH_SIZE = 10

    # GPU setup
    use_gpu = torch.cuda.is_available()
    device_num = 0
    device = torch.device(f"cuda:{device_num}" if use_gpu else "cpu")
    device

    DATA_DIR = '../../../'
    INFILE_PATH = os.path.join(DATA_DIR, 'train.csv')
    
    train = pd.read_csv(INFILE_PATH)

    vectorizer = pickle.load(open(os.path.join(DATA_DIR,"vectorizer.pkl"), "rb"))
    train_ds = TextDataset(data=train, vectorizer=vectorizer, 
                       text_col='text', label_col='useful')

    clf_dl = CLFDataLoader(dataset=train_ds, batch_size=BATCH_SIZE)
    # Test the iterators
    clf_it = iter(clf_dl)
    x, y = next(clf_it)
    print(x.shape)
    print(x[0].shape)
    print(y.shape)
    print(y[0])
    
    d = torch.load("../../../models/LM__2019-01-09.json")
    #for i in d:
    #    print(i)
    del d["decoder.weight"]
    del d["decoder.bias"]

    m = EncoderModel(vectorizer)
    #for i in m.state_dict():
    #    print(i)
    m.load_state_dict(d)
    
    c = ClassifierModel()

    final = nn.Sequential(m,c)
    finalOutput = final.forward(x)
    print("Shape of final outputs: ",finalOutput.shape)
    print("Final outputs: ",finalOutput)
        
if __name__=="__main__":
    main()
