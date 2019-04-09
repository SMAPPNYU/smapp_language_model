from classifierModel import EncoderModel, ClassifierModel
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
BATCH_SIZE = 25

# GPU setup
use_gpu = torch.cuda.is_available()
device_num = 0
device = torch.device(f"cuda:{device_num}" if use_gpu else "cpu")
print(device)

DATA_DIR = '../data/imdb'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
VALID_PATH = os.path.join(DATA_DIR, 'valid.csv')

train = pd.read_csv(TRAIN_PATH)
valid = pd.read_csv(VALID_PATH)

vectorizer = pickle.load(
    open(DATA_DIR+"/models/lm_vectorizer.pkl", "rb"))
train_ds = TextDataset(data=train, vectorizer=vectorizer,
                       text_col='text', label_col='label')
valid_ds = TextDataset(data=valid, vectorizer=vectorizer,
                       text_col='text', label_col='label')

train_dl = CLFDataLoader(dataset=train_ds, batch_size=BATCH_SIZE)
valid_dl = CLFDataLoader(dataset=valid_ds, batch_size=BATCH_SIZE)

d = torch.load(DATA_DIR+"/models/LM__2019-03-22.json")
del d["decoder.weight"]
del d["decoder.bias"]

embedding_size = d['encoder.weight'].shape[1]
hidden_size = d['rnns.0.weight_hh_l0'].shape[1]
tie_weights = True
m = EncoderModel(device, vectorizer, hidden_size, embedding_size,
                 batch_size=BATCH_SIZE, tie_weights=tie_weights, 
                 bidirectional=False)
m.load_state_dict(d)
m.requires_grad = True

c = ClassifierModel(
    lm_hidden_size=embedding_size if tie_weights else hidden_size, 
    hidden_size=200, output_size=3)
final = nn.Sequential(m, c)
if use_gpu:
    final = final.to(device)


def get_accuracy(pred_probs, true_class):
    '''Calculates average accuracy over batch'''
    pred_class = torch.argmax(pred_probs, dim=1)
    errors = pred_class == y
    return torch.mean(errors.type(torch.float)).item()


optimizer = torch.optim.Adam(final.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

n_epochs = 30

for epoch in range(n_epochs):
    final.train()
    final[0].init_hidden()
    epoch_train_accs = []
    for x, y in train_dl:
        #final[0].init_hidden()
        if x.shape[0] != BATCH_SIZE:
            continue
        x = x.to(device)
        y = y.to(device)
        final.zero_grad()
        #tempResEnc = m(x)
        # print(tempResEnc.shape())
        res = final(x)
        error = criterion(res, y)
        error.backward()
        optimizer.step()
        epoch_train_accs.append(get_accuracy(res, y))
        del error
    print("Beginning eval")
    # Validation accuracy
    with torch.no_grad():
        final.eval()
        epoch_train_acc = round(np.mean(epoch_train_accs), 3)
        valid_accs = []
        for x, y in valid_dl:
            if x.shape[0] != BATCH_SIZE:
                continue
            x = x.to(device)
            y = y.to(device)
            pred_prob = final(x)
            valid_accs.append(get_accuracy(pred_prob, y))
        valid_acc = round(np.mean(valid_accs), 3)
        print(
            f'Epoch {epoch}:\n\tTraining accuracy: {epoch_train_acc}\n\tValidation accuracy: {valid_acc}')
