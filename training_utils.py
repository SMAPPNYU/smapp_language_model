import os
import re
import json
import glob
import time
import random
import datetime
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split

def log(msg):
    print(msg)
    
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class History(object):
    """Records Loss and Validation Loss"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss = dict()
        self.val_loss = dict()
        self.min_loss = 100
    
    def update_min_loss(self, min_loss):
        self.min_loss = min_loss
        
    def update_loss(self, loss):
        epoch = len(self.loss.keys())
        self.loss[epoch] = loss
    
    def update_val_loss(self, val_loss):
        epoch = len(self.val_loss.keys())
        self.val_loss[epoch] = val_loss
        
    def plot(self):
        loss = sorted(self.loss.items())
        x, y = zip(*loss)
        
        if self.val_loss:
            val_loss = sorted(self.val_loss.items())
            x1, y1 = zip(*val_loss)
            plt.plot(x, y, 'C0', label='Loss')
            plt.plot(x1, y1, 'C2', label='Validation Loss')
            plt.legend();
        else:
            plt.plot(x, y, 'C0');
    
def categorical_accuracy(y_true, y_pred):
    y_true = y_true.float()
    _, y_pred = torch.max(y_pred.squeeze(), dim=-1)
    return (y_pred.float() == y_true).float().mean()

def softmax_trick(x):
    logits_exp = torch.exp(x - torch.max(x))
    weights = torch.div(logits_exp, logits_exp.sum())
    return weights

def save_state_dict(model, filepath):
    '''Saves the model weights as a dictionary'''
    model_dict = model.state_dict()
    torch.save(model_dict, filepath)
    return model_dict
    
def run_epoch(model, dataset, criterion, optim, scheduler, batch_size, device,
              train=False, shuffle=True):
    '''A wrapper for a training, validation or test run.'''
    model.train() if train else model.eval()
    loss = AverageMeter()
    accuracy = AverageMeter()
    #loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    loader = dataset
    for X, y in loader:
        model.zero_grad() 
        X = X.squeeze().to(device)
        y = y.squeeze().contiguous().view(-1).to(device)        

        # get a predition    
        hidden = model.init_hidden(X.size(0))
        for _ in hidden:
            _.to(device)
        y_, hidden = model(X, hidden)
        
        # calculate loss and accuracy
        lossy = criterion(y_.squeeze(), y.squeeze())
        accy = categorical_accuracy(y_.squeeze().data, 
                                    y.squeeze().data)
        
        loss.update(lossy.data.item())
        accuracy.update(accy)
        
        # backprop
        if train:
            lossy.backward()
            optim.step()
            if scheduler is not None:
                scheduler.step()
    return loss.avg, accuracy.avg


def training_epoch(*args, **kwargs):
    '''Training Epoch'''
    return run_epoch(train=True, *args, **kwargs)
    
    
def validation_epoch(*args, **kwargs):
    '''Validation Epoch'''
    return run_epoch(*args, **kwargs)
 
    
def sample_lm(model, length):
    '''Samples a language model and returns generated words'''
    #start_idx = model.word2idx['<START>']
    model.eval()
    indices = model.sample()
    words = [model.idx2word[index] for index in indices]
    
    return words


def training_loop(batch_size, num_epochs, display_freq, model, criterion, 
                  optim, scheduler, device, training_set, validation_set=None, 
                  best_model_path='model', history=None):
    '''Training iteration.'''
    if not history:
        history = History()
    
    try: 
        for epoch in tqdm(range(num_epochs)):
            # scheduler goes here...
            loss, accuracy = training_epoch(model=model, dataset=training_set, 
                                            criterion=criterion, optim=optim, 
                                            scheduler=scheduler, batch_size=batch_size, 
                                            device=device)
            history.update_loss(loss)
            
            if validation_set:
                val_loss, val_accuracy = validation_epoch(model=model, dataset=validation_set, 
                                                          criterion=criterion, optim=optim, 
                                                          scheduler=scheduler, batch_size=batch_size, 
                                                          device=device)  
                history.update_val_loss(val_loss)
                if val_loss < history.min_loss:
                    save_state_dict(model, best_model_path)
                    history.update_min_loss(val_loss)
            else:
                if loss < history.min_loss:
                    save_state_dict(model, best_model_path)
                    history.update_min_loss(loss)
                
            if epoch % display_freq == 0:
                # display stats
                if validation_set:
                    log("Epoch: {:04d}; Loss: {:.4f}; Val-Loss {:.4f}; "
                        "Perplexity {:.4f}; Val-Perplexity {:.4f}".format(
                            epoch, loss, val_loss, np.exp(loss), np.exp(val_loss)))
                else:
                    log("Epoch: {:04d}; Loss: {:.4f}; Perplexity {:.4f};".format(
                            epoch, loss, np.exp(loss)))
                
                # sample from the language model
                words = sample_lm(model, 70)
                log("Sample: {}".format(' '.join(words)))
                time.sleep(1)
        
        log('-' * 89)
        log("Training complete")
        log("Lowest loss: {:.4f}".format(history.min_loss))
        
        return history
        
    except KeyboardInterrupt:
        log('-' * 89)
        log('Exiting from training early')
        log("Lowest loss: {:.4f}".format(history.min_loss))

        return history
    
def test_loop(batch_size, model, criterion, optim, scheduler, test_set, device):
    '''Data iterator for the test set'''
    model.eval()
    
    try:
        test_loss, test_accuracy = validation_epoch(model = model, dataset = test_set, criterion = criterion, optim = optim, scheduler = scheduler, batch_size = batch_size, device=device)
        log('Evaluation Complete')
        log('Test set Loss: {}'.format(test_loss))
        log('Test set Perplexity: {}'.format(np.exp(test_loss)))
        log('Test set Accuracy: {}'.format(test_accuracy))
    
    except KeyboardInterrupt:
        log('-' * 89)
        log('Exiting from testing early')
