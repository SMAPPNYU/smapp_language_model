import os
import re
import json
import glob
import time
import random
import datetime
import spacy
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
from scipy.stats import truncnorm

class SpacyTokenizer:
    def __init__(self):
        self.nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])

    @staticmethod
    def rescue_hashtags(token_list):
        tokens = iter(token_list)
        return([t + next(tokens, '') if t == '#' else t for t in tokens])

    def tokenize(self, text):
        return self.rescue_hashtags([x.orth_.lower() for x in self.nlp(text)])


class IndexVectorizer:
    """
    Transforms a Corpus into lists of word indices.
    
    Arguments
    ----------
    :param max_words: Maximum vocabulary size
    :param min_frequency: Minimum document for a word to be included
        in vocabulary.
    :param start_end_tokens: bool, should start of document and end of 
        document tokens be inserted?
    :param tokenize: Function that takes a string as an argument and 
        returns an iterable of tokens.
    :param offset: offset a vector by offset tokens.
    """
    def __init__(self, max_words=None, min_frequency=None, 
                 start_end_tokens=False, tokenize=None, 
                 offset=0):
        if tokenize is None:
            self.tokenize = lambda x: x.lower().split()
        else:
            self.tokenize = tokenize
        self.vocabulary = None
        self.vocabulary_size = 0
        self.word2idx = dict()
        self.idx2word = dict()
        self.max_words = max_words
        self.min_frequency = min_frequency
        self.start_end_tokens = start_end_tokens
        
    def fit(self, documents):
        corpus = [self.tokenize(doc) for doc in documents]
        self._build_vocabulary(corpus)
        self._build_word_index()
        return self
    
    def transform(self, documents, offset=0):           
        out = [] 
        for document in documents:
            words = self.tokenize(document)
            vector = [self.word2idx.get(word, self.word2idx['<UNK>']) 
                      for word in words]
            if self.start_end_tokens:
                vector = self.add_start_end(vector)
            out.append(vector[offset:])         
        return out  
                
    def _build_vocabulary(self, corpus):
        vocabulary = Counter(word for document in corpus for word in document)
        if self.max_words:
            vocabulary = {word: freq for word,
                          freq in vocabulary.most_common(self.max_words)}
        if self.min_frequency:
            vocabulary = {word: freq for word, freq in vocabulary.items()
                          if freq >= self.min_frequency}
        self.vocabulary = vocabulary
        self.vocabulary_size = len(vocabulary) + 2  # padding and unk tokens
        if self.start_end_tokens:
            self.vocabulary_size += 2

    def _build_word_index(self):
        self.word2idx['<PAD>'] = 0
        self.word2idx['<UNK>'] = 1

        if self.start_end_tokens:
            self.word2idx['<START>'] = 2
            self.word2idx['<END>'] = 3

        offset = len(self.word2idx)
        for idx, word in enumerate(self.vocabulary):
            self.word2idx[word] = idx + offset
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def add_start_end(self, vector):
        vector.append(self.word2idx['<END>'])
        return [self.word2idx['<START>']] + vector
    
class TextDataset(Dataset):
    def __init__(self, data, vectorizer, text_col='text', label_col=None):
        '''
        Extends PyTorch Dataset.
        
        :param path: The path to the input csv. There must be a ``text_col`` column.
        :param text_col: The name of the column containing text.
        :param label_col: The name of the column containing the labels.
        :param vectorizer: A class that contains ``fit``, ``transform`` and
            ```fit_transform`` functions
        '''
        if isinstance(data, str):
            data = pd.read_csv(data)
        self.vectorizer = vectorizer
        if not self.vectorizer.vocabulary:
            self.vectorizer.fit(data[text_col])
        self.texts = self.vectorizer.transform(data[text_col])
        if label_col is None:
            self.labels = [1 for x in range(data.shape[0])]
        else:
            self.labels = [x for x in data[label_col]]

    def __getitem__(self, index):
        return {'text': self.texts[index], 
                'label': self.labels[index]}

    def __len__(self):
        return len(self.labels)
    
class LMDataLoader:
    '''Data Loader for language model training
    
    This dataloader returns the data in batches with randomized sequence 
    length, according to the algorithm described by Jeremy Howard.
    
    Arguments
    ----------
    param: dataset: torch.utils.data.Dataset, returning  `{'text': single vectorized 
        doc on `__getitem__`, 'label': integer label}
    param: batch_size: int, batch size of the language model
    param: target_seq_len: int, the expected value (ish) for the randomized 
        sequence length
    param: shuffle: bool, Should the data be shuffled (doc-wise)
    param: max_seq_len: int, maximum length that randomized sequence length
        can't surpass.
    param: min_seq_len: int, smallest possible sequence length
    param: p_half_seq_len float(0,1): probability with which expected value
        of sequence length is halfed
        
    Methods
    ----------
    __iter__: yields batches of documents as X and shifted by one token 
        as y. Both of shape `batch_size x random_seq_len`.
      
    Details
    ----------
    See https://github.com/SMAPPNYU/smapp_stance_sentiment/wiki/ULMFiT-Data-Tricks
    for detailed description of the (somewhat confusing) process.
    '''
    def __init__(self, dataset, batch_size, target_seq_len, 
                 shuffle, max_seq_len, min_seq_len, p_half_seq_len):
        self.dataset = dataset
        self.batch_size = batch_size
        self.target_seq_len = target_seq_len
        self.shuffle = shuffle,
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.p_half_seq_len = p_half_seq_len
    
    def __len__(self):
        return len(self.dataset)
    
    @property
    def _iter_dataset(self):
        '''
        Returns the items of thedataset either in original or shuffled order
        '''
        if self.shuffle:
            order = np.random.permutation(len(self.dataset))
            return [self.dataset[i]['text'] for i in order]
        else:
            return [self.dataset[i]['text'] for i in range(len(self.dataset))]
        
    def _batchify(self, data):
        '''
        Creates a 'batchified' dataset that can be iterated
        throuth with varying sequence lengths. 
        '''
        iter_all = np.concatenate(data)
        chunk_size = len(iter_all) // self.batch_size
        iter_batched = np.array(
            iter_all[:chunk_size*self.batch_size]
        ).reshape(self.batch_size, -1)
        iter_batched = np.transpose(iter_batched)
        return torch.LongTensor(iter_batched)
    
    def _random_seq_len(self, start_batch_idx, data_set_len):
        # The first batch should have maximum length for GPU memory reasons
        if start_batch_idx == 0:
            seq_len = self.max_seq_len
        # All later batches have a sequence lenght that is randomized by 
        # drawing from truncated normal distribution with mean of
        # target_seq_len and truncation points (min_seq_len, max_seq_len)
        else:
            # In 5% of cases we half the expected sequence length
            if np.random.random() <= self.p_half_seq_len:
                target_seq_len = self.target_seq_len / 2
            else:
                target_seq_len = self.target_seq_len
                
            seq_len = int(truncnorm.rvs(a=self.min_seq_len, 
                                        b=self.max_seq_len, 
                                        loc=target_seq_len, 
                                        scale=1))

        ## make sure it's not larger than how much is left in the data        
        seq_len = min(seq_len, data_set_len - start_batch_idx)
        return seq_len    
            
        
    def __iter__(self): 
        '''
        In each iteration the data is re-shuffled, batchified and 
        yielded in batches
        '''
        # Get the (shuffled) dataset for this iteration
        iter_dataset = self._iter_dataset
        
        # Batchify the documents as described in the fastai lectures
        iter_batched = self._batchify(iter_dataset)
        ds_len = iter_batched.shape[0]

        # Iterate through the 'batchified' data with randomized sequence lengths
        start_batch_idx = 0
        
        while start_batch_idx < iter_batched.shape[0]:
            seq_len = self._random_seq_len(start_batch_idx, ds_len)
            x = iter_batched[start_batch_idx:start_batch_idx+seq_len]
            y = iter_batched[start_batch_idx+1:start_batch_idx+seq_len+1]
            start_batch_idx += seq_len
            # Don't return the last batch if the target has less values 
            # than the input (which would break the model or requre padding)
            if x.shape[0] > y.shape[0]:
                break
            # TODO: the double transpose can probably be solved more elegantly
            yield torch.transpose(x, 0, 1), torch.transpose(y, 0, 1)
            
class CLFDataLoader(DataLoader):
    '''Dataloader for text classification data
    
    Extends the standard torch.utils.data.DataLoader by adding,
    batch-wise (front-)padding
    
    Arguments
    ----------
    param: dataset: torch.utils.data.Dataset
    '''
    
    def __init__(self, dataset, batch_size, padding_idx=0, 
                 sampler=None, shuffle=True):
        super().__init__(dataset=dataset, collate_fn=self._pad_collate, 
                         sampler=sampler, shuffle=shuffle, batch_size=batch_size)
        self.padding_idx = padding_idx
        
    def _pad_collate(self, samples):
        max_len = max(len(s['text']) for s in samples)
        res = torch.zeros(max_len, len(samples)).long() + self.padding_idx
        for i,s in enumerate(samples):
            res[-len(s['text']):, i] = torch.LongTensor(s['text'])
        # TODO: Here again, these shouldn't have to be transposed in the first place
        return torch.transpose(res, 0, 1), torch.tensor([s['label'] for s in samples])