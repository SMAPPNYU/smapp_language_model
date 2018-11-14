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
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split

class IndexVectorizer:
    """
    Transforms a Corpus into lists of word indices.
    """
    def __init__(self, max_words=None, min_frequency=None, 
                 start_end_tokens=False, maxlen=None):
        self.vocabulary = None
        self.vocabulary_size = 0
        self.word2idx = dict()
        self.idx2word = dict()
        self.max_words = max_words
        self.min_frequency = min_frequency
        self.start_end_tokens = start_end_tokens
        self.maxlen = maxlen

    def _find_max_document_length(self, corpus):
        self.maxlen = max(len(document) for document in corpus)
        if self.start_end_tokens:
            self.maxlen += 2

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

    def fit(self, corpus):
        if not self.maxlen:
            self._find_max_document_length(corpus)
        self._build_vocabulary(corpus)
        self._build_word_index()

    def pad_document_vector(self, vector):
        padding = self.maxlen - len(vector)
        vector.extend([self.word2idx['<PAD>']] * padding)
        return vector

    def add_start_end(self, vector):
        vector.append(self.word2idx['<END>'])
        return [self.word2idx['<START>']] + vector

    def transform_document(self, document, offset=0):
        """
        Vectorize a single document
        """
        vector = [self.word2idx.get(word, self.word2idx['<UNK>']) 
                  for word in document]
        if len(vector) > self.maxlen:
            vector = vector[:self.maxlen]
        if self.start_end_tokens:
            vector = self.add_start_end(vector)
        vector = vector[offset:self.maxlen]
        
        return self.pad_document_vector(vector)

    def transform(self, corpus):
        """
        Vectorizes a corpus in the form of a list of lists.
        A corpus is a list of documents and a document is a list of words.
        """
        return [self.transform_document(document) for document in corpus]
    
    
class TextDataset(Dataset):
    def __init__(self, data, vectorizer,  tokenizer=None, text_col='text', stopwords=None):
        '''
        Extends PyTorch Dataset.
        
        :param path: The path to the input csv. There must be a ``text_col`` column.
        :param text_col: The name of the column containing text.
        :param vectorizer: A class that contains a ``transform_document`` function, 
            which converts word2idx. Needs an ``offset`` param for language model.
        :param tokenizer: Pass a tokenizer function that takes a string and returns
            a list of tokens. Defaults to split and lower.
        :stopwords: A list of set of stopwords to ignore during tokenization.
        '''
        if isinstance(data, str):
            self.corpus = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.corpus = data
        else:
            raise "data must be a filepath to a csv or a Pandas dataframe"
        self.text_col = text_col
        self.tokenizer = tokenizer
        self.vectorizer = vectorizer
        self.stopwords = stopwords
        self._tokenize_corpus()
        if self.stopwords: self._remove_stopwords() 
        self._vectorize_corpus()

    def _remove_stopwords(self):
        stopfilter = lambda doc: [word for word in doc if word not in self.stopwords]
        self.corpus['tokens'] = self.corpus['tokens'].apply(stopfilter)

    def _tokenize_corpus(self):
        if self.tokenizer:
            self.corpus['tokens'] = self.corpus[self.text_col].apply(self.tokenizer)
        else:
            self.corpus['tokens'] = self.corpus[self.text_col].apply(lambda x: x.lower().split())

    def _vectorize_corpus(self):
        '''
        Vectorizes the input (X) and the target (y), which is just the input offset by 1.
        Ex -> "the boy eats too much" becomes "boy eats too much"
        '''
        if not self.vectorizer.vocabulary:
            self.vectorizer.fit(self.corpus['tokens'])
        self.corpus['vectors'] = self.corpus['tokens'].apply(self.vectorizer.transform_document)
        self.corpus['target'] = self.corpus['tokens'].apply(self.vectorizer.transform_document,
                                                            offset=1)

    def __getitem__(self, index):
        sentence = self.corpus['vectors'].iloc[index]
        target = self.corpus['target'].iloc[index]
        return torch.LongTensor(sentence), torch.LongTensor(target)

    def __len__(self):
        return len(self.corpus)
    
def simple_tokenizer(text):
    '''
    An example of a tokenizer to pass to TextDataset's ``tokenizer`` param
    '''
    return text.lower().split()