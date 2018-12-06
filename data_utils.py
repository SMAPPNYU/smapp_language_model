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

class LanguageModelDataset(Dataset):

    def __init__(self, dataframe, text_column='text', label_column='label', 
                 tokenizer=None):
        self.input_data = dataframe
        if tokenizer is None:
            self.tokenizer = lambda x: x.lower().split()
        else:
            self.tokenizer = tokenizer
        self.id2token = []
        self.token2id = {}
        self.data = []
    
        # Make vocab
        for doc in self.input_data[text_column]:
            tokens = self.tokenizer(doc)
            for t in tokens:
                if t not in self.token2id:
                    self.id2token.append(t)   
                    self.token2id[t] = len(self.id2token) - 1
        # Numericalize corpus
        for doc in self.input_data[text_column]:
            tokens = self.tokenizer(doc)
            self.data.append([self.token2id[t] for t in tokens])
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    
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


class LanguageModelDataLoader:
    '''Data Loader for language model training
    
    This dataloader returns the data in batches with randomized sequence 
    length, according to the algorithm described by Jeremy Howard.
    
    Arguments
    ----------
    param: dataset: torch.utils.data.Dataset, returning a single vectorized 
        doc on `__getitem__`
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
        as y of shape `random_seq_len x batch_size`.
      
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
            return [self.dataset[i] for i in order]
        else:
            return [self.dataset[i] for i in range(len(self.dataset))]
        
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
            yield x, y