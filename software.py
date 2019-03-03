import os
import sys
import datetime
import pandas as pd
import torch
import torch.nn as nn
sys.path.append('../')
from lr_scheduler import CyclicLR
from training_utils import training_loop, test_loop
from model import RNNLM
from data_utils import (IndexVectorizer, 
                        TextDataset, 
                        SpacyTokenizer,
                        LMDataLoader)
import pickle

class SMaPPLearn:
    
    def __init__(self, data_dir, max_vocab_size = 20000, batch_size = 50, 
            revectorize = False):
        
        self.data_dir = data_dir
        self.TOKENIZE = SpacyTokenizer().tokenize
        self.MIN_WORD_FREQ = 2
        self.MAX_VOCAB_SIZE = max_vocab_size
        self.STAT_END_TOK = True

        ## Training Language Model
        self.batch_size = batch_size
        self.target_seq_len = 65
        self.max_seq_len = 75
        self.min_seq_len = 5
        
        # GPU setup
        self.use_gpu = torch.cuda.is_available()
        device_num = 0
        self.device = torch.device(f"cuda:{device_num}" if self.use_gpu else "cpu")

        # IO setup
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        model_cache_dir = os.path.join(data_dir, 'models')
        self.data_cache = os.path.join(model_cache_dir, 'data_cache.pkl')
        self.vectorizer_cache = os.path.join(model_cache_dir, 'lm_vectorizer.pkl')
        os.makedirs(model_cache_dir, exist_ok=True)
        self.model_file_lm = os.path.join(model_cache_dir, f'LM__{today}.json')
        
        self.train_file = os.path.join(data_dir, 'unsup.csv')
        self.valid_file = os.path.join(data_dir, 'valid.csv')
        
        self.revectorize = revectorize
        if self.revectorize or not os.path.isfile(self.data_cache):
            print("Vectorizing starting...")
            train = pd.read_csv(self.train_file)
            valid = pd.read_csv(self.valid_file)
            self.vectorizer = IndexVectorizer(max_words = self.MAX_VOCAB_SIZE, 
                                     min_frequency=self.MIN_WORD_FREQ,
                                     start_end_tokens=self.STAT_END_TOK, 
                                     tokenize=self.TOKENIZE)
            
            self.train_ds = TextDataset(data=train, vectorizer=self.vectorizer, 
                text_col='text')
            self.valid_ds = TextDataset(data=valid, vectorizer=self.vectorizer, 
                text_col='text')
            
            pickle.dump([self.train_ds, self.valid_ds], open(self.data_cache, 'wb'))
            pickle.dump(self.vectorizer, open(self.vectorizer_cache, 'wb'))
        else:
            self.train_ds, self.valid_ds = pickle.load(open(self.data_cache, 'rb'))
            self.vectorizer = pickle.load(open(self.vectorizer_cache, 'rb'))
        
        print("Vectorizing is complete.")        
        print(f'Train size: {len(self.train_ds)}\nvalid size: {len(self.valid_ds)}')
        print(f"Vocab size: {len(self.vectorizer.vocabulary)}")
        
        self.train_dl = LMDataLoader(dataset=self.train_ds, 
                        target_seq_len=self.target_seq_len, 
                        shuffle=True, 
                        max_seq_len=self.max_seq_len, 
                        min_seq_len=self.min_seq_len, 
                        p_half_seq_len=0.05,
                        batch_size=self.batch_size)
                        
        self.valid_dl = LMDataLoader(dataset=self.valid_ds,
                                target_seq_len=self.target_seq_len, 
                                shuffle=True, 
                                max_seq_len=self.max_seq_len, 
                                min_seq_len=self.min_seq_len, 
                                p_half_seq_len=0.05,
                                batch_size=self.batch_size) 
        
        print("Created Data Loaders for documents")
        
    def fit_language_model(self, lm_hidden_dim = 1150, lm_embedding_dim = 400, 
            lm_lstm_layers = 3, num_epochs=100, display_epoch_freq = 10):
        
        ## Model Architecture
        self.hidden_dim = lm_hidden_dim
        self.embedding_dim = lm_embedding_dim
        self.dropout = 0.3
        self.lstm_layers = lm_lstm_layers
        self.lstm_bidirection = False
        self.lstm_tie_weights = True
        self.num_epochs = num_epochs
        self.display_epoch_freq = display_epoch_freq
        
        if self.use_gpu: torch.cuda.manual_seed(303)
        else: torch.manual_seed(303)

        # set up Files to save stuff in
        runtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            
        # Build and initialize the model
        self.lm = RNNLM(self.device, self.vectorizer.vocabulary_size, 
                    self.embedding_dim, self.hidden_dim, 
                    self.batch_size, dropout = self.dropout, 
                    tie_weights = self.lstm_tie_weights, 
                    num_layers = self.lstm_layers, 
                    bidirectional = self.lstm_bidirection, 
                    word2idx = self.vectorizer.word2idx,
                    log_softmax = False)

        if self.use_gpu:
            self.lm = self.lm.to(self.device)
        
        # Loss and Optimizer
        self.loss = nn.CrossEntropyLoss()

        # Extract pointers to the parameters of the lstms
        param_list = [{'params': rnn.parameters(), 'lr': 1e-3} for rnn in self.lm.rnns]

        # If weights are tied between encoder and decoder, we can only optimize 
        # parameters in one of those two layers
        if not self.lstm_tie_weights:
            param_list.extend([
                    {'params': self.lm.encoder.parameters(), 'lr':1e-3},
                    {'params': self.lm.decoder.parameters(), 'lr':1e-3},
                ])
        else:
            param_list.extend([
                {'params': self.lm.decoder.parameters(), 'lr':1e-3},
            ])

        self.optimizer = torch.optim.Adam(param_list, lr=0.01)
        print("Beginning LM Fine Tuning")
        history = training_loop(batch_size=self.batch_size, 
                                num_epochs=self.num_epochs,
                                display_freq=self.display_epoch_freq, 
                                model=self.lm, 
                                criterion=self.loss,
                                optim=self.optimizer,
                                scheduler=None,
                                device=self.device,
                                training_set=self.train_dl,
                                validation_set=self.valid_dl,
                                best_model_path=self.model_file_lm,
                                history=None)

if __name__ == '__main__':
    test = SMaPPLearn(data_dir = '/home/vishakh/Stuff/SMaPP/', max_vocab_size = 10000, revectorize = False)
    test.fit_language_model(lm_hidden_dim = 50, lm_embedding_dim=50, display_epoch_freq = 1, num_epochs = 1)
