import os
import sys
import datetime
import pandas as pd
import torch
import torch.nn as nn
import copy
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
    
    def __init__(self, data_dir, train_file, valid_file, 
            max_vocab_size = 20000, batch_size = 50, revectorize = False):
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
        self.device = torch.device(f"cuda:{device_num}" if self.use_gpu 
            else "cpu")

        # IO setup
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        model_cache_dir = os.path.join(data_dir, 'models')
        self.data_cache = os.path.join(model_cache_dir, 'data_cache.pkl')
        self.vectorizer_cache = os.path.join(model_cache_dir, 
            'lm_vectorizer.pkl')
        os.makedirs(model_cache_dir, exist_ok=True)
        self.model_file_lm = os.path.join(model_cache_dir, f'LM__{today}.json')
        
        self.train_file = train_file
        self.valid_file = valid_file
        
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
            
            pickle.dump([self.train_ds, self.valid_ds], open(self.data_cache, 
                'wb'))
            pickle.dump(self.vectorizer, open(self.vectorizer_cache, 'wb'))
        else:
            self.train_ds, self.valid_ds = pickle.load(open(self.data_cache, 
                'rb'))
            self.vectorizer = pickle.load(open(self.vectorizer_cache, 'rb'))
        
        print("Vectorizing is complete.")        
        print(f'Train size: {len(self.train_ds)}\n \
            valid size:{len(self.valid_ds)}')
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
    
    def freezeTo(self, n):
        if n>1 and n<=1+self.lstm_layers:
            n = 1 + (n-1)*4
        print("Freezing layers until ", n)
        i=0
        for name, p in self.lm.named_parameters():
            if i<n or n==-1:
                p.requires_grad = False
            else:
                p.requires_grad = True
            i+=1
        for name, p in self.lm.named_parameters():
            print(name, p.requires_grad)
    
    def fit_language_model(self, pretrained_itos = None,
            pretrained_weight_file = None, lm_hidden_dim = 1150, 
            lm_embedding_dim = 400, lm_lstm_layers = 3, num_epochs=100, 
            display_epoch_freq = 1, scheduler = 'ulmfit',
            max_lrs = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3]):
        
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
        
        if pretrained_weight_file is not None and pretrained_itos is not None: 
            print("Starting Loading pretrained Wikitext model")
            enc = torch.load(pretrained_weight_file, 
                map_location=lambda storage, loc: storage)
            self.embedding_dim = enc['0.encoder.weight'].shape[1]
            self.hidden_dim = int(
                enc['0.rnns.0.module.weight_hh_l0_raw'].shape[0]/4)
            self.lstm_layers = 3
            
            new_enc = {}
            for k,v in enc.items():
                layer_detail = k.split('.')
                layer_name = layer_detail[-1].replace('_raw', '')
                if len(layer_detail) == self.lstm_layers: 
                    new_enc[f'{layer_detail[1]}.{layer_name}'] = v
                else:
                    new_enc[f'{layer_detail[1]}.{layer_detail[2]}.{layer_name}'
                        ] = v
            
            del new_enc['encoder_with_dropout.embed.weight']
            
            pretrained_idx2word = pickle.load(open(pretrained_itos, 'rb'))
            pretrained_word2idx =\
                {k: i for i,k in enumerate(pretrained_idx2word)}
            
            new_model_vectorizer = self.vectorizer
            pretrained_encoder_weights = enc['0.encoder.weight']
            
            row_m = pretrained_encoder_weights.mean(dim=0)
            row_m = [x.item() for x in row_m]
            
            new_vocab_size = len(new_model_vectorizer.word2idx)
            new_encoder_weights = torch.tensor(
                [row_m for i in range(new_vocab_size)])
            
            new_idx2weights = {}
            for word, i in new_model_vectorizer.word2idx.items():
                if word in pretrained_word2idx:
                    word_idx = pretrained_word2idx[word]
                    new_encoder_weights[i] =\
                        pretrained_encoder_weights[word_idx]

            new_enc['encoder.weight'] = new_encoder_weights
            new_enc['decoder.weight'] = copy.copy(new_encoder_weights)
            new_enc['decoder.bias'] =\
                torch.zeros(new_enc['decoder.weight'].shape[0])

            self.lm = RNNLM(device=self.device, vocab_size=new_vocab_size, 
                embedding_size=self.embedding_dim, hidden_size=self.hidden_dim, 
                batch_size=50, num_layers=3, tie_weights=True, 
                word2idx = new_model_vectorizer.word2idx)
            print("Initialised loading with pretrained Wikitext model")
            
        else:
            # Build and initialize the model
            print("No Pretrained Model specified")
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
        
        param_list = [{'params': rnn.parameters(), 'lr': 1e-3} 
            for rnn in self.lm.rnns]

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
        
        self.optimizer = torch.optim.Adam(param_list)
        if scheduler == 'ulmfit':
            self.scheduler = CyclicLR(self.optimizer,  max_lrs=max_lrs, 
                     mode='ulmfit', ratio=1.5, cut_frac=0.4, 
                     train_data_loader = self.train_dl, 
                     verbose=False)
        print("Beginning LM Fine Tuning")
        self.freezeTo(3)
        history = training_loop(batch_size=self.batch_size, 
                                num_epochs=1,
                                display_freq=self.display_epoch_freq, 
                                model=self.lm, 
                                criterion=self.loss,
                                optim=self.optimizer,
                                scheduler=self.scheduler,
                                device=self.device,
                                training_set=self.train_dl,
                                validation_set=self.valid_dl,
                                best_model_path=self.model_file_lm,
                                history=None)

        self.freezeTo(2)
        history = training_loop(batch_size=self.batch_size, 
                                num_epochs=1,
                                display_freq=self.display_epoch_freq, 
                                model=self.lm, 
                                criterion=self.loss,
                                optim=self.optimizer,
                                scheduler=self.scheduler,
                                device=self.device,
                                training_set=self.train_dl,
                                validation_set=self.valid_dl,
                                best_model_path=self.model_file_lm,
                                history = history)
        
        self.freezeTo(0)
        history = training_loop(batch_size=self.batch_size, 
                                num_epochs=self.num_epochs-2,
                                display_freq=self.display_epoch_freq, 
                                model=self.lm, 
                                criterion=self.loss,
                                optim=self.optimizer,
                                scheduler=self.scheduler,
                                device=self.device,
                                training_set=self.train_dl,
                                validation_set=self.valid_dl,
                                best_model_path=self.model_file_lm,
	                        history = history)
	
if __name__ == '__main__':
    
    test = SMaPPLearn(data_dir = '../data/imdb/', 
        train_file = '../data/imdb/unsup.csv', 
        valid_file = '../data/imdb/valid.csv',  
        max_vocab_size = 20000, revectorize = False)
    """
    test.fit_language_model(lm_embedding_dim = 200, lm_hidden_dim = 250, 
        num_epochs = 10, display_epoch_freq = 1)
    """ 
    test.fit_language_model(
        pretrained_weight_file = 
            '../data/imdb/weights_pretrained/fwd_wt103.h5', 
        pretrained_itos = 
            '../data/imdb/weights_pretrained/fitos_wt103.pkl',
        display_epoch_freq = 1, num_epochs = 15, scheduler = 'ulmfit', 
        max_lrs = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    #"""
