import torch
import torch.nn as nn
import torch.nn.functional as F

class LockedDropout(nn.Module):
    def __init__(self, p=.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.:
            return x
        m = x.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p).div_(1 - self.p)
        return m * x
    
class RNNLM(nn.Module):
    def __init__(self, device, vocab_size, embedding_size, 
                 hidden_size, batch_size, 
                 dropouti=.5,
                 dropouth=.5,
                 dropout=.5, num_layers=1, tie_weights=False, 
                 bidirectional=False, word2idx={}, log_softmax=False,):
        super(RNNLM, self).__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.tie_weights = tie_weights
        self.num_layers = num_layers
        self.num_directions = 1 if not bidirectional else 2
        self.word2idx = word2idx
        self.idx2word = {v:k for k,v in word2idx.items()}
        
        # Model Pieces
        self.dropout = LockedDropout(dropout)
        #self.embedding_dropout = LockedDropout(dropout3)
        self.input_dropout = LockedDropout(dropouti)
        self.hidden_dropout = LockedDropout(dropouth)
        self.log_softmax = nn.LogSoftmax(dim = 1) if log_softmax else None
        
        # Model Layers
        self.encoder = nn.Embedding(vocab_size, embedding_size, 
                                    padding_idx = word2idx.get('<PAD>', 1))
        
        self.rnns = [nn.LSTM(embedding_size, hidden_size, 
                             num_layers = 1, 
                             bidirectional = bidirectional,
                             batch_first = True) for l in range(num_layers)]
        self.rnns = nn.ModuleList(self.rnns)
        
        self.hidden_dropout = nn.ModuleList([LockedDropout(dropouth) for l in range(num_layers)])
        
        self.decoder = nn.Linear(hidden_size * self.num_directions, vocab_size)

        # tie enc/dec weights
        if self.tie_weights:
            if hidden_size != embedding_size:
                raise ValueError('When using the `tied` flag, hidden_size'
                                 'must be equal to embedding_dim')
            self.decoder.weight = self.encoder.weight
            
        self.init_weights()

        
    def init_hidden(self, bsz=None):
        '''
        For the nn.LSTM.
        Defaults to the batchsize stored in the class params, but can take in an argument
        in the case of sampling.
        '''
        if bsz == None: 
            bsz = self.batch_size
        h0 = torch.zeros(self.num_layers * self.num_directions, 
                         bsz, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, 
                         bsz, self.hidden_size ).to(self.device)
        return (h0, c0)
    
    
    def init_weights(self):
        initrange = 0.1
        em_layer = [self.encoder]
        lin_layers = [self.decoder]
        for layer in lin_layers + em_layer:
            layer.weight.data.uniform_(-initrange, initrange)
            if layer in lin_layers:
                layer.bias.data.fill_(0)
    
    
    def sample(self, x_start, length):
        '''
        Generates a sequence of text given a starting word ix and hidden state.
        '''
        with torch.no_grad():
            indices = [x_start]
            for i in range(length):
                # create inputs
                x_input = torch.LongTensor(indices).to(self.device)
                x_embs = self.encoder(x_input.view(1, -1))

                # send input through the rnn
                output, hidden = self.lstm1(x_embs)
                output, hidden = self.lstm2(output, hidden)

                # format the last word of the rnn so we can softmax it.
                one_dim_last_word = output.squeeze()[-1] if i > 0 else output.squeeze()
                fwd = one_dim_last_word[ : self.hidden_size ]
                bck = one_dim_last_word[ self.hidden_size : ]

                # pick a word from the disto
                word_weights = torch.softmax(fwd, dim=0)
                word_idx = torch.multinomial(word_weights, num_samples=1).squeeze().item()
                indices.append(word_idx)

        return indices
    
    
    def forward(self, x, hidden, log_softmax=False):
        '''
        Iterates through the input, encodes it.
        Each embedding is sent through the step function.
        Dropout the last hidden layer and decode to a logit
        x.size() #(bsz, seq_len)
        
        logit.size # (bsz, seq_len, vocab_size)
        equivalent to (output.size(0), output.size(1), logit.size(1)
        '''
        seq_len = x.size(1)
        x_emb = self.input_dropout(self.encoder(x))
        
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, (rnn, drop) in enumerate(zip(self.rnns, self.hidden_dropout)):
            output, hidden = rnn(x_emb, hidden)
            new_hidden.append(hidden)
            raw_outputs.append(output)
            
            if l != self.n_layers - 1: 
                raw_output = drop(output)
                outputs.append(raw_output)

        logit = self.decoder(self.dropout(output))
        if self.log_softmax:
            logit = self.log_softmax(logit)       
        logit = logit.view(logit.size(0) * seq_len, self.vocab_size)
        
        return logit, hidden
