import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNLM(nn.Module):
    def __init__(self, device, vocab_size, embedding_size, 
                 hidden_size, batch_size, 
                 dropout=.5, num_layers=1, tie_weights=False, 
                 bidirectional=False, word2idx={}, log_softmax=False):
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
        self.dropout = nn.Dropout(p = dropout)
        self.log_softmax = nn.LogSoftmax(dim = 1) if log_softmax else None
        
        # Model Layers
        self.encoder = nn.Embedding(vocab_size, embedding_size, 
                                    padding_idx = word2idx.get('<PAD>', 1))
        
        self.lstm1 = nn.LSTM(embedding_size, hidden_size, 
                             num_layers = 1, 
                             bidirectional = bidirectional,
                             batch_first = True)
        
        self.lstm2 = nn.LSTM(hidden_size * self.num_directions, hidden_size, 
                             num_layers = 1, 
                             bidirectional = bidirectional,
                             batch_first = True)
        
        self.lstm3 = nn.LSTM(hidden_size * self.num_directions, hidden_size, 
                             num_layers = 1, 
                             bidirectional = bidirectional,
                             batch_first = True)
        
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
    
    
    def sample(self, seed='This ', length=20):
        '''
        Generates a sequence of text given a starting word ix and hidden state.
        '''
        with torch.no_grad():
            indices = [self.word2idx.get(w, 1) for w in seed.lower().split()]
            hidden = self.init_hidden(1)
            for i in range(length):
                # create inputs
                x_input = torch.LongTensor(indices).to(self.device)
                x_input = x_input.unsqueeze(0)
                
                output, hidden = self(x_input, hidden)
                last_state = output[-1, :]
                predicted_probabilities = torch.softmax(last_state, dim=0)

                idx = torch.multinomial(predicted_probabilities, 1).item()
                indices.append(idx)

            return indices
    
    
    def forward(self, x, hidden):
        '''
        Iterates through the input, encodes it.
        Each embedding is sent through the step function.
        Dropout the last hidden layer and decode to a logit
        x.size() #(bsz, seq_len)
        
        logit.size # (bsz, seq_len, vocab_size)
        equivalent to (output.size(0), output.size(1), logit.size(1)
        '''
        seq_len = x.size(1)
        x_emb = self.encoder(x)
        
        output, hidden = self.lstm1(x_emb, hidden)
        output, hidden = self.lstm2(self.dropout(output), hidden)
        output, hidden = self.lstm3(self.dropout(output), hidden)
        
        logit = self.decoder(self.dropout(output))
        if self.log_softmax:
            logit = self.log_softmax(logit)       
        logit = logit.view(logit.size(0) * seq_len, self.vocab_size)
        
        return logit, hidden
