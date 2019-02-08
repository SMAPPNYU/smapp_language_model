import torch
import torch.nn as nn
import torch.nn.functional as F

class LockedDropout(nn.Module):
    '''
    Handles dropout for an entire batch.
    '''
    def __init__(self, p=.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        '''
        During training, will create a mask to dropout and scale
        each row uniformly. Assumes x is 3-dimensional.
        '''
        if not self.training or self.p == 0.:
            return x
        
        mask = x.new(1, x.size(1), x.size(2)) \
                .bernoulli_(1 - self.p) \
                .div_(1 - self.p)
        
        return mask * x
    
def _detach(X, cpu=True):
    '''
    Detaches a torch tensor from the graph (and optionally moved to CPU).
    If the X is a tuple, such as the hidden states returned from LSTMs,
    then each hidden state is detached recursively.
    '''
    if isinstance(X, tuple) or isinstance(X, list):
        return [_detach(_, cpu) for _ in X]
    if not isinstance(X, torch.Tensor):
        return X
    X = X.detach()
    return X.cpu() if cpu else X


def embedded_dropout(embed, inputs, p=0.1, scale=None):
    '''
    This creates a dropout mask to operate on each row of an embedding.
    What that means is for each batch the same embedding indices (or words)
    are dropped out.
    
    :param embed: is a torch embedding
    :param inputs: is a LongTensor to feed into the embedding
    :param p: is the dropout rate.
    '''
    if p:
        mask = embed.weight.data.new() \
                    .resize_([embed.weight.size(0), 1]) \
                    .bernoulli_(1 - p) \
                    .expand_as(embed.weight) \
                    .div_(1 - p)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) \
                            * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = F.embedding(inputs, masked_embed_weight,
        padding_idx, embed.max_norm, embed.norm_type,
        embed.scale_grad_by_freq, embed.sparse
    )
    
    return X  
    
    
class RNNLM(nn.Module):
    def __init__(self, device, vocab_size, embedding_size, 
                 hidden_size, batch_size, dropout_i=.5, 
                 dropout_h=.5, dropout=.5, dropout_e=.1,
                 num_layers=3, tie_weights=False, 
                 bidirectional=False, word2idx={}, 
                 log_softmax=False):
        '''
        TODO: explain params
        '''
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
        self.p_embedding_dropout = dropout_e
        self.input_dropout = LockedDropout(dropout_i)
        self.hidden_dropout = LockedDropout(dropout_h)
        self.log_softmax = nn.LogSoftmax(dim = 1) if log_softmax else None
        
        # Model Layers
        self.encoder = nn.Embedding(vocab_size, embedding_size, 
                                    padding_idx = word2idx.get('<PAD>', 1))
        self.rnns = [nn.LSTM(embedding_size if l == 0 else hidden_size, # see footnote1
                             hidden_size, 
                             num_layers = 1, 
                             bidirectional = bidirectional,
                             batch_first = True) for l in range(num_layers)]
        self.rnns = nn.ModuleList(self.rnns)
        self.hidden_dropout = nn.ModuleList([LockedDropout(dropout_h) for l in range(num_layers)])
        self.decoder = nn.Linear(hidden_size * self.num_directions, vocab_size)
        
        # initialize the encoder and decoder weights, initialize the hidden state.
        self.init_hidden()
        self._init_weights()
        
        # tie enc/dec weights
        if self.tie_weights:
            self._tie_weights()
            
    def _tie_weights():
        '''
        Tie the encoder and decoder's weights
        '''
        if self.hidden_size != self.embedding_size:
                raise ValueError('When using the `tied` flag, hidden_size'
                                 'must be equal to embedding_dim')
        self.decoder.weight = self.encoder.weight
    
    
    def _reset_hidden_layer(self, bsz=None):
        '''
        Resets (or initalizes) the initial hidden (h0) and output (c0) for an LSTM.
        Returns a tuple of tensors!
        '''
        if bsz == None: 
            bsz = self.batch_size
        h0 = torch.zeros(self.num_directions, bsz, 
                         self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_directions, bsz, 
                         self.hidden_size).to(self.device)
        return (h0, c0)
    
    
    def init_hidden(self, bsz=None):
        '''
        Initalizes the hidden state for each layer of the RNN.
        Note that hidden states are stored in the class!
        The hidden state is a list (of length num_layers) of tuples.
        See `_reset_hidden_layer()` for the dimensions of the tuples of tensors.
        
        '''
        self.hidden = [self._reset_hidden_layer(bsz=bsz) for l in range(self.num_layers)]
    
    
    def _init_weights(self):
        '''
        Initializes the bias and weights for the encoder and decoder
        '''
        initrange = 0.1
        em_layer = [self.encoder]
        lin_layers = [self.decoder]
        for layer in lin_layers + em_layer:
            layer.weight.data.uniform_(-initrange, initrange)
            if layer in lin_layers:
                layer.bias.data.fill_(0)
    
    
    def sample(self, seed='This is bad', length=10, return_words=False):
        '''
        Generates a sequence of text given some start seed words
        '''
        with torch.no_grad():
            indices = [self.word2idx.get(w, 1) for w in seed.lower().split()]
            self.init_hidden(bsz=1)
            """
            for i in range(len(indices)): # see footnote2
                # create inputs
                x_input = torch.LongTensor(indices[:i+1]).to(self.device)
                x_input = x_input.unsqueeze(0)
                output = self(x_input)
            """
            
            for i in range(length):
                # create inputs
                x_input = torch.LongTensor(indices).to(self.device)
                x_input = x_input.unsqueeze(0)
                
                output = self(x_input)
                last_state = output[-1, :]
                predicted_probabilities = torch.softmax(last_state, dim=0)
                idx = torch.multinomial(predicted_probabilities, 1).item()
                indices.append(idx)
        
        if return_words:        
            words = [self.idx2word.get(i, 'UNK') for i in indices]
            return words
        
        return indices
    
    def forward(self, x, return_hidden=False, intermediaries=False):
        '''
        Iterates through the input (which are LongTensors of word indices),
        converts indicies to word embeddings, and 
        each embedding is sent through the forward function of each RNN layer.
        Dropout the last hidden layer and decode to a logit
        
        Hidden states for x are stored in `self.hidden`, if you want to manually set a hidden state,
        you can set it as model.hidden_state = my_hidden_states.
        
        Be sure the new hidden states are the same dimensions and on the same device.
        See `_init_hidden_state()` for more information about the dimensions.
        
        x.size() #(bsz, seq_len)
        logit.size # (bsz, seq_len, vocab_size)
        equivalent to (output.size(0), output.size(1), logit.size(1)
        '''
        bsz, seq_len = x.size()
        
        # drop out weights in the embedding
        x_emb = embedded_dropout(
            self.encoder, 
            inputs=x, 
            p=self.p_embedding_dropout if self.training else 0
        )
        
        # dropout words from the embedding matrix
        x_emb = self.input_dropout(x_emb)        
        
        # send the embedding matrix through each RNN layer.
        # remember: each rnn is an lstm...
        output = x_emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, (rnn, drop) in enumerate(zip(self.rnns,
                                            self.hidden_dropout)):
            output, hidden = rnn(output, self.hidden[l])
            new_hidden.append(hidden)
            raw_outputs.append(output)
            
            # dropout between inner RNN layers
            if l != self.num_layers - 1: 
                raw_output = drop(output)
                outputs.append(raw_output)
                
        # overwrite the old hidden states, and detach from from GPU memory. NOTE why are we detaching?
        self.hidden = [_detach(h, cpu=False) for h in new_hidden]
        
        # send the output of the last RNN layer through the decoder (linear layer)
        logit = self.decoder(self.dropout(raw_output))
        outputs.append(logit)
        if self.log_softmax:
            logit = self.log_softmax(logit)
        logit = logit.view(bsz * seq_len, self.vocab_size)

        if intermediaries:
            return logit, self.hidden, raw_outputs, outputs
        if return_hidden:
            return logit, self.hidden
        return logit

"""
# FOOTNOTES
footnote1: 
the first RNN layer recieves an embedding, 
subsequent layers get the output of the previous layer, 
which has a hidden_size number of features rather than embedding_size.
"""