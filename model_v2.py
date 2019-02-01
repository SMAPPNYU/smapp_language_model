import torch
import torch.nn as nn
import torch.nn.functional as F

class LockedDropout(nn.Module):
    '''
    Handles dropout for an entire batch
    '''
    def __init__(self, p=.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.:
            return x
        m = x.new(x.size(0), 1, x.size(2)) \
             .bernoulli_(1 - self.p) \
             .div_(1 - self.p)
        return m * x
    
def _detach(X):
    '''
    Detaches a torch tensor from the graph and switches to CPU.
    If the X is a tuple, such as the hidden states returned from LSTMs,
    Then each hidden state is detached recursively.
    '''
    if isinstance(X, tuple):
        return [_detach(_) for _ in X]
    if not isinstance(X, torch.Tensor):
        return X
    X = X.detach()
    return X.cpu()


def embedded_dropout(embed, inputs, p=0.1, scale=None):
    '''
    This droputs out weights in the embedding.
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
        self.rnns = [nn.LSTM(embedding_size, hidden_size, 
                             num_layers = 1, 
                             bidirectional = bidirectional,
                             batch_first = True) for l in range(num_layers)]
        self.rnns = nn.ModuleList(self.rnns)
        self.hidden_dropout = nn.ModuleList([LockedDropout(dropout_h) for l in range(num_layers)])
        self.decoder = nn.Linear(hidden_size * self.num_directions, vocab_size)
        
        # initialize the encoder and decoder weights, initialize the hidden state.
        self._init_weights()
        self._init_hidden()
        
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
        h0 = torch.zeros(self.num_directions, bsz, self.hidden_size)#.to(self.device)
        c0 = torch.zeros(self.num_directions, bsz, self.hidden_size )#.to(self.device)
        return (h0, c0)
    
    
    def _init_hidden(self, bsz=None):
        '''
        Initalizes each row of the RNNs
        '''
        self.hidden = [self._reset_hidden_layer(bsz=bsz) for l in range(self.num_layers)]
    
    
    def _init_weights(self):
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
    
    
    def forward(self, x, hidden, intermediaries=False):
        '''
        Iterates through the input (which are LongTensors of word indices),
        converts indicies to word embeddings, and 
        each embedding is sent through the forward function of each RNN layer.
        Dropout the last hidden layer and decode to a logit
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
            if l != self.n_layers - 1: 
                raw_output = drop(output)
                outputs.append(raw_output)
                
        # overwrite the old hidden states, and detach from from GPU memory.
        self.hidden = [_detach(h) for h in new_hidden]
        
        # send the output of the last RNN layer through the decoder (linear layer)
        logit = self.decoder(self.dropout(raw_output))
        outputs.append(logit)
        if self.log_softmax:
            logit = self.log_softmax(logit)
        logit = logit.view(bsz * seq_len, self.vocab_size)

        if intermediaries:
            return logit, self.hidden, raw_outputs, outputs
        
        return logit, self.hidden
