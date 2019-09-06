# Deep Learning Language Model for Text Classification

This repository contains our implementation of a language model similar to [ulmfit](https://arxiv.org/abs/1801.06146). 

In a nutshell, a language model learns a probability distribution over a sequence of words in your corpus. A neural LM makes use of unidirectional or bidirectional RNN(s) to achieve the same. In the training phase, LM accepts a sequence of tokens (words, characters etc.) as input and learns to predict the next token in that sequence. 

The output state of the trained RNN after a forward pass on a sequence of tokens is a fixed length vector that encodes an arbitrary length document while preserving word order information. So this state can be consumed by a classifier as a feature vector would to learn specific supervised predictions. 

On how to perform classification follow this [Jupyter notebook](https://github.com/SMAPPNYU/smapp_language_model/blob/master/example_notebooks/EXAMPLE_Classification.ipynb). 

For more details on general text classification from a language model, follow through this [link](http://nlp.fast.ai/).

## Usage
### Object initialisation:
```
    from software import SMaPPLearn
    LM = SMaPPLearn(data_dir = '../data/imdb/', 
        train_file = '../data/imdb/unsup.csv', 
        valid_file = '../data/imdb/valid.csv',  
        max_vocab_size = 20000, revectorize = True)
```
Initialises the language model object, selects the vocabulary with upper limit of terms "max_vocab_size" from the training file and creates a vectorizer for the same which is persistently stored in the data directory. Training file and validation file are expected to have a column 'text' containing the text sequences. 

### Fitting
```
    LM.fit_language_model(lm_embedding_dim = 200, lm_hidden_dim = 250, 
         num_epochs = 10, display_epoch_freq = 1)
```
Training the LM from scratch. set the embedding size and size of hidden dimensions according to your system memory specs. 
Display_epoch_freq to sanity check the learned language model. 

### Transfer Learning
```
    LM.fit_language_model(
        pretrained_weight_file = 
            '../data/imdb/weights_pretrained/fwd_wt103.h5', 
        pretrained_itos = 
            '../data/imdb/weights_pretrained/fitos_wt103.pkl',
        display_epoch_freq = 1, num_epochs = 15, scheduler = 'ulmfit', 
        max_lrs = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
```
Alternatively you can load the weights from a pretrained language model and fine tune them to your corpus. Typically achieves better performance. Parameters embedding size and hidden state size are inferred from the stored model. 

Parameter 'scheduler' is to be used if learning rate is to be modified within an epoch as per policies 'triangular' or 'ulmfit'. For details on cyclical learning rates refer this [paper](https://arxiv.org/abs/1506.01186). Documentation on the policies and how to use them is in class [CyclicLR](https://github.com/SMAPPNYU/smapp_language_model/blob/3875c532b4675a104a4e98ae889f7c23e15e5d2b/lr_scheduler.py#L59)
