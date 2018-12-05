import os
import datetime
import pandas as pd
import torch
import torch.nn as nn
from lr_scheduler import CyclicLR
from training_utils import training_loop, test_loop
from model import RNNLM
from data_utils import IndexVectorizer, TextDataset, simple_tokenizer

# GPU variables
use_gpu = torch.cuda.is_available()
device_num = 0
device = torch.device(f"cuda:{device_num}" if use_gpu else "cpu")

# Text-related global variables
max_seq_len = 30
min_word_freq = 20

# File-writing variables
today = datetime.datetime.now().strftime('%Y-%m-%d')
working_directory = './'
data_directory = os.path.join(working_directory, '../../data/yelp')
model_directory = os.path.join(working_directory, 'model')


train_file = os.path.join(data_directory, 'train.csv')
valid_file = os.path.join(data_directory, 'valid.csv')
test_file = os.path.join(data_directory, 'test.csv')

model_file_lm = os.path.join(model_directory, f'LM__{today}.json')
model_file_class = os.path.join(model_directory, f'CLASS__{today}.json')
for _dir in [working_directory, model_directory, data_directory]:
    os.makedirs(_dir, exist_ok=True)

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
valid= pd.read_csv(valid_file)

vectorizer = IndexVectorizer(max_words=None, min_frequency=min_word_freq, 
                             start_end_tokens=True, maxlen=max_seq_len)

# these can take a DF or a path to a csv
training_set = TextDataset(train, text_col='text',
                           vectorizer=vectorizer, 
                           tokenizer=simple_tokenizer)
test_set = TextDataset(test, text_col='text',
                       vectorizer=vectorizer, 
                       tokenizer=simple_tokenizer)
validation_set = TextDataset(valid, text_col='text',
                             vectorizer=vectorizer, 
                             tokenizer=simple_tokenizer)

print("Next line should be 5k, 1k, 2k")
print(len(training_set), len(validation_set), len(test_set))
print("Next line should be 2389")
print("Vocab size: {}".format(vectorizer.vocabulary_size))

if use_gpu: torch.cuda.manual_seed(303)
else: torch.manual_seed(303)

# set up Files to save stuff in
runtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
model_file_lm = model_file_lm
    
# Model Hyper Parameters 
hidden_dim = 100
embedding_dim = 200
batch_size = 50
dropout = 0.2
lstm_layers = 1 # this is useless atm
lstm_bidirection = True

# Training
learning_rate = 1e-3
num_epochs = 300
display_epoch_freq = 10

# Build and initialize the model
lm = RNNLM(device, vectorizer.vocabulary_size, max_seq_len, embedding_dim, hidden_dim, batch_size, 
           dropout = dropout, 
           tie_weights = False, 
           num_layers = lstm_layers, 
           bidirectional = lstm_bidirection, 
           word2idx = vectorizer.word2idx,
           log_softmax = True)

if use_gpu:
    lm = lm.to(device)
lm.init_weights()

# Loss and Optimizer
loss = nn.NLLLoss()
optimizer = torch.optim.Adam(
		[
			{"params":lm.lstm1.parameters(), "lr":0.002},
			{"params":lm.lstm2.parameters(), "lr":0.003}
		], lr=learning_rate)


scheduler = CyclicLR(optimizer,  max_lrs = [0.01, 0.008], mode = 'ulmfit', ratio = 3, cut_frac = 0.4, n_epochs = num_epochs, batchsize = batch_size, verbose = False, epoch_length = 5000)

history = training_loop(batch_size, num_epochs, display_epoch_freq, 
                        lm, loss, optimizer, scheduler, device, 
                        training_set, validation_set, 
                        best_model_path=model_file_lm)

lm.load_state_dict(torch.load(model_file_lm))
test_loop(256, lm, loss, optimizer, scheduler, test_set, device=device)

