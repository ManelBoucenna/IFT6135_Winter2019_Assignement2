import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import random
from models import RNN, GRU
from models import make_model as TRANSFORMER

torch.manual_seed(1111)
DATAPATH = "data"
OUTPUTPATH="./Problem5_generation"

# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")

# Processes the raw data from text files
def ptb_raw_data(data_path=None, prefix="ptb"):
    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")
    test_path = os.path.join(data_path, prefix + ".test.txt")

    word_to_id, id_2_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return train_data, valid_data, test_data, word_to_id, id_2_word

def generation(model, seq_len, batch_size=10, rawdata, device):

    #Load data
    train_data, valid_data, test_data, word_to_id, id_2_word = rawdata
    vocab_size = len(word_to_id)

    #Initialize model's hyperparameters
    model.batch_size = batch_size
    model.seq_len = seq_len
    model.vocab_size = vocab_size

    #Randomly choose firs word
    inputs = np.ones(batch_size)
    for i in range(batch_size):
        # Transform input word to id/token
        inputs[i]= word_to_id[random.randint(1,vocab_size)]

    first_token= torch.Tensor(inputs)
    hidden = model.init_hidden()

    samples = model.generate(first_token, hidden, seq_len)
    Separator = " "

    # Join all the generated word
    for sentence in range(batch_size):
        generated_sentences.append(Separator.join(id_2_word[first_word] + [id_2_word[id] for id in samples[:, sentence]]))

    return generated_sentences


if __name__ == "__main__":
    seq_lens = [35, 70]
    BatchSize = 10
    print('Enter best_param.pt path of RNN:')
    RNN_bestparams_path = input()
    print('Enter best_param.pt path of GRU:')
    GRU_bestparams_path = input()

     RNN = RNN(emb_size=200, hidden_size=1500,
                seq_len=seq_len, batch_size=20,
                num_layers=2).to(device)

    GRU =  GRU(emb_size=200, hidden_size=1500,
                seq_len=seq_len, batch_size=20,
                num_layers=2).to(device)

    print("Generation:")
    raw_data = ptb_raw_data(data_path=DATAPATH)
    for seq_len in range(seq_lens):
        print("Sequence length: ",seq_len)
        #RNN output
        #Load "Best params model"
        RNN.load_state_dict(torch.load(RNN_bestparams_path))
        RNN_generation = generation(RNN, seq_len, BatchSize, rawdata, device)
        print("RNN generated:")
        print(RNN_generation)
        with open(os.path.join(OUTPUTPATH, 'RNN_%s_samples.txt'%(seq_len)), 'w') as f:
            f.write("%s\n" % RNN_generation)

        #GRU output
        #Load "Best params model"
        GRU.load_state_dict(torch.load(GRU_bestparams_path))
        GRU_generation = generation(GRU, seq_len, BatchSize, rawdata, device)
        print("GRU generated:")
        print(GRU_generation)
        with open(os.path.join(OUTPUTPATH, 'GRU_%s_samples.txt'%(seq_len)), 'w') as f:
            f.write("%s\n" % GRU_generation)
