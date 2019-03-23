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
def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word
def _read_words(filename):
    with open(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()
def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def generation(model, train_data, valid_data, test_data, word_to_id, id_2_word, seq_len,batch_size=10):

    vocab_size = len(word_to_id)

    #Initialize model's hyperparameters
    model.batch_size = batch_size
    model.seq_len = seq_len
    model.vocab_size = vocab_size

    #Randomly choose firs word
    inputs = np.ones(batch_size)
    for i in range(batch_size):
        # Transform input word to id/token
        inputs[i]= random.randint(1,vocab_size)

    first_token= torch.LongTensor(inputs)
    hidden = model.init_hidden()

    samples = np.asarray(model.generate(first_token, hidden, seq_len))
    Separator = " "
    sentences =[]
    # Join all the generated word
    for sentence in range(batch_size):
        #Transform first id to word
        first_id = first_token[sentence].numpy()
        first_word = id_2_word[int(first_id)]

        restOfSentence =[]
        for id in samples[:, sentence]:
            restOfSentence.append(str(id_2_word[id]))
        sample = " ".join( [first_word] + restOfSentence)
        sentences.append(sample)

    return sentences


if __name__ == "__main__":
    seq_lens = [35, 70]
    BatchSize = 10
    # print('Enter best_param.pt path of RNN:')
    # RNN_bestparams_path = input()'/home/manal/Desktop/assignment2/Problem4.2/RNN_SGD_problem4.2'+'/best_params.pt'

    RNN_bestparams_path = '/home/manal/Desktop/assignment2/Problem4.2/RNN_SGD_problem4.2'+'/best_params.pt'
    # print('Enter best_param.pt path of GRU:')
    GRU_bestparams_path = '/home/manal/Desktop/assignment2/problem4.1/GRU_problem4.1'+'/best_params.pt'
    # GRU_bestparams_path = input()+'/best_params.pt'

    print("Generation:")
    raw_data = ptb_raw_data(data_path=DATAPATH)
    train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
    vocab_size = len(word_to_id)
    RNN = RNN(emb_size=200, hidden_size=1500,
                seq_len=0, batch_size=20,
                num_layers=2, vocab_size=vocab_size, dp_keep_prob=0.35).to(device)
    GRU =  GRU(emb_size=200, hidden_size=1500,
                seq_len=0, batch_size=20,
                num_layers=2, vocab_size=vocab_size, dp_keep_prob=0.35).to(device)
    for seq_len in seq_lens:
        print("Sequence length: ",seq_len)
        #RNN output
        #Load "Best params model"
        RNN.seq_len = seq_len
        RNN.load_state_dict(torch.load(RNN_bestparams_path, map_location=device))
        RNN_generation = generation(RNN, train_data, valid_data, test_data, word_to_id, id_2_word, seq_len,BatchSize)
#         print("RNN generated:")
#         print(RNN_generation)
        with open(os.path.join(OUTPUTPATH, 'RNN_%s_samples.txt'%(seq_len)), 'w') as f:
            f.write("Model RNN. Sequence length: %s\n" % (seq_len))
            for index,sentence in enumerate(RNN_generation):
                f.write("Sentence %s: %s\n" % (index,sentence))
        #GRU output
        #Load "Best params model"
        GRU.seq_len = seq_len
        GRU.load_state_dict(torch.load(GRU_bestparams_path,map_location=device))
        GRU_generation = generation(GRU, train_data, valid_data, test_data, word_to_id, id_2_word, seq_len, BatchSize)
#         print("GRU generated:")
#         print(GRU_generation)
        with open(os.path.join(OUTPUTPATH, 'GRU_%s_samples.txt'%(seq_len)), 'w') as f:
            f.write("Model GRU. Sequence length: %s\n" % (seq_len))
            for (index,sentence) in enumerate(GRU_generation):
                f.write("Sentence %s: %s\n" % (index,sentence))
