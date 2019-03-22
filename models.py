import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler
from torchsummary import summary
from torch.distributions.categorical import Categorical

import math
import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# user-defined RNN subclass of torch.nn.Module
class Recurrent(torch.nn.Module):
    def __init__(self, x_features, h_features, out_features,hidden_size):
        super(Recurrent, self).__init__()

        self.x_term = torch.nn.Linear(x_features, out_features, bias=False)
        self.h_term = torch.nn.Linear(h_features, out_features, bias=True)

        dl = 1.0/math.sqrt(hidden_size)
        nn.init.uniform_(self.x_term.weight,-dl, dl)
        nn.init.uniform_(self.h_term.weight,-dl, dl)
        nn.init.uniform_(self.h_term.bias,-dl, dl)
        self.tanh = nn.Tanh()
    def forward(self, x, h_prev):
        a_t = self.x_term(x) + self.h_term(h_prev)
        return self.tanh(a_t)


class GRU_Layer(torch.nn.Module):
    def __init__(self, x_features, h_features, out_features,hidden_size):
        super(GRU_Layer, self).__init__()
        self.Wr = torch.nn.Linear(x_features, out_features, bias=False)
        self.Ur = torch.nn.Linear(h_features, out_features, bias=True)

        self.Wz = torch.nn.Linear(x_features, out_features, bias=False)
        self.Uz = torch.nn.Linear(h_features, out_features, bias=True)

        self.Wh = torch.nn.Linear(x_features, out_features, bias=False)
        self.Uh = torch.nn.Linear(h_features, out_features, bias=True)

        dl = 1.0/math.sqrt(hidden_size)
        nn.init.uniform_(self.Ur.weight,-dl, dl)
        nn.init.uniform_(self.Ur.bias,-dl, dl)

        nn.init.uniform_(self.Uz.weight,-dl, dl)
        nn.init.uniform_(self.Uz.bias,-dl, dl)

        nn.init.uniform_(self.Uh.weight,-dl, dl)
        nn.init.uniform_(self.Uh.bias,-dl, dl)

        nn.init.uniform_(self.Wr.weight,-dl, dl)
        nn.init.uniform_(self.Wz.weight,-dl, dl)
        nn.init.uniform_(self.Wh.weight,-dl, dl)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h_prev):
        r = self.sigmoid(self.Wr(x) + self.Ur(h_prev))
        z = self.sigmoid(self.Wz(x) + self.Uz(h_prev))
        h_tilde = self.tanh(self.Wh(x) + self.Uh((r * h_prev)))
        h_t = (1 - z) * h_prev + z * h_tilde
        return h_t


class Stacked_layer(nn.Module):
    def __init__(self, X_in, H_in, D_out, p,hidden_size, rec_model='Recurrent'):
        super(Stacked_layer, self).__init__()
        if rec_model == 'Recurrent':
            self.recurrent_layer = Recurrent(X_in, H_in, D_out,hidden_size)
        else:
            self.recurrent_layer = GRU_Layer(X_in, H_in, D_out,hidden_size)
        self.dropout_layer = nn.Dropout(p=p)

    def forward(self, x, h):
        hidden = self.recurrent_layer(x, h)
        out = self.dropout_layer(hidden)
        return out, hidden


class Embedding_layer(nn.Module):
    def __init__(self, vocab_size, emb_size, p):
        super(Embedding_layer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.dropout_layer = nn.Dropout(p=p)
        nn.init.uniform_(self.embedding.weight,-0.1, 0.1)

    def forward(self, inputs):
        x = self.embedding(inputs)
        return self.dropout_layer(x)

class RNN(nn.Module):  # Implement a stacked vanilla RNN with Tanh nonlinearities.
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        super(RNN, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob

        dp_prob = 1 - self.dp_keep_prob

        self.layers = nn.ModuleList()

        self.emb_layer = Embedding_layer(self.vocab_size, self.emb_size, p=dp_prob)
        for layer in range(1, self.num_layers+1):
            self.layers.append(Stacked_layer(
                                X_in = self.emb_size if  layer==1 else self.hidden_size,
                                H_in = self.hidden_size,
                                D_out = self.hidden_size,
                                p=dp_prob,
                                hidden_size = self.hidden_size
                                ))
        self.layers.append(nn.Linear(self.hidden_size, self.vocab_size, bias=True))

        self.init_weights_uniform()
    def init_weights_uniform(self):
        nn.init.uniform_(self.layers[-1].weight,-0.1, 0.1)
        nn.init.zeros_(self.layers[-1].bias)
        nn.init.uniform_(self.emb_layer.embedding.weight,-0.1, 0.1)


    def init_hidden(self):
        hiddenState = torch.tensor(())
        return hiddenState.new_zeros(self.num_layers, self.batch_size, self.hidden_size,requires_grad=False)

    def forward(self, inputs, hidden):
        logits = []
        for i in range(self.seq_len):
            temp = []
            out = self.emb_layer.forward(inputs[i])
            for j in range(self.num_layers):
                out, h = self.layers[j].forward(out, hidden[j])
                temp.append(h)
            hidden = torch.stack(temp)
            out = self.layers[-1].forward(out)
            logits.append(out)
        logits = torch.stack(logits)
        return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

    def generate(self, input, hidden, generated_seq_len):
        samples = []
        for i in range(generated_seq_len):
            temp = []
            out = self.emb_layer.forward(input)
            for j in range(self.num_layers):
                out, h = self.layers[j].forward(out, hidden[j])
                temp.append(h)

            hidden = torch.stack(temp)
            out = self.self.layers[-1].forward(out)
            probs = torch.softmax(out, axis=1)
            input = Categorical(probs=probs).sample()
            samples.append(input)
        samples = torch.stack(samples)
        return samples


class GRU(nn.Module):  # Implement a stacked GRU RNN
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        super(GRU, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob

        dp_prob = 1 - self.dp_keep_prob

        self.layers = nn.ModuleList([])

        self.emb_layer = Embedding_layer(self.vocab_size, self.emb_size, dp_prob)
        for layer in range(1, self.num_layers+1):
            self.layers.append(Stacked_layer(
                                X_in = self.emb_size if layer ==1 else self.hidden_size,
                                H_in = self.hidden_size,
                                D_out = self.hidden_size,
                                p=dp_prob,
                                hidden_size=self.hidden_size,
                                rec_model='GRU'
                                ))
        self.layers.append(nn.Linear(self.hidden_size, self.vocab_size, bias=True))

        self.init_weights_uniform()

    def init_weights_uniform(self):
        nn.init.uniform_(self.emb_layer.embedding.weight,-0.1, 0.1)
        nn.init.uniform_(self.layers[-1].weight, -0.1, 0.1)
        nn.init.zeros_(self.layers[-1].bias)

    def init_hidden(self):
        hiddenState = torch.tensor(())
        return hiddenState.new_zeros(self.num_layers, self.batch_size, self.hidden_size,requires_grad=False)

    def forward(self, inputs, hidden):
        logits = []
        for i in range(self.seq_len):
            temp = []
            out = self.emb_layer.forward(inputs[i])
            for j in range(self.num_layers):
                out, h = self.layers[j].forward(out, hidden[j])
                temp.append(h)
            hidden = torch.stack(temp)
            out = self.layers[-1].forward(out)
            logits.append(out)
        logits = torch.stack(logits)
        return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

    def generate(self, inpute, hidden, generated_seq_len):
        samples = []
        for i in range(generated_seq_len):
            temp = []
            out = self.emb_layer.forward(input)
            for j in range(self.num_layers):
                out, h = self.layers[j].forward(out, hidden[j])
                temp.append(h)
            hidden = torch.stack(temp)
            out = self.self.layers[-1].forward(out)
            probs = torch.softmax(out, axis=1)
            input = Categorical(probs=probs).sample()
            samples.append(input)
        samples = torch.stack(samples)
        return samples


# Problem 3
class OneHeadedAttention(nn.Module):
    def __init__(self, n_units, d_k,dropout):
        super(OneHeadedAttention, self).__init__()

        self.d_k = d_k
        self.n_units = n_units

        self.W_q = nn.Linear(self.n_units, self.d_k)
        self.W_k = nn.Linear(self.n_units, self.d_k)
        self.W_v = nn.Linear(self.n_units, self.d_k)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        x_hat = (Q @ torch.transpose(K, 1, 2))/math.sqrt(self.d_k)

        if mask is not None:
            mask = mask.float()
            x_hat_masked = (x_hat * mask) - ((10**9)*(1-mask))
        else:
            x_hat_masked = x_hat

        softmax_output = F.softmax(x_hat_masked, -1)
        dropout_output = self.dropout(softmax_output)

        out = dropout_output @ V
        return out  # size: (batch_size, seq_len, self.n_units)

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()

        self.d_k = n_units // n_heads

        assert n_units % n_heads == 0

        self.n_units = n_units
        self.n_heads = n_heads

        m = math.sqrt(1/self.n_units)

        self.attention = clones(OneHeadedAttention(
                                 n_units = self.n_units,
                                 d_k = self.d_k,
                                 dropout = dropout),
                                 n_heads)
        self.W_o = nn.Linear(self.n_units, self.n_units)

        self.init_weights_uniform(m)

    def init_weights_uniform(self,m):
        torch.nn.init.uniform_(self.W_o.weight, -m, m)
        torch.nn.init.uniform_(self.W_o.bias, -m, m)
        for i in range(self.n_heads):
            nn.init.uniform_(self.attention[i].W_q.weight, -m, m)
            nn.init.uniform_(self.attention[i].W_q.bias, -m, m)

            nn.init.uniform_(self.attention[i].W_k.weight, -m, m)
            nn.init.uniform_(self.attention[i].W_k.bias, -m, m)

            nn.init.uniform_(self.attention[i].W_v.weight, -m, m)
            nn.init.uniform_(self.attention[i].W_v.bias, -m, m)

    def forward(self, query, key, value, mask=None):
        Z = torch.cat([self.attention[head].forward(query, key, value, mask) for head in range(self.n_heads)],-1)
        return self.W_o(Z)  # size: (batch_size, seq_len, self.n_units)

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        # print (x)
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


# ----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # apply the self-attention
        return self.sublayer[1](x, self.feed_forward)  # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """

    def __init__(self, layer, n_blocks):  # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)

    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6,
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
    )

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# ----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)

    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


# ----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """

    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
