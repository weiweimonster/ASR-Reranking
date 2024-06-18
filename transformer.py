import os
import sys
import argparse
from typing import Dict, List
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from load_data import load_data
from perplexity import evaluate_perplexity
from wer import rerank_sentences_for_wer, compute_wer
from util import *
from treebank_dataset import TreebankDataset
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class CharacterLevelTransformer(nn.Module):
    """
    For this part of the assignment, we provide you with a skeleton for the Transformer
    decoder. However, we've introduced numerous errors to the code! The model currently compiles,
    but performs the incorrect computations. You must fix them to pass the unit tests.

    You may introduce additional keyword arguments after fixing the transformer, as long as the
    default behavior does not stray from the skeleton provided to you.
    """

    def __init__(self, num_layers: int, hidden_dim: int, num_heads: int,
                 ff_dim: int, dropout: float, vocab_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=vocab_size - 1)
        self.pos_embed = PositionalEncoding(hidden_dim, dropout)
        self.decoder = Decoder(num_layers, hidden_dim, num_heads, ff_dim, dropout)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.padding_idx = vocab_size - 1

    def log_probability(self, input_tokens: torch.Tensor, target_tokens: torch.Tensor, base=np.e):
        """
        Computes the log-probabilities for the inputs in the given minibatch.

        Input:
            input_tokens (torch.Tensor): A tensor of shape (B, T), where B is the 
                                         batch-size and T is the input length. 
            target_tokens (torch.Tensor): A tensor of shape (B, T). For a given (i, j),
                                          target_tokens[i, j] should be the token following
                                          input_tokens[i, j]
        Output (torch.Tensor): A tensor of shape (B,) containing the log-probability for each
                               example in the minibatch
        """
        # TODO
        # raise(NotImplementedError)
        o = self.forward(input_tokens)
        log_probs = F.log_softmax(o, dim=-1)

        target_log_probs = torch.gather(log_probs, -1, target_tokens.unsqueeze(-1)).squeeze(-1)

        mask = (target_tokens != self.padding_idx)

        # Apply mask to target log probabilities
        target_log_probs = target_log_probs * mask

        avg_log_prov_per_token = target_log_probs.sum(dim=1)
        if base != np.e:
            avg_log_prov_per_token /= np.log(base)

        return avg_log_prov_per_token

    def forward(self, model_input):
        # Perform the embedding
        embeds = self.embed(model_input) * math.sqrt(self.hidden_dim)
        embeds = self.pos_embed(embeds)

        # Pass through the decoder
        mask = construct_self_attn_mask(model_input)
        decoder_output = self.decoder(embeds, mask)
        output = self.lm_head(decoder_output)
        return output


def construct_self_attn_mask(x: torch.Tensor):
    """
    The output to this function should be a mask of shape
    (1, T, T). Indices that a token can attend to should be
    set to true.

    There are two errors in this function.
    """
    # fix
    T = x.size(1)
    all_ones = torch.ones(T, T)

    mask = torch.tril(all_ones).bool()
    mask = mask.unsqueeze(0)
    return mask.to(x.device)


class Decoder(nn.Module):

    def __init__(self, num_layers, hidden_dim, num_heads, ff_dim, dropout):
        """
        There is a single error in this function that will prevent the model from learning.
        """
        # fix
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(num_heads, hidden_dim, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, num_heads, hidden_dim, ff_dim, dropout):
        super().__init__()

        # Attention block
        self.attn_block = MultiHeadAttention(num_heads, hidden_dim, dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Feedforward block
        self.mlp_block = TransformerMLP(hidden_dim, ff_dim, dropout)
        self.mlp_dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):
        """
        There are two types of errors in this function.
        """
        # fix
        a_output = self.attn_block(x, mask)
        a_output = self.attn_dropout(a_output)
        x = self.attn_norm(x + a_output)

        mlp_output = self.mlp_block(x)
        mlp_output = self.mlp_dropout(mlp_output)
        x = self.mlp_norm(x + mlp_output)

        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, hidden_dim, dropout=0.1):
        super().__init__()

        self.h = num_heads
        self.qkv_dim = hidden_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def attention(self, query, key, value, mask):
        """
        There are three errors in this function to fix.
        """
        # fix
        # 1. the dot product should be between query and key
        dot_products = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(float(self.qkv_dim))
        # 2. the mask should be set to negative infinity for the softmax to go to 0
        dot_products = dot_products.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(dot_products, dim=-1))
        return torch.matmul(attn, value)

    def forward(self, x, mask):
        """
        There are two errors in this function to fix
        """
        # fix
        mask = mask.unsqueeze(1)
        B = x.size(0)

        # Compute the query, key and value vectors
        query = self.q_proj(x).view(B, -1, self.h, self.qkv_dim).transpose(1, 2)
        key = self.k_proj(x).view(B, -1, self.h, self.qkv_dim).transpose(1, 2)
        value = self.v_proj(x).view(B, -1, self.h, self.qkv_dim).transpose(1, 2)

        # Perform self-attention
        x = self.attention(query, key, value, mask)

        # Concatenate the outputs for each attention head
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.qkv_dim)
        return self.out_proj(x)


class TransformerMLP(nn.Module):

    def __init__(self, hidden_dim, ff_dim, dropout=0.1):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        There is a single error in this function to fix.
        """
        # fix
        # need to apply non-linearity
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x



class PositionalEncoding(nn.Module):

    def __init__(self, hidden_dim, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        positional_encodings = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (- math.log(10000) / hidden_dim))
        positional_encodings[:, 0::2] = torch.sin(position * div_term)
        positional_encodings[:, 1::2] = torch.cos(position * div_term)
        positional_encodings = positional_encodings.unsqueeze(0)

        self.register_buffer('positional_encodings', positional_encodings, persistent=False)

    def forward(self, x):
        x = x + self.positional_encodings[:, :x.size(1)]
        return self.dropout(x)


def train(model, train_loader, val_loader, dev_wer_data, loss_fct, optimizer, max_epochs, vocab_size):
    """
    Training loop for the transformer model. You may change the header as you see fit.
    """
    # TODO
    for epoch in range(max_epochs):
        model.train()
        losses = []
        accuracies = []
        val_losses = []
        val_accuracies = []
        for data, label in tqdm(train_loader, desc="training"):
            data = data.to(DEVICE)
            label = label.to(DEVICE)
            output = model(data)
            loss = loss_fct(output.contiguous().view(-1, vocab_size), label.contiguous().view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"epoch: {epoch}")
        print(f"train loss: {np.mean(losses)}")




def get_args():
    """
    You may freely add new command line arguments to this function.
    """
    parser = argparse.ArgumentParser(description='Transformer model')
    parser.add_argument('--num_layers', type=int, default=6,
                        help="How many transformer blocks to use")
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help="What is the transformer hidden dimension")
    parser.add_argument('--num_heads', type=int, default=8,
                        help="How many heads to use for Multihead Attention")
    parser.add_argument('--ff_dim', type=int, default=2048,
                        help="What is the intermediate dimension for the feedforward layer")
    parser.add_argument('--dropout_p', type=int, default=0.1,
                        help="The dropout probability to use")

    parser.add_argument('--experiment_name', type=str, default='testing_')
    parser.add_argument('--num_samples', type=int, default=10,
                        help="How many samples should we get from our model??")
    parser.add_argument('--max_steps', type=int, default=40,
                        help="What should the maximum output length be?")

    args = parser.parse_args()
    return args


def main():
    # Get key arguments
    args = get_args()

    # Get the data
    tokenization_level = "character"
    model_type = "transformer"
    train_data, val_data, dev_data, test_data, dev_wer_data, test_wer_data = load_data(
        tokenization_level, None, model_type
    )  # TODO
    block_size = 32
    train_params = {'batch_size': 2,
                    'shuffle': True,
                    'num_workers': 6}
    other_params = {'batch_size': 256,
                    'shuffle': False,
                    'num_workers': 6}
    atoi, iota = build_character_vocab(train_data)

    # load train dataset using custom Dataset and DataLoader
    train_dataset = TreebankDataset(train_data, atoi, iota, block_size)
    train_loader = DataLoader(train_dataset, **train_params)

    # load validation dataset using custom Dataset and DataLoader
    val_dataset = TreebankDataset(val_data, atoi, iota, block_size)
    val_loader = DataLoader(val_dataset, **other_params)

    # load dev dataset using custom Dataset and DataLoader
    dev_dataset = TreebankDataset(dev_data, atoi, iota, block_size)
    dev_loader = DataLoader(dev_dataset, **other_params)

    # load test dataset using custom Dataset and DataLoader
    test_dataset = TreebankDataset(test_data, atoi, iota, block_size)
    test_loader = DataLoader(test_dataset, **other_params)

    # Initialize the transformer and train
    num_layers = args.num_layers
    hidden_dim = args.hidden_dim
    num_heads = args.num_heads
    ff_dim = args.ff_dim
    dropout_p = args.dropout_p
    vocab_size = len(atoi)
    model = CharacterLevelTransformer(num_layers, hidden_dim, num_heads, ff_dim,
                                      dropout_p, vocab_size).to(DEVICE)
    #
    # optimizer = torch.optim.AdamW(model.parameters())
    # loss_fct = nn.CrossEntropyLoss(ignore_index=vocab_size-1)
    # max_epochs = 2
    # # TODO
    #
    # train(model, train_loader, val_loader, dev_wer_data, loss_fct, optimizer, max_epochs, vocab_size)
    #
    # # Evaluate model perplexity
    # model.eval()
    # val_perplexity = evaluate_perplexity(model, val_loader)
    # print(f'Model perplexity on the val set: {val_perplexity}')
    # dev_perplexity = evaluate_perplexity(model, dev_loader)
    # print(f'Model perplexity on the dev set: {dev_perplexity}')
    # test_perplexity = evaluate_perplexity(model, test_loader)
    # print(f'Model perplexity on the test set: {test_perplexity}')


    # Evaluate model WER
    experiment_name = args.experiment_name
    dev_wer_savepath = os.path.join('results', f'{experiment_name}transformer_dev_wer_predictions.csv')
    rerank_sentences_for_wer(model, dev_wer_data, dev_wer_savepath, True, atoi)
    dev_wer = compute_wer('data/wer_data/dev_ground_truths.csv', dev_wer_savepath)
    print("Dev set WER was: ", dev_wer)

    test_wer_savepath = os.path.join('results', f'{experiment_name}transformer_test_wer_predictions.csv')
    rerank_sentences_for_wer(model, test_wer_data, test_wer_savepath)

    # Generate text from the model
    generation_path = os.path.join('generations', f'{experiment_name}transformer_generation_examples.pkl')
    num_samples = args.num_samples
    max_steps = args.max_steps
    model.generate(num_samples, max_steps, generation_path)


if __name__ == "__main__":
    main()
