import os
import json
import csv
from typing import List, Tuple
import random
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from collections import Counter
import re


def load_pre_tokenized_text_file(file_path: str) -> List[List[str]]:
    """Loads a pre-tokenized text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip().split() for line in file.readlines()]  # Split each line into words
    return lines


def tokenize_sentences(sentences: List[str], tokenizer: Tokenizer) -> List[List[str]]:
    """Tokenizes a list of sentences using the provided tokenizer."""
    return [tokenizer.encode(sentence).tokens for sentence in sentences]


def split_train_val(data: List[List[str]], split_ratio: float = 0.9) -> Tuple[List[List[str]], List[List[str]]]:
    """Splits the data into training and validation sets."""
    random.shuffle(data)
    split_point = int(len(data) * split_ratio)
    return data[:split_point], data[split_point:]


def load_wer_data(ground_truths_path: str, hypotheses_path: str) -> Tuple[List[List[str]], List[List[List[str]]], List[List[float]]]:
    """Loads WER data, including ground truths, hypotheses, and acoustic scores."""
    # Load ground truths
    """Loads WER hypotheses and acoustic scores, keeping original IDs."""
    with open(hypotheses_path, 'r', encoding='utf-8') as jsonfile:
        hypotheses_json = json.load(jsonfile)

    wer_data = []
    for id, info in hypotheses_json.items():
        hypotheses = [h.split() for h in info['sentences']]
        scores = info['acoustic_scores']
        wer_data.append((id, hypotheses, scores))

    return wer_data


def load_wer_data_character(path: str):
    data = []
    with open(path, 'r', encoding='utf-8') as jsonfile:
        data_json = json.load(jsonfile)
    for id, x in data_json.items():
        sentences = x['sentences']
        scores = x['acoustic_scores']
        temp = []
        for sentence, score in zip(sentences, scores):
            chars = [ch for ch in sentence.strip()]
            temp.append([chars, score])
        data.append([id, temp])
    return data


def build_vocab_from_dataset(file_path: str, tokenizer: Tokenizer):
    word_counts = Counter()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            words = line.strip().split()  # Assuming space-separated tokens
            word_counts.update(words)
    
    # Update tokenizer vocabulary with each word and [UNK]
    for word in word_counts:
        tokenizer.token_to_id(word)  # This will add the word if not present

    # Ensure [UNK] token is in the vocabulary
    if "[UNK]" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["[UNK]"])
        
    return tokenizer


def load_data(tokenization_level: str, tokenizer_path, model_type: str) -> Tuple[any, ...]:
    """
    Load data for language modeling and WER computation.
    
    Returns:
        Tuple containing train_data, val_data, dev_data, test_data, dev_wer_data, test_wer_data.
    """

    if model_type == "transformer":
        return transformer_load_data()
    # Assuming your project directory structure
    base_path = 'data'
    wsj_path = os.path.join(base_path, 'lm_data')
    wer_path = os.path.join(base_path, 'wer_data')
    if tokenization_level == 'bpe' and tokenizer_path:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        train_data, val_data = split_train_val(tokenize_sentences(load_pre_tokenized_text_file(os.path.join(wsj_path, 'treebank-sentences-train.txt')),tokenizer))
        dev_data = tokenize_sentences(load_pre_tokenized_text_file(os.path.join(wsj_path, 'treebank-sentences-dev.txt')),tokenizer)
        test_data = tokenize_sentences(load_pre_tokenized_text_file(os.path.join(wsj_path, 'treebank-sentences-test.txt')),tokenizer)
    else:
        # Load WSJ dataset
        train_data, val_data = split_train_val(load_pre_tokenized_text_file(os.path.join(wsj_path, 'treebank-sentences-train.txt')))
        dev_data = load_pre_tokenized_text_file(os.path.join(wsj_path, 'treebank-sentences-dev.txt'))
        test_data = load_pre_tokenized_text_file(os.path.join(wsj_path, 'treebank-sentences-test.txt'))

    dev_ground_truths_path = os.path.join(wer_path, 'dev_ground_truths.csv')
    dev_hypotheses_path = os.path.join(wer_path, 'dev_sentences.json')
    dev_wer_data = load_wer_data(dev_ground_truths_path, dev_hypotheses_path)

    test_ground_truths_path = os.path.join(wer_path, 'test_ground_truths.csv')
    test_hypotheses_path = os.path.join(wer_path, 'revised_test_sentences.json')
    test_wer_data = load_wer_data(None, test_hypotheses_path)


    return train_data, val_data, dev_data, test_data, dev_wer_data, test_wer_data


def load_text_file_character(file_path: str):
    chars = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            chars += [ch for ch in line.strip()]
    return chars


def transformer_load_data(k=0.9):
    base_path = 'data'
    wsj_path = os.path.join(base_path, 'lm_data')
    wer_path = os.path.join(base_path, 'wer_data')
    train_data = load_text_file_character(os.path.join(wsj_path, 'treebank-sentences-train.txt'))
    split_point = int(len(train_data) * k)

    train_data, val_data = train_data[:split_point], train_data[split_point:]
    dev_data = load_text_file_character(os.path.join(wsj_path, 'treebank-sentences-dev.txt'))
    test_data = load_text_file_character(os.path.join(wsj_path, 'treebank-sentences-test.txt'))

    dev_ground_truths_path = os.path.join(wer_path, 'dev_ground_truths.csv')
    dev_hypotheses_path = os.path.join(wer_path, 'dev_sentences.json')
    dev_wer_data = load_wer_data_character(dev_hypotheses_path)

    test_ground_truths_path = os.path.join(wer_path, 'test_ground_truths.csv')
    test_hypotheses_path = os.path.join(wer_path, 'revised_test_sentences.json')
    test_wer_data = load_wer_data_character(test_hypotheses_path)
    return train_data, val_data, dev_data, test_data, dev_wer_data, test_wer_data


