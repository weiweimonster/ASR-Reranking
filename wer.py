from typing import List, Any, Tuple
import pandas as pd
import torch
from evaluate import load
import csv
import numpy as np
from util import *


def rerank_sentences_for_wer(model: Any, wer_data: List[Any], savepath: str, transformer=False, atoi=None):
    """
    Function to rerank candidate sentences in the HUB dataset. For each set of sentences,
    you must assign each sentence a score in the form of the sentence's acoustic score plus
    the sentence's log probability. You should then save the top scoring sentences in a .csv
    file similar to those found in the results directory.

    Inputs:
        model (Any): An n-gram or Transformer model.
        wer_data (List[Any]): Processed data from the HUB dataset. 
        savepath (str): The path to save the csv file pairing sentence set ids and the top ranked sentences.
    """
    # TODO
    with open(savepath, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['id', 'sentences'])  # Confirm the header matches the expected format

        if transformer:
            for id, data in wer_data:
                best_score = float("-inf")
                top_s = None
                for s, score in data:
                    input_tokens, target_tokens = to_tokens(s, atoi)
                    input_tokens = input_tokens.unsqueeze(0)
                    target_tokens = target_tokens.unsqueeze(0)
                    log_prob = model.log_probability(input_tokens, target_tokens)
                    combined_score = score + log_prob
                    if combined_score > best_score:
                        best_score = combined_score
                        top_s = ''.join(s)
                csvwriter.writerow([id, top_s])
        else:
            for id, hypotheses, scores in wer_data:
                best_score = float('-inf')
                top_sentence = None

                for hypothesis, acoustic_score in zip(hypotheses, scores):
                    log_prob = model.log_probability(hypothesis, base=np.e)  # Ensure the model and method are correctly set up for this
                    combined_score = acoustic_score + log_prob

                    if combined_score > best_score:
                        best_score = combined_score
                        top_sentence = ' '.join(hypothesis)

                csvwriter.writerow([id, top_sentence])


def compute_wer(gt_path, model_path):

    # Load the sentences
    ground_truths = pd.read_csv(gt_path)['sentences'].tolist()
    guesses = pd.read_csv(model_path)['sentences'].tolist()

    # Compute wer
    wer = load("wer")
    wer_value = wer.compute(predictions=guesses, references=ground_truths)
    return wer_value
