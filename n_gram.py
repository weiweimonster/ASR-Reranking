"""
n-gram language model for Assignment 2: Starter code.
"""
from collections import defaultdict

import os
import sys
import argparse
from typing import Dict, List, Any
from tqdm import tqdm
from collections import Counter
import numpy as np

from load_data import load_data
from perplexity import evaluate_perplexity
from wer import rerank_sentences_for_wer, compute_wer

def get_args():
    """
    You may freely add new command line arguments to this function.
    """
    parser = argparse.ArgumentParser(description='n-gram model')
    parser.add_argument('-t', '--tokenization_level', type=str, default='character',
                        help="At what level to tokenize the input data")
    parser.add_argument('-n', '--n', type=int, default=1,
                        help="The value of n to use for the n-gram model")

    parser.add_argument('-e', '--experiment_name', type=str, default='testing',
                        help="What should we name our experiment?")
    parser.add_argument('-s', '--num_samples', type=int, default=10,
                        help="How many samples should we get from our model??")
    parser.add_argument('-x', '--max_steps', type=int, default=40,
                        help="What should the maximum output length of our samples be?")

    args = parser.parse_args()
    return args

class NGramLM():
    """
    N-gram language model
    """

    def __init__(self, n: int, smoothing = False):
        """
        Initializes the n-gram model. You may add keyword arguments to this function
        to modify the behavior of the n-gram model. The default behavior for unit tests should
        be that of an n-gram model without any label smoothing.

        Important for unit tests: If you add <bos> or <eos> tokens to model inputs, this should 
        be done in data processing, outside of the NGramLM class. 

        Inputs:
            n (int): The value of n to use in the n-gram model
        """
        self.n = n
        self.pad_token = '[PAD]'
        self.unknown_token='[UNK]'
        self.vocabulary=set()
        self.vocabulary.add(self.pad_token)
        self.vocabulary.add(self.unknown_token)
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.smoothing = smoothing
    def log_probability(self, model_input: List[Any], base = np.e):
        """
        Returns the log-probability of the provided model input.

        Inputs:
            model_input (List[Any]): The list of tokens associated with the input text.
            base (float): The base with which to compute the log-probability
        """
        # TODO
        # log_prob = 0.0

        # for i in range(len(model_input) - self.n + 1):
        #     ngram = tuple(model_input[i:i + self.n])
        #     context = tuple(model_input[i:i + self.n - 1])

        #     ngram_count = self.ngram_counts[ngram]
        #     context_count = self.context_counts[context]

        #     if context_count > 0:
        #         log_prob += np.log(ngram_count / context_count)
        #     else:
        #         # Apply smoothing or handle zero context count
        #         # For example, you can add a small value to avoid division by zero
        #         log_prob += np.log(ngram_count / (context_count + 1))  # Additive smoothing

        # return log_prob / np.log(base)
        log_prob = 0.0
        model_input = [self.pad_token] * (self.n - 1) + model_input
        for i in range(len(model_input) - self.n + 1):
            n_gram = tuple(model_input[i:i + self.n])
            prefix = n_gram[:-1]

            n_gram_count = self.ngram_counts.get(n_gram, 0.0)
            prefix_count = self.context_counts.get(prefix, 0.0)

            if self.smoothing:
                smoothed_n_gram_count = n_gram_count + 1
                smoothed_prefix_count = prefix_count + len(self.vocabulary)
                prob = smoothed_n_gram_count / smoothed_prefix_count
            else:
                # Handling the case where the n-gram or its prefix has not been seen
                if n_gram_count == 0 or prefix_count == 0:
                    # Log of zero is undefined, so we return negative infinity
                    return float('-inf')

                prob = n_gram_count / prefix_count

            log_prob += np.log(prob) / np.log(base)

        return log_prob
    def generate(self, num_samples: int, max_steps: int, results_file: str):
        """
        Function for generating text using the n-gram model.

        Inputs:
            num_samples (int): How many samples to generate
            max_steps (int): The maximum length of any sampled output
            results_file (str): Where to save the generated examples
        """
        # TODO
        with open(results_file, 'w') as f:
            for _ in range(num_samples):
                tokens = ['<s>'] * (self.n - 1)
                for _ in range(max_steps):
                    token = self.predict_next_token(tokens)
                    if token == '</s>':
                        break
                    tokens.append(token)
                generated_text = ' '.join(tokens)
                f.write(generated_text + '\n')
        with open(results_file, 'w') as f:
            for text in generated_texts:
                f.write(f"{text}\n")
    def learn(self, training_data: List[List[Any]]):
        """
        Function for learning n-grams from the provided training data. You may
        add keywords to this function as needed, provided that the default behavior
        is that of an n-gram model without any label smoothing.
        
        Inputs:
            training_data (List[List[Any]]): A list of model inputs, which should each be lists
                                             of input tokens
        """
        # TODO
        self.ngram_counts[self.unknown_token] = 0

        for sentence in training_data:
            for token in sentence:
                self.vocabulary.add(token)
            sentence = [self.pad_token] * (self.n - 1) + sentence

            for i in range(len(sentence) - self.n + 1):
                ngram = tuple(sentence[i:i + self.n])
                context = tuple(sentence[i:i + self.n - 1])

                # Initialize counts if not present
                if ngram not in self.ngram_counts:
                    self.ngram_counts[ngram] = 0
                if context not in self.context_counts:
                    self.context_counts[context] = 0

                # Update counts
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1
    def get_probability(self, token: Any, context: tuple[Any, ...]):
        numerator = self.counts.get(context + (token,), 0)
        denominator = self.context_counts.get(context, 0)
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def predict_next_token(self, context: List[Any]):
        context = tuple(context[-(self.n - 1):])
        max_prob = float('-inf')
        best_token = None
        for token in self.vocab:
            prob = self.get_probability(token, context)
            if prob > max_prob:
                max_prob = prob
                best_token = token
        return best_token
def main():
    # Get key arguments
    args = get_args()

    # Get the data for language-modeling and WER computation
    tokenization_level = args.tokenization_level
    model_type = "n_gram"
    train_data, val_data, dev_data, test_data, dev_wer_data, test_wer_data = load_data("","trained_tokenizer.json", model_type) # TODO

    # Initialize and "train" the n-gram model
    n = 1
    model = NGramLM(n)
    model.learn(train_data)

    # Evaluate model perplexity
    val_perplexity = evaluate_perplexity(model, val_data)
    print(f'Model perplexity on the val set: {val_perplexity}')
    dev_perplexity = evaluate_perplexity(model, dev_data)
    print(f'Model perplexity on the dev set: {dev_perplexity}')
    test_perplexity = evaluate_perplexity(model, test_data)
    print(f'Model perplexity on the test set: {test_perplexity}')    

    # Evaluate model WER
    experiment_name = args.experiment_name
    dev_wer_savepath = os.path.join('results', f'{experiment_name}_n_gram_dev_wer_predictions.csv')

    rerank_sentences_for_wer(model, dev_wer_data, dev_wer_savepath)
    dev_wer = compute_wer('data/wer_data/dev_ground_truths.csv', dev_wer_savepath)
    print("Dev set WER was: ", dev_wer)

    test_wer_savepath = os.path.join('results', f'{experiment_name}_n_gram_test_wer_predictions.csv')
    rerank_sentences_for_wer(model, test_wer_data, test_wer_savepath)

    # Generate text from the model
    generation_path = os.path.join('generations', f'{experiment_name}_n_gram_generation_examples.pkl')
    num_samples = args.num_samples
    max_steps = args.max_steps
    model.generate(num_samples, max_steps, generation_path)
    

if __name__ == "__main__":
    main()
    
