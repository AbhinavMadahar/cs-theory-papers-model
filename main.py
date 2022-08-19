import argparse
import random
import torch

from collections import namedtuple
from enum import Enum
from os import cpu_count
from typing import Iterable, List, Tuple
from multiprocessing import Pool
from model.evaluate import evaluate
from model.generate import generate
from model.model import Encoder, Decoder
from model.train import train
from model.vocabulary import Vocabulary, tensor_from_sentence
from util.beam_search import beam_search


arg_parser = argparse.ArgumentParser(description='Control the NLG model')
arg_parser.add_argument('--mode', type=str, help='in which mode to run the code (i.e. debug or prod)')
args = arg_parser.parse_args()

# Describes the mode to run the code.
# Debug is when we just want to make sure the code works.
# In this mode, we only train for a few iterations at each step, and we don't look as deeply through the search space for hyperparameters.
# Prod is when we run the model in production.
# We train for many iterations and look deeply through the search space.
mode = args.mode

# the hyperparameters for the model, the training process, etc.
# are stored here. when we implement beam search, we can use
# this dictionary to store all the information for the beam search.
Hyperparameters = namedtuple('Hyperparameters', ['hidden_size'])
seed_hyperparameter_configurations = [
    Hyperparameters(hidden_size=64)
]

# we can keep running beam search for as long as we want to. if we
# wanted to, we could continuously run beam search, saving the best
# model found so far. when we need to run the model for real, we
# just use the best model found so far. over time, we would find
# better and better models. however, we don't have a machine which
# we can use continuously, so we search the hyperparameter space
# for a finite period of time, find the best model so far, and
# then stop. in particular, we consider only a certain number of
# hyperparameter configurations, given in the following variable.
# there are other ways to handle the stop condition; for example,
# we could run the beam search until a certain amount of time
# has elapsed, e.g. 72 hours.
total_number_of_hyperparameter_configurations_to_try = 100

abstracts_file_name = 'data/abstracts.txt'
acl_bib_file_name = 'data/acl.bib'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 500

def unlimited_tensors(source: Iterable[Iterable[str]]):
    while True:
        for abstract in source:
            yield tensor_from_sentence(vocabulary, abstract, device) 

if __name__ == '__main__':
    vocabulary = Vocabulary()

    with open(acl_bib_file_name, 'r') as acl_bib_file:
        acl_abstracts = []
        abstract_lines = []
        currently_on_abstract = False
        for line in acl_bib_file:
            line = line.strip()

            if line.startswith('abstract ='):
                currently_on_abstract = True
                line = line[len('abstract = "'):]

            if line == '}':
                if len(abstract_lines) > 0:
                    abstract = ' '.join(abstract_lines)
                    abstract = abstract.lower()
                    abstract = [word for word in abstract.split(' ') if all('a' <= char <= 'z' for char in word)]
                    acl_abstracts.append(abstract)
                    for word in abstract:
                        vocabulary.add_word(word)
                abstract_lines = []
                currently_on_abstract = False
            elif currently_on_abstract:
                abstract_lines.append(line)

        if len(abstract_lines) > 0:
            acl_abstracts.append(' '.join(abstract_lines))
    
    with open(abstracts_file_name, 'r') as abstracts_file:
        abstracts = []
        abstract_lines = []
        for line in abstracts_file:
            line = line.strip()

            if line == '':
                if len(abstract_lines) > 0:
                    abstract = ' '.join(abstract_lines)
                    abstract = abstract.lower()
                    abstract = [word for word in abstract.split(' ') if all('a' <= char <= 'z' for char in word)]
                    abstracts.append(abstract)
                    for word in abstract:
                        vocabulary.add_word(word)
                abstract_lines = []
            else:
                abstract_lines.append(line)
        if len(abstract_lines) > 0:
            abstracts.append(' '.join(abstract_lines))
    
    acl_tensors = unlimited_tensors(acl_abstracts)
    cs_theory_tensors = unlimited_tensors(abstracts)

    def evaluate_hyperparameter_configuration(configuration: Hyperparameters) -> Tuple[float, Hyperparameters]:
        encoder = Encoder(len(vocabulary), configuration.hidden_size, device)
        decoder = Decoder(len(vocabulary), configuration.hidden_size, device)

        for _ in range(1):
            train(encoder, decoder, acl_tensors, device, MAX_LENGTH, iterations=10 if mode == 'debug' else 1000)

        for _ in range(1):
            loss = train(encoder, decoder, cs_theory_tensors, device, MAX_LENGTH, iterations=1 if mode == 'debug' else 1000)
        
        # TODO: return a validation loss instead of a train loss
        return configuration, loss
    
    def children(configuration: Hyperparameters, number: int) -> List[Hyperparameters]:
        kids = []
        for _ in range(number):
            hidden_size = int(configuration.hidden_size * random.random() * 2)
            kids.append(Hyperparameters(hidden_size=hidden_size))
        return kids
    
    seed_hyperparameter_configurations = [Hyperparameters(hidden_size=64)]

    best_hyperparameter_configuration = beam_search(evaluate_hyperparameter_configuration,
                                                    children,
                                                    seed_hyperparameter_configurations,
                                                    mode)

    encoder = Encoder(len(vocabulary), best_hyperparameter_configuration.hidden_size, device)
    decoder = Decoder(len(vocabulary), best_hyperparameter_configuration.hidden_size, device)

    for _ in range(1):
        train(encoder, decoder, acl_tensors, device, MAX_LENGTH, iterations=10 if mode == 'debug' else 10000)

    for _ in range(1):
        loss = train(encoder, decoder, cs_theory_tensors, device, MAX_LENGTH, iterations=1 if mode == 'debug' else 1000)
    
    generated = generate(decoder, vocabulary, best_hyperparameter_configuration.hidden_size, MAX_LENGTH, device)
    print(generated)