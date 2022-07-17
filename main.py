import itertools
import torch

from collections import namedtuple
from os import cpu_count
from typing import Iterable, Tuple
from multiprocessing import Pool
from model.evaluate import evaluate
from model.model import Encoder, Decoder
from model.train import train
from model.vocabulary import Vocabulary, tensor_from_sentence

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
    
    def unlimited_tensors(source: Iterable[Iterable[str]]):
        while True:
            for abstract in source:
                yield tensor_from_sentence(vocabulary, abstract, device) 

    acl_tensors = unlimited_tensors(acl_abstracts)
    cs_theory_tensors = unlimited_tensors(abstracts)

    def evaluate_hyperparameter_configuration(hyperparameters: Hyperparameters) -> Tuple[float, Hyperparameters]:
        encoder = Encoder(len(vocabulary), hyperparameters.hidden_size, device)
        decoder = Decoder(len(vocabulary), hyperparameters.hidden_size, device)

        for _ in range(1):
            train(encoder, decoder, acl_tensors, device, MAX_LENGTH, iterations=10)

        for _ in range(1):
            loss = train(encoder, decoder, cs_theory_tensors, device, MAX_LENGTH, iterations=1)
        
        # TODO: return a validation loss instead of a train loss
        return loss, hyperparameters

    hyperparameter_configurations_tried = 0
    hyperparameter_configurations = list(seed_hyperparameter_configurations)  # TODO: make this a priority queue
    hyperparameter_configurations_which_have_already_been_tested = set(seed_hyperparameter_configurations)

    number_of_hyperparameters_to_consider_per_step = 10
    number_of_full_stages_to_consider = total_number_of_hyperparameter_configurations_to_try // number_of_hyperparameters_to_consider_per_step 
    residual_for_the_last_stage = total_number_of_hyperparameter_configurations_to_try % number_of_hyperparameters_to_consider_per_step 

    top = lambda iterator, k: sorted(iterator)[:k]
    def valid(hyperparameters: Hyperparameters) -> bool:
        return 1 <= hyperparameters.hidden_size

    with Pool(cpu_count()) as pool:
        for _ in range(number_of_full_stages_to_consider):
            hyperparameters_with_score = pool.map(evaluate_hyperparameter_configuration, hyperparameter_configurations)
            best_hyperparameter_configurations = [hp for score, hp in top(hyperparameters_with_score, number_of_hyperparameters_to_consider_per_step)]
            hyperparameter_configurations = []
            for hyperparameters in best_hyperparameter_configurations:
                for hyperparameter_configuration in (Hyperparameters(*args) for args in itertools.product([hyperparameters.hidden_size * 2, hyperparameters.hidden_size // 2])):
                    if valid(hyperparameter_configuration) and hyperparameter_configuration not in hyperparameter_configurations_which_have_already_been_tested:
                       hyperparameter_configurations.append(hyperparameter_configuration)
                       hyperparameter_configurations_which_have_already_been_tested.add(hyperparameter_configuration)