from typing import Iterable
import torch

from torch import tensor
from model.model import Encoder, Decoder
from model.train import train
from model.evaluate import evaluate
from model.vocabulary import Vocabulary, tensor_from_sentence

# the hyperparameters for the model, the training process, etc.
# are stored here. when we implement beam search, we can use
# this dictionary to store all the information for the beam search.
hyperparameters = {
    'hidden_size': 64,
}

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

    encoder = Encoder(len(vocabulary), hyperparameters['hidden_size'], device)
    decoder = Decoder(len(vocabulary), hyperparameters['hidden_size'], device)

    iterations_between_evaluation = 1000
    for _ in range(len(acl_abstracts) // iterations_between_evaluation):
        train(encoder, decoder, acl_tensors, device, MAX_LENGTH, print_every=10, iterations=iterations_between_evaluation)
        print(*[vocabulary.index2word[index] for index in evaluate(encoder, decoder, next(acl_tensors), device, MAX_LENGTH)], sep=' ')

    for _ in range(100):
        train(encoder, decoder, cs_theory_tensors, device, MAX_LENGTH, print_every=1, iterations=100)
        print(*[vocabulary.index2word[index] for index in evaluate(encoder, decoder, next(cs_theory_tensors), device, MAX_LENGTH)], sep=' ')
