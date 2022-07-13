import torch

from torch import tensor
from model.model import Encoder, Decoder
from model.train import train
from model.evaluate import evaluate
from model.vocabulary import Vocabulary, tensor_from_sentence

hidden_size = 128

abstracts_file_name = 'data/abstracts.txt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 500

if __name__ == '__main__':
    vocabulary = Vocabulary()

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
    
    def unlimited_tensors():
        while True:
            for abstract in abstracts:
                yield tensor_from_sentence(vocabulary, abstract, device) 
    tensors = unlimited_tensors()

    encoder = Encoder(len(vocabulary), hidden_size, device)
    decoder = Decoder(len(vocabulary), hidden_size, device)

    for _ in range(100):
        train(encoder, decoder, tensors, device, MAX_LENGTH, print_every=1, iterations=10)
        print(*[vocabulary.index2word[index] for index in evaluate(encoder, decoder, next(tensors), device, MAX_LENGTH)], sep=' ')