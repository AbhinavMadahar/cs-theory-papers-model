from model.model import Encoder, Decoder
from model.train import train
from model.vocabulary import Vocabulary

abstracts_file_name = 'data/abstracts.txt'

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