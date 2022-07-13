from model.model import Encoder, Decoder
from model.train import train

abstracts_file_name = 'data/abstracts.txt'

if __name__ == '__main__':
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
                abstract_lines = []
            else:
                abstract_lines.append(line)
        if len(abstract_lines) > 0:
            abstracts.append(' '.join(abstract_lines))