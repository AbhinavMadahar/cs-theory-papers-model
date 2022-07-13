import re


SOS_token = 0
EOS_token = 1


class Vocabulary:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence: str) -> None:
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word: str) -> None:
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def normalize(text: str) -> str:
    text = text.lower()
    text = [word for word in text.split(' ') if all('a' <= character <= 'z' for character in word)]
    return text