import argparse
import time
import os.path as osp
import nltk.tag.hmm as hmm
from nltk.probability import LaplaceProbDist as laplace, ConditionalProbDist, FreqDist, ConditionalFreqDist
from nltk.probability import MLEProbDist
from nltk.corpus import treebank
from sklearn.metrics import accuracy_score as score
from tqdm import tqdm
import string


class DataLoader:

    def __init__(self, data_dir, task, extra_data=None):
        self.train_cipher = osp.join(data_dir, task, 'train_cipher.txt')
        self.train_plain = osp.join(data_dir, task, 'train_plain.txt')
        self.test_cipher = osp.join(data_dir, task, 'test_cipher.txt')
        self.test_plain = osp.join(data_dir, task, 'test_plain.txt')
        self.extra_data = extra_data
        self.train_corpus = None
        self.test_corpus = None
        self.train_dictionary = None

    def __prepare_train_corpus__(self):

        if not osp.exists(self.train_cipher):
            print('INVALID TRAIN CIPHER PATH: {}'.format(self.train_cipher))
            exit(0)

        if not osp.exists(self.train_plain):
            print('INVALID TRAIN PLAIN PATH: {}'.format(self.train_plain))
            exit(0)

        plain = list()
        cipher = list()
        p = open(self.train_plain, 'r', errors='ignore')
        c = open(self.train_cipher, 'r', errors='ignore')

        for encoded, decoded in zip(c.readlines(), p.readlines()):
            encoded = list(encoded.rstrip('\n'))
            decoded = list(decoded.rstrip('\n'))

            plain.append(decoded)
            cipher.append(encoded)

        self.train_corpus = [cipher, plain]

    def get_train_corpus(self):
        if self.train_corpus is None:
            self.__prepare_train_corpus__()
        return self.train_corpus

    def __prepare_test_corpus__(self):

        if not osp.exists(self.test_cipher):
            print('INVALID TEST CIPHER PATH: {}'.format(self.test_cipher))
            exit(0)

        if not osp.exists(self.test_plain):
            print('INVALID TEST PLAIN PATH: {}'.format(self.test_plain))
            exit(0)

        plain = list()
        cipher = list()
        p = open(self.test_plain, 'r', errors='ignore')
        c = open(self.test_cipher, 'r', errors='ignore')

        for encoded, decoded in zip(c.readlines(), p.readlines()):
            encoded = list(encoded.rstrip('\n'))
            decoded = list(decoded.rstrip('\n'))

            plain.append(decoded)
            cipher.append(encoded)

        self.test_corpus = [cipher, plain]

    def get_test_corpus(self):
        if self.test_corpus is None:
            self.__prepare_test_corpus__()
        return self.test_corpus

    @staticmethod
    def flat(l):
        f = list()
        for i in l:
            f += i
        return f


class HMMTrainer(hmm.HiddenMarkovModelTrainer):

    def train_supervised(self, labelled_sequences, extra_data=False, estimator=None):
        # This is copied from HiddenMarkovModelTrainer

        if estimator is None:
            estimator = lambda fdist, bins: MLEProbDist(fdist)

        # count occurrences of starting states, transitions out of each state
        # and output symbols observed in each state
        known_symbols = set(self._symbols)
        known_states = set(self._states)

        starting = FreqDist()
        transitions = ConditionalFreqDist()
        outputs = ConditionalFreqDist()
        for sequence in labelled_sequences:
            lasts = None
            for token in sequence:
                state = token[1]
                symbol = token[0]
                if lasts is None:
                    starting[state] += 1
                else:
                    transitions[lasts][state] += 1
                outputs[state][symbol] += 1
                lasts = state

                # update the state and symbol lists
                if state not in known_states:
                    self._states.append(state)
                    known_states.add(state)

                if symbol not in known_symbols:
                    self._symbols.append(symbol)
                    known_symbols.add(symbol)

        if extra_data:
            print('-'*20)
            print("Using extra data to calculate transition probability")
            sent = ""
            for word in tqdm(treebank.words()):
                if word == '.':
                    sent = sent[:-1] + word
                    lasts = None
                    for c in sent:
                        if c in list(string.ascii_lowercase)+[' ', ',', '.']:
                            if lasts is not None:
                                transitions[lasts][c] += 1
                        lasts = c
                    sent = ""
                elif word == ',':
                    sent = sent[:-1] + word + ' '
                else:
                    sent += word + ' '

        # create probability distributions (with smoothing)
        N = len(self._states)
        pi = estimator(starting, N)
        A = ConditionalProbDist(transitions, estimator, N)
        B = ConditionalProbDist(outputs, estimator, len(self._symbols))

        return hmm.HiddenMarkovModelTagger(self._symbols, self._states, A, B, pi)

class Tagger:

    def __init__(self, data_loader, laplace=False, lm=False):
        self.data_loader = data_loader
        self.laplace = laplace
        self.dictionary = list(string.ascii_lowercase) + [',', ' ', '.']
        self.model = HMMTrainer(states=self.dictionary, symbols=self.dictionary)
        self.lm = lm

    def train(self):
        print('-' * 20)
        print("Start training standard HMM tagger")
        train_corpus = self.data_loader.get_train_corpus()
        labelled = list()
        for cipher, text in zip(train_corpus[0], train_corpus[1]):
            labelled.append(zip(cipher, text))

        if self.laplace:
            self.model = self.model.train_supervised(labelled_sequences=labelled, extra_data=self.lm, estimator=laplace)
        else:
            self.model = self.model.train_supervised(labelled_sequences=labelled, extra_data=self.lm)
        prediction = self._tag(train_corpus[0])
        acc = score(DataLoader.flat(train_corpus[1]), DataLoader.flat(prediction))
        print("Training Accuracy : %f" % acc)
        self.display(prediction)

    def test(self):
        print('-' * 20)
        print('Start testing the HMM tagger')
        test_corpus = self.data_loader.get_test_corpus()
        prediction = self._tag(test_corpus[0])
        acc = score(DataLoader.flat(test_corpus[1]), DataLoader.flat(prediction))
        print("Testing Accuracy : %f" % acc)
        self.display(prediction)

    def _tag(self, corpus):
        prediction = list()
        for sent in corpus:
            decode = self.model.tag(sent)
            prediction.append([d for (_, d) in decode])
        return prediction

    @staticmethod
    def display(prediction):
        print("-"*20)
        print("Tagged sentences")
        for lines in prediction:
            print("".join(lines))


def main():
    parser = argparse.ArgumentParser("COMP550_Assignment2_Question3")
    parser.add_argument('--data_dir', type=str, default='D:\\McGill\\19Fall\\COMP 550\\Project\\data\\a2data',
                        help='Path to data directory')
    parser.add_argument('-laplace', action='store_true', help='Use laplace smoothing')
    parser.add_argument('-lm', action='store_true', help='Improved plaintext modelling')
    parser.add_argument('cipher_folder', type=str, default='cipher1', choices=['cipher1', 'cipher2', 'cipher3'], help='Which cipher to use')
    config = parser.parse_args()

    task = config.cipher_folder

    # Modelling start below here
    dl = DataLoader(config.data_dir, task)
    # Declare the tagger
    tagger = Tagger(dl, config.laplace, config.lm)
    # Train the tagger
    tagger.train()
    # Test the tagger
    tagger.test()


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('Total time consumption: %fs' % (end_time - start_time))
