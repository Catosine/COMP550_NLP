import argparse
import time
import os.path as osp
import nltk.tag.hmm as hmm
from nltk.probability import LaplaceProbDist as laplace
from sklearn.metrics import accuracy_score as score
import string
import gc


class DataLoader:

    def __init__(self, data_dir, task, lm=False):
        self.train_cipher = osp.join(data_dir, task, 'train_cipher.txt')
        self.train_plain = osp.join(data_dir, task, 'train_plain.txt')
        self.test_cipher = osp.join(data_dir, task, 'test_cipher.txt')
        self.test_plain = osp.join(data_dir, task, 'test_plain.txt')
        self.lm = lm
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

            if self.lm:
                to_keep = list(string.ascii_lowercase) + [' ', ',', '.']
                for j, (char_e, char_d) in enumerate(zip(encoded, decoded)):
                    if char_e not in to_keep or char_d not in to_keep:
                        del encoded[j]
                        del decoded[j]
                        gc.collect()

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

            if self.lm:
                to_keep = list(string.ascii_lowercase) + [' ', ',', '.']
                for j, (char_e, char_d) in enumerate(zip(encoded, decoded)):
                    if char_e not in to_keep or char_d not in to_keep:
                        del encoded[j]
                        del decoded[j]
                        gc.collect()

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


class Tagger:

    def __init__(self, data_loader, laplace=False):
        self.data_loader = data_loader
        self.laplace = laplace
        self.dictionary = list(string.ascii_lowercase) + [',', ' ', '.']
        self.model = hmm.HiddenMarkovModelTrainer(states=self.dictionary, symbols=self.dictionary)

    def train(self):
        print('-' * 20)
        print("Start training standard HMM tagger")
        train_corpus = self.data_loader.get_train_corpus()
        labelled = list()
        for cipher, text in zip(train_corpus[0], train_corpus[1]):
            labelled.append([(c, t) for c, t in zip(cipher, text)])
        if self.laplace:
            self.model = self.model.train(labeled_sequences=labelled, estimator=laplace)
        else:
            self.model = self.model.train(labeled_sequences=labelled)
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
    # TODO Uncomment this line below before officially submit the code
    # parser.add_argument('cipher_folder', type=str, default='cipher1', choices=['cipher1', 'cipher2', 'cipher3'], help='Which cipher to use')
    config = parser.parse_args()

    # TODO Uncomment this line below before officially submit the code
    # task = config.cipher_folder
    task = 'cipher1'
    config.laplace = True
    config.lm = False

    # Modelling start below here
    dl = DataLoader(config.data_dir, task, config.lm)
    # Declare the tagger
    tagger = Tagger(dl, config.laplace)
    # Train the tagger
    tagger.train()
    # Test the tagger
    tagger.test()


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('Total time consumption: %fs' % (end_time - start_time))
