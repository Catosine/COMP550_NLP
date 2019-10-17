import argparse
import time
import os
import os.path as osp
import logging
import shutil
import sys
import nltk.tag.hmm as hmm
from sklearn.metrics import accuracy_score as evaluate
from collections import Counter
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
        logging.info('-' * 20)
        logging.info('Start loading train corpus')

        if not osp.exists(self.train_cipher):
            logging.info('INVALID TRAIN CIPHER PATH: {}'.format(self.train_cipher))
            exit(0)

        if not osp.exists(self.train_plain):
            logging.info('INVALID TRAIN PLAIN PATH: {}'.format(self.train_plain))
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
        logging.info('-' * 20)
        logging.info('Start loading test corpus')

        if not osp.exists(self.test_cipher):
            logging.info('INVALID TEST CIPHER PATH: {}'.format(self.test_cipher))
            exit(0)

        if not osp.exists(self.test_plain):
            logging.info('INVALID TEST PLAIN PATH: {}'.format(self.test_plain))
            exit(0)

        plain = list()
        cipher = list()
        p = open(self.test_cipher, 'r', errors='ignore')
        c = open(self.test_plain, 'r', errors='ignore')

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

    def get_states_and_symbols(self):
        if self.train_dictionary is None:
            if self.train_corpus is None:
                self.__prepare_train_corpus__()
            cipher_c = Counter()
            text_c = Counter()
            for c, t in zip(self.train_corpus[0], self.train_corpus[1]):
                cipher_c.update(c)
                text_c.update(c)
            self.train_dictionary = [cipher_c, text_c]

        return self.train_dictionary

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
        self.dictionary = self.data_loader.get_states_and_symbols()
        self.model = hmm.HiddenMarkovModelTrainer(states=list(self.dictionary[0].keys()), symbols=list(self.dictionary[1].keys()))

    def train(self):
        logging.info('-' * 20)
        logging.info("Start training standard HMM tagger")
        train_corpus = self.data_loader.get_train_corpus()
        labelled = list()
        for cipher, text in zip(train_corpus[0], train_corpus[1]):
            labelled.append([(c, t) for c, t in zip(cipher, text)])
        if self.laplace:
            raise NotImplementedError
        else:
            self.model = self.model.train(labeled_sequences=labelled)
        train_predict = DataLoader.flat(self._tag(train_corpus[0]))
        ground_truth = DataLoader.flat(train_corpus[1])
        acc = evaluate(ground_truth, train_predict)
        logging.info('Training accuracy : %f', acc)

    def test(self):
        logging.info('-' * 20)
        logging.info('Start testing the trained HMM tagger')
        test_corpus = self.data_loader.get_test_corpus()
        test_predict = DataLoader.flat(self._tag(test_corpus[0]))
        ground_truth = DataLoader.flat(test_corpus[1])
        acc = evaluate(ground_truth, test_predict)
        logging.info('Testing accuracy : %f', acc)

    def _tag(self, corpus):
        pred = list()
        for sent in corpus:
            decode = self.model.tag(sent)
            pred.append([d for (_, d) in decode])
        return pred


def main():
    parser = argparse.ArgumentParser("COMP550_Assignment2_Question3")
    parser.add_argument('--data_dir', type=str, default='D:\\McGill\\19Fall\\COMP 550\\Project\\data\\a2data',
                        help='Path to data directory')
    parser.add_argument('--log_dir', type=str, default='experiments_logs', help='Directory to save the experiment log')
    parser.add_argument('--note', type=str, default='testShot', help='Notes for this experiment')
    parser.add_argument('-laplace', action='store_true', help='Use laplace smoothing')
    parser.add_argument('-lm', action='store_true', help='Improved plaintext modelling')
    # TODO Uncomment this line below before officially submit the code
    # parser.add_argument('cipher_folder', type=str, default='cipher1', choices=['cipher1', 'cipher2', 'cipher3'], help='Which cipher to use')
    config = parser.parse_args()

    # TODO Uncomment this line below before officially submit the code
    # task = config.cipher_folder
    task = 'cipher1'

    # Create log dir for experiments
    if not osp.exists(config.log_dir):
        os.mkdir(config.log_dir)
    laplace = 'laplace_smoothing' if config.laplace else 'no_smoothing'
    lm = 'improved_plaintext_modelling' if config.lm else 'no_extra_improvement'
    log_path = 'standardHMM-{}-{}-{}-{}-{}'.format(task, config.note, laplace, lm, time.strftime("%Y%m%d-%H%M%S"))
    config.log_dir = osp.join(os.getcwd(), config.log_dir, log_path)
    if not osp.exists(config.log_dir):
        os.mkdir(config.log_dir)
        dst_file = os.path.join(config.log_dir, os.path.basename('decipher.py'))
        shutil.copyfile('decipher.py', dst_file)
    print("Experiment log saved at : {}".format(config.log_dir))

    # Setup logging system
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(osp.join(config.log_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info('Config = %s', config)

    # Modelling start below here
    dl = DataLoader(config.data_dir, task)
    # Declare the tagger
    tagger = Tagger(dl)
    # Train the tagger
    tagger.train()
    # Test the tagger
    tagger.test()



if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    logging.info('Total time consumption: %ds', end_time - start_time)
