import argparse
import time
import os
import os.path as osp
import logging
import shutil
import sys
import numpy
import nltk
import nltk.tag.hmm as hmm
import sklearn
from tqdm import tqdm
import string
import gc

class DataLoader():

    def __init__(self, data_dir, task, lm=False):
        self.train_cipher = osp.join(data_dir, task, 'train_cipher.txt')
        self.train_plain = osp.join(data_dir, task, 'train_plain.txt')
        self.test_cipher = osp.join(data_dir, task, 'test_cipher.txt')
        self.test_plain = osp.join(data_dir, task, 'test_plain.txt')
        self.lm = lm

    def __get_corpus__(self, cipher_path, plain_path):
        corpus = list()
        p = open(plain_path, 'r', errors='ignore')
        c = open(cipher_path, 'r', errors='ignore')

        for encoded, decoded in tqdm(zip(c.readlines(), p.readlines())):
            encoded = list(encoded)
            decoded = list(decoded)

            if self.lm:
                to_keep = list(string.ascii_lowercase) + [' ', ',', '.']
                for j, (char_e, char_d) in enumerate(zip(encoded, decoded)):
                    if char_e not in to_keep or char_d not in to_keep:
                        del encoded[j]
                        del decoded[j]
                        gc.collect()

            corpus.append(list(zip(encoded, decoded)))

        return corpus

    def get_train_corpus(self):
        logging.info('-'*20)
        logging.info('Start loading train corpus')

        if not osp.exists(self.train_cipher):
            logging.info('INVALID TRAIN CIPHER PATH: {}'.format(self.train_cipher))
            exit(0)

        if not osp.exists(self.train_plain):
            logging.info('INVALID TRAIN PLAIN PATH: {}'.format(self.train_plain))
            exit(0)

        return self.__get_corpus__(self.train_cipher, self.train_plain)

    def __get_test_corpus__(self):
        logging.info('-'*20)
        logging.info('Start loading test corpus')

        if not osp.exists(self.test_cipher):
            logging.info('INVALID TEST CIPHER PATH: {}'.format(self.test_cipher))
            exit(0)

        if not osp.exists(self.train_plain):
            logging.info('INVALID TEST PLAIN PATH: {}'.format(self.test_plain))
            exit(0)

        return self.__get_corpus__(self.test_cipher, self.test_plain)

class Tagger():

    def __init__(self, train_feature_extractor, test_feature_extractor):
        self.train_feature_extractor = train_feature_extractor
        self.test_feature_extractor = test_feature_extractor

def main():
    parser = argparse.ArgumentParser("COMP550_Assignment2_Question3")
    parser.add_argument('--data_dir', type=str, default='D:\\McGill\\19Fall\\COMP 550\\Project\\data\\a2data',
                        help='Path to data directory')
    parser.add_argument('--log_dir', type=str, default='experiments_logs', help='Directory to save the experiment log')
    parser.add_argument('--note', type=str, default='testShot', help='Notes for this experiment')
    parser.add_argument('-laplace', action='store_true', help='Use laplace smoothing')
    parser.add_argument('-lm', action='store_true', help='Improved plaintext modelling')
    #TODO Uncomment this line below before officially submit the code
    #parser.add_argument('cipher_folder', type=str, default='cipher1', choices=['cipher1', 'cipher2', 'cipher3'], help='Which cipher to use')
    config = parser.parse_args()

    #TODO Uncomment this line below before officially submit the code
    #task = config.cipher_folder
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

    train_corpus = dl.get_train_corpus()
    print(train_corpus[0][:10])

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    logging.info('Total time consumption: %ds', end_time - start_time)
