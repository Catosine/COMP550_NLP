import argparse
import time
import os
import os.path as osp
import logging
import shutil
import sys

class DataLoader():
    def __init__(self, data_dir, task):
        self.train_cipher = osp.join(data_dir, task, 'train_cipher.txt')
        self.train_plain = osp.join(data_dir, task, 'train_plain.txt')
        self.test_cipher = osp.join(data_dir, task, 'test_cipher.txt')
        self.test_plain = osp.join(data_dir, task, 'test_plain.txt')

    def get_train_corpus(self):
        logging.info('-'*20)
        logging.info('Start loading train cipher text')

        if not osp.exists(self.train_cipher):
            logging.info('INVALID TRAIN CIPHER PATH: {}'.format(self.train_cipher))

        corpus = list()
        plain = open(self.train_plain, 'r', errors='ignore')
        cipher = open(self.train_cipher, 'r', errors='ignore')

        for encoded, decoded in zip(cipher.readlines(), plain.readlines()):
            corpus.append((encoded, decoded))

        return corpus

    def get_test_corpus(self):
        logging.info('-'*20)
        logging.info('Start loading test cipher text')

        if not osp.exists(self.test_cipher):
            logging.info('INVALID TEST CIPHER PATH: {}'.format(self.test_cipher))

        corpus = list()
        plain = open(self.test_plain, 'r', errors='ignore')
        cipher = open(self.test_cipher, 'r', errors='ignore')

        for encoded, decoded in zip(cipher.readlines(), plain.readlines()):
            corpus.append((encoded, decoded))

        return corpus


def main():
    parser = argparse.ArgumentParser("COMP550_Assignment2_Question3")
    parser.add_argument('--data_dir', type=str, default='D:\\McGill\\19Fall\\COMP 550\\Project\\data\\a2data',
                        help='Path to data directory')
    parser.add_argument('--log_dir', type=str, default='experiments_logs', help='Directory to save the experiment log')
    parser.add_argument('--note', type=str, default='testShot', help='Notes for this experiment')
    parser.add_argument('-laplace', action='store_true', help='Use laplace smoothing')
    parser.add_argument('-lm', action='store_true', help='Improved plaintext modelling')
    config = parser.parse_args()

    task = sys.argv[-1]

    # Create log dir for experiments
    if not osp.exists(config.log_dir):
        os.mkdir(config.log_dir)
    laplace = 'laplace_smoothing' if config.laplace else 'no_smoothing'
    lm = 'improved_plaintext_modelling' if config.lm else 'no_extra_improvement'
    log_path = 'standardHMM-{}-{}-{}-{}'.format(config.note, laplace, lm, time.strftime("%Y%m%d-%H%M%S"))
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

    dl = DataLoader(config.data_dir, task)
    train_corpus = dl.get_train_corpus()

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    logging.info('Total time consumption: %ds', end_time - start_time)
