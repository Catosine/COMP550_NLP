# Package used for this project
import argparse
import logging
import os
import os.path as osp
import random
import shutil
import sys
import time
from collections import Counter

import matplotlib
import numpy as np
import sklearn.metrics as evaluator
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer as PS
from nltk.stem import WordNetLemmatizer as WNL
from nltk.tokenize import RegexpTokenizer as RT
from nltk.tokenize import word_tokenize as WT
from sklearn.dummy import DummyClassifier as DC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB as BNB
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.svm import LinearSVC as SVM
from tqdm import tqdm

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
import gc


class DataLoader():

    def __init__(self, data_dir, pos_path, neg_path):
        pos_path = osp.join(data_dir, pos_path)
        neg_path = osp.join(data_dir, neg_path)

        self.train_path = osp.join(data_dir, 'rt-polaritydata/train_data.txt')
        self.test_path = osp.join(data_dir, 'rt-polaritydata/test_data.txt')

        if not osp.exists(self.train_path) or not osp.exists(self.test_path):
            self.__prepare_dataset__(pos_path, neg_path)

    def get_trainset(self):

        data = list()
        label = list()

        logging.info('-' * 20)
        logging.info("Start loading train data at : %s", self.train_path)

        if not osp.exists(self.train_path):
            logging.info("Invalid Data path : %s", self.train_path)
            exit(0)

        with open(self.train_path, "r", errors='ignore') as f:
            lines = f.readlines()
            shuffled_idx = np.random.permutation(len(lines))
            for id in shuffled_idx:
                data.append(lines[id][:-2])
                label.append(int(lines[id][-2]))

        return data, label

    def get_testset(self):

        data = list()
        label = list()

        logging.info('-' * 20)
        logging.info("Start loading test data at : %s", self.test_path)

        if not osp.exists(self.test_path):
            logging.info("Invalid Data path : %s", self.test_path)
            exit(0)

        with open(self.test_path, "r", errors='ignore') as f:
            for d in f.readlines():
                data.append(d[:-2])
                label.append(int(d[-2]))

        return data, label

    def __load_raw__(self, path, label):
        logging.info("Staring loading data at : %s", path)

        if not osp.exists(path):
            logging.info("Invalid Data path : %s", path)
            exit(0)

        with open(path, "r", errors='ignore') as d:
            data = d.readlines()

        return data, [label] * len(data)

    def __prepare_dataset__(self, pos_path, neg_path, ratio=0.8):

        logging.info('-' * 20)
        logging.info("Start preparing train/test dataset")

        pos = self.__load_raw__(pos_path, 1)
        neg = self.__load_raw__(neg_path, 0)

        train_pos_data, test_pos_data, train_pos_label, test_pos_label = train_test_split(pos[0], pos[1],
                                                                                          train_size=ratio,
                                                                                          shuffle=True)
        train_neg_data, test_neg_data, train_neg_label, test_neg_label = train_test_split(neg[0], neg[1],
                                                                                          train_size=ratio,
                                                                                          shuffle=True)

        train_data = train_pos_data + train_neg_data
        train_label = train_pos_label + train_neg_label
        test_data = test_pos_data + test_neg_data
        test_label = test_pos_label + test_neg_label

        with open(self.train_path, 'w') as f:
            shuffled = list(zip(train_data, train_label))
            random.shuffle(shuffled)
            for data, label in shuffled:
                to_write = data[:-2] + " " + str(label)
                f.write('%s\n' % to_write)

        with open(self.test_path, 'w') as f:
            shuffled = list(zip(test_data, test_label))
            random.shuffle(shuffled)
            for data, label in shuffled:
                to_write = data[:-2] + " " + str(label)
                f.write('%s\n' % to_write)


class FeatureExtractor():

    def __init__(self, config):
        self.list_of_process = config.feature_extract
        self.model = config.model
        self.dictionary = None
        self.threshold = config.frequency_threshold
        self.glove_size = config.feature_size
        self.glove_dir = osp.join(config.data_dir, "glove.6B", "glove.6B." + str(config.feature_size) + "d.txt")

        for process in self.list_of_process:
            if process == 'regex_tokenize':
                self.tok = RT(r'\w+')
            elif process == 'lemmatize':
                self.lem = WNL()
            elif process == 'stopwords':
                self.sw = stopwords.words('english')
            elif 'stem' == process:
                self.stemmer = PS()

    def __clean_sentence__(self, sentence):
        for process in self.list_of_process:
            if process == 'tokenize':
                sentence = WT(sentence)
            elif process == 'regex_tokenize':
                sentence = self.tok.tokenize(sentence)
            elif process == 'lemmatize':
                sentence = [self.lem.lemmatize(word) for word in sentence]
            elif process == 'stopwords':
                sentence = [word for word in sentence if word not in self.sw]
            elif process == 'stem':
                sentence = [self.stemmer.stem(word) for word in sentence]

        return sentence

    def __clean_data__(self, data):
        size = len(data[0])
        for idx in range(size):
            data[0][idx] = self.__clean_sentence__(data[0][idx])

        return data

    def __extract_binary_feature__(self, data, dictionary):
        size = len(data[0])
        for idx in tqdm(range(size)):
            feature = list()
            for word in dictionary:
                sent = data[0][idx]
                freq = sent.count(word)
                if freq > self.threshold:
                    feature.append(1)
                else:
                    feature.append(0)
            data[0][idx] = feature

        return [np.array(data[0], dtype='uint8'), np.array(data[1], dtype='uint8')]

    def __extract_frequency_feature__(self, data, dictionary):
        size = len(data[0])
        for idx in tqdm(range(size)):
            feature = list()
            for word in dictionary:
                temp = data[0][idx].count(word)
                temp = temp if temp > self.threshold else 0
                feature.append(temp)
            data[0][idx] = feature

        return [np.array(data[0], dtype='uint16'), np.array(data[1], dtype='uint16')]

    def __extract_glove_vectors__(self, data):
        glove_dict = {}
        with open(self.glove_dir, 'r') as f:
            for line in f:
                splitLine = line.split()
                word = splitLine[0]
                embedding = np.array([float(val) for val in splitLine[1:]])
                glove_dict[word] = embedding

        size = len(data[0])
        for idx in tqdm(range(size)):
            feature = np.zeros(self.glove_size)
            for word in data[0][idx]:
                try:
                    feature += glove_dict[word]
                except:
                    feature += np.zeros(self.glove_size)
            data[0][idx] = feature

        data = [np.array(data[0]), np.array(data[1])]

        # Normalize data
        data[0] = data[0] - data[0].mean()
        data[0] = data[0] / data[0].std()

        return data

    def extract_feature(self, data):
        logging.info('-' * 20)
        logging.info('Start extracting features')
        # data cleaning
        data = self.__clean_data__(data)

        if self.dictionary is None:
            self.dictionary = self.count_frequency(data)

        # encoding feature
        if 'binary' in self.list_of_process:
            data = self.__extract_binary_feature__(data, self.dictionary.keys())
        elif 'frequency' in self.list_of_process:
            data = self.__extract_frequency_feature__(data, self.dictionary.keys())
        elif 'glove' in self.list_of_process:
            data = self.__extract_glove_vectors__(data)
        else:
            print(self.list_of_process)
            logging.info("No available feature encoding found")
            exit(0)

        return data

    def count_frequency(self, data):
        all_dic = Counter()
        for exp in data[0]:
            all_dic.update(exp)
        all_dic = {k: v for k, v in all_dic.items() if v > self.threshold}
        return all_dic


class Classifier():

    def __init__(self, data_loader, config):
        self.model = config.model
        self.data_loader = data_loader
        self.feature_extractor = FeatureExtractor(config)
        self.penalty = config.penalty
        self.c = config.C
        self.epoch = config.epoch

    def train(self):
        logging.info('-' * 20)
        logging.info('Start training the %s model', self.model)
        train_data = self.feature_extractor.extract_feature(self.data_loader.get_trainset())
        if self.model == 'BNB':
            # Bernoulli naive bayes
            self.classifier = BNB()
            self.classifier.fit(train_data[0], train_data[1])
        elif self.model == 'MNB':
            # Multinomial naive bayes
            self.classifier = MNB()
            self.classifier.fit(train_data[0], train_data[1])
        elif self.model == 'LR':
            # Logistic regression
            self.classifier = LR(penalty=self.penalty, C=self.c, max_iter=self.epoch, solver='liblinear')
            self.classifier.fit(train_data[0], train_data[1])
        elif self.model == 'SVM':
            # Support vector machine
            self.classifier = SVM(penalty=self.penalty, C=self.c, max_iter=self.epoch)
            self.classifier.fit(train_data[0], train_data[1])
        elif self.model == 'R':
            # RandomGuess
            self.classifier = DC(strategy='stratified')
            self.classifier.fit(train_data[0], train_data[1])
        else:
            logging.info('Unsupported model : %s', self.model)
            exit(0)

        del train_data
        gc.collect()

    def evaluate(self):
        logging.info('-' * 20)
        logging.info('Start evaluating the %s model', self.model)
        test_data = self.data_loader.get_testset()
        test_data = self.feature_extractor.extract_feature(test_data)
        predictions = self.classifier.predict(test_data[0])
        tn, fp, fn, tp = evaluator.confusion_matrix(test_data[1], predictions).ravel()
        pos_precision = tp / (tp + fp)
        neg_precision = tn / (tn + fn)
        macro_avg_precision = (pos_precision + neg_precision) / 2.0
        micro_avg_precision = (tp + tn) / (tp + fp + tn + fn)
        pos_recall = tp / (tp + fn)
        neg_recall = tn / (tn + fp)
        macro_avg_recall = (pos_recall + neg_recall) / 2.0
        micro_avg_recall = (tp + tn) / (tp + fp + tn + fn)
        return {
            'pos_precision': pos_precision, 'neg_precision': neg_precision,
            'pos_recall': pos_recall, 'neg_recall': neg_recall,
            'macro_avg_precision': macro_avg_precision,
            'micro_avg_precision': micro_avg_precision,
            'macro_avg_recall': macro_avg_recall,
            'micro_avg_recall': micro_avg_recall,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }


def main():
    # Parameters
    parser = argparse.ArgumentParser("COMP550_Assignment1_Question3")
    parser.add_argument('--model', type=str, default='MNB', choices=['BNB', 'MNB', 'LR', 'SVM', 'R'],
                        help='Model used for this experiment')
    parser.add_argument('--note', type=str, default='Baseline', help='Note for this experiment')
    parser.add_argument('--feature_extract', action='append', default=['regex_tokenize', 'glove'],
                        help='Feature extraction processdures')
    parser.add_argument('--feature_size', type=int, default=50, choices=[50, 100, 200, 300], help='Only used for GloVe')
    parser.add_argument('--frequency_threshold', type=int, default=0,
                        help='Threshold for frequency count, all word present less than threshold will be counted as 0')
    parser.add_argument('--data_dir', type=str, default='D:\\McGill\\19Fall\\COMP 550\\Project\\data',
                        help='Path to data directory')
    parser.add_argument('--log_dir', type=str, default='experiments_logs', help='Directory to save the experiment log')
    parser.add_argument('--pos_data', type=str, default='rt-polaritydata\\rt-polarity.pos',
                        help='Path to positive data')
    parser.add_argument('--neg_data', type=str, default='rt-polaritydata\\rt-polarity.neg',
                        help='Path to negative data')
    parser.add_argument('--penalty', type=str, default='l2', choices=['l1', 'l2', 'elasticnet', 'none'], help='Penalty')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Inverse of regularization strength, smaller means stronger')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Maximum umber of iteration for gradient-based algorithm to converge')
    config = parser.parse_args()

    # Create logs for experiments
    if not osp.exists(config.log_dir):
        os.mkdir(config.log_dir)
    if 'glove' in config.feature_extract:
        config.feature_extract[-1] = config.feature_extract[-1] + '_' + str(config.feature_size)
    if config.model in ['LR', 'SVM']:
        log_path = '{}-{}-Threshold_{}-Epoch_{}-Penalty_{}-C_{}-{}-{}'.format(
            config.model,
            '_'.join(config.feature_extract),
            config.frequency_threshold,
            config.epoch, config.penalty,
            config.C, config.note,
            time.strftime('%Y%m%d-%H%M%S')
        )
    elif 'NB' in config.model:
        log_path = '{}-{}-Threshold_{}-{}-{}'.format(
            config.model,
            '_'.join(config.feature_extract),
            config.frequency_threshold,
            config.note,
            time.strftime("%Y%m%d-%H%M%S")
        )
    elif config.model == 'R':
        log_path = '{}-{}-Threshold_{}-{}-{}'.format(
            config.model,
            '_'.join(config.feature_extract),
            config.frequency_threshold,
            config.note,
            time.strftime("%Y%m%d-%H%M%S")
        )
    else:
        logging.info("Invalid model: ", config.model)
        exit(0)
    if 'glove' in config.feature_extract[-1]:
        config.feature_extract[-1] = 'glove'
    config.log_dir = osp.join(os.getcwd(), config.log_dir, log_path)
    if not osp.exists(config.log_dir):
        os.mkdir(config.log_dir)
        dst_file = os.path.join(config.log_dir, os.path.basename('a1q3.py'))
        shutil.copyfile('a1q3.py', dst_file)
    print("Experiment log saved at : {}".format(config.log_dir))

    # setup logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(osp.join(config.log_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info('Config = %s', config)

    # Load data
    data_loader = DataLoader(config.data_dir, config.pos_data, config.neg_data)

    # Setup the model
    model = Classifier(data_loader, config)

    # Train the model
    model.train()

    # Evaluate the model
    result = model.evaluate()

    logging.info('-' * 20)
    logging.info('Evaluation for %s model', config.model)
    logging.info('Accuracy = %f', result['micro_avg_recall'])
    logging.info('-' * 20)
    logging.info('Precision of positive data = %f', result['pos_precision'])
    logging.info('Precision of negative data = %f', result['neg_precision'])
    logging.info('Macro average precision = %f', result['macro_avg_precision'])
    logging.info('Micro average precision = %f', result['micro_avg_precision'])
    logging.info('Recall of positive data = %f', result['pos_recall'])
    logging.info('Recall of negative data = %f', result['neg_recall'])
    logging.info('Macro average recall = %f', result['macro_avg_recall'])
    logging.info('Micro average recall = %f', result['micro_avg_recall'])
    logging.info('True Positive = %d', result['tp'])
    logging.info('False Positive = %d', result['fp'])
    logging.info('True Negative = %d', result['tn'])
    logging.info('False Negative = %d', result['fn'])

    cm = {
        'Label': ['Positive', 'Negative', 'Positive', 'Negative'],
        'Prediction': ['Positive', 'Positive', 'Negative', 'Negative'],
        'Frequency': [result['tp'], result['fp'], result['fn'], result['tn']]
    }

    cm = DataFrame(cm)
    cm = cm.pivot('Label', 'Prediction', 'Frequency')
    cm = cm.reindex(index=cm.index[::-1])
    cm = cm[cm.columns[::-1]]

    ax = sns.heatmap(cm, linewidth=1, annot=True, fmt='d')

    plt.savefig(osp.join(config.log_dir, 'confusion_matrix.png'))


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    logging.info('Total time consumption: %ds', end_time - start_time)
