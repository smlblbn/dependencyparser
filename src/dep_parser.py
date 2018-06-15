# -*- coding:utf-8 -*-

import logging
import pickle
import os
import numpy as np
from nltk import DependencyGraph, DependencyEvaluator

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, accuracy_score

from data import Data
from sentence import Sentence
from mst import mst

LOG_LEVEL = logging.INFO


class Parser(object):
    '''
    docstring for Parser
    '''

    def __init__(self, params=None):
        '''
        initialize the parser models (and the vectorizers if necessary)
        params is a dict of arguments that can be passed to models
        '''
        self.logger = None
        self.init_logging(LOG_LEVEL)
        # self.logger.info('Parser class begin')

        self.model_head = Pipeline([
            ('vectorizer', DictVectorizer()),
            ('classifier', SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, \
                                         max_iter=100, tol=None, shuffle=True, verbose=0, epsilon=0.1, n_jobs=4, \
                                         random_state=463, learning_rate='optimal', eta0=0.0, power_t=0.5, \
                                         class_weight='balanced', warm_start=True, average=False, n_iter=None))
            ])

        self.model_relation = Pipeline([
            ('vectorizer', DictVectorizer()),
            ('classifier', SGDClassifier(loss='log', penalty='l2', alpha=0.001, l1_ratio=0.15, fit_intercept=True, \
                                         max_iter=100, tol=None, shuffle=True, verbose=0, epsilon=0.1, n_jobs=4, \
                                         random_state=463, learning_rate='optimal', eta0=0.0, power_t=0.5, \
                                         class_weight=None, warm_start=True, average=False, n_iter=None))
            ])


    def train(self, file_name):
        '''
        extract the features and train the models
        file_name shows the path to the training file with task specific format
        '''

        data = Data(file_name, LOG_LEVEL)
        for s in data.sentences:
            s.extract_edge_features(type='train')

        feat = []
        label = []
        for s in data.sentences:
            feat = feat + s.edge_features
            label = label + s.edge_labels

        self.model_head.fit(feat, label)

        y_pred = self.model_head.predict(feat)

        dec = self.model_head.decision_function(feat)

        indeces = np.argwhere(np.array(label) > 0)
        label_nonzero = np.take(label, indeces[:, 0])
        y_pred_nonzero = np.take(y_pred, indeces[:, 0])

        print('train len: ', len(y_pred))
        print('train acc: ', accuracy_score(label, y_pred))
        print('train f1_socre: ', f1_score(label, y_pred))
        print('train precision', precision_score(label, y_pred))
        print('train recall', recall_score(label, y_pred))
        print('train acc non zero: ', accuracy_score(label_nonzero, y_pred_nonzero))

        min_values = np.amin(dec, axis=0)
        max_values = np.amax(dec, axis=0)
        score_pred = (dec - min_values) / (max_values - min_values)

        index = 0
        head_pred = []
        head_label = []
        for s in data.sentences:

            scores = np.full([len(s.tokens) + 1, len(s.tokens) + 1], -1.)
            size_tokens = len(s.tokens)

            for it1 in range(size_tokens):
                for it2 in range(size_tokens):
                    if it1 != it2:
                        scores[it1 + 1][it2 + 1] = score_pred[index]
                        index = index + 1

            for it3 in range(size_tokens):
                scores[0][it3 + 1] = score_pred[index]
                index = index + 1

            heads = mst(scores.T)
            s.index_preds = heads[1:].tolist()

            head_pred = head_pred + s.index_preds
            head_label = head_label + s.index_labels

        print("train head acc: ", accuracy_score(head_label, head_pred))

        for s in data.sentences:
            s.extract_relation_features()

        feat = []
        label = []
        for s in data.sentences:
            feat = feat + s.relation_features
            label = label + s.relation_labels

        self.model_relation.fit(feat, label)

        y_pred = self.model_relation.predict(feat)
        y_pred = y_pred.tolist()

        idx_sum = 0
        for s in data.sentences:
            idx = len(s.tokens)
            s.relation_preds = y_pred[idx_sum: idx_sum + idx]
            idx_sum = idx_sum + idx

        print('train relation acc: ', accuracy_score(label, y_pred))
        print()

    def test(self, in_file_name, out_file_name=None):
        '''
        test the model, extract features and parse the sentences
        file_name shows the path to the test file with task specific format
        return uas and las
        '''

        data = Data(in_file_name, LOG_LEVEL)
        for s in data.sentences:
            s.extract_edge_features(type='train')

        feat = []
        label = []
        size = 0
        for s in data.sentences:
            feat = feat + s.edge_features
            label = label + s.edge_labels
            size = size + len(s.tokens)*len(s.tokens)

        y_pred = self.model_head.predict(feat)

        dec = self.model_head.decision_function(feat)

        indeces = np.argwhere(np.array(label) > 0)
        label_nonzero = np.take(label, indeces[:, 0])
        y_pred_nonzero = np.take(y_pred, indeces[:, 0])

        print('test token len', size)
        print('test len: ', len(y_pred))
        print('test acc: ', accuracy_score(label, y_pred))
        print('test f1_socre: ', f1_score(label, y_pred))
        print('test precision', precision_score(label, y_pred))
        print('test recall', recall_score(label, y_pred))

        print('test acc non zero: ', accuracy_score(label_nonzero, y_pred_nonzero))

        min_values = np.amin(dec, axis=0)
        max_values = np.amax(dec, axis=0)
        score_pred = (dec - min_values) / (max_values - min_values)

        index = 0
        head_pred = []
        head_label = []
        for s in data.sentences:
            scores = np.full([len(s.tokens) + 1, len(s.tokens) + 1], -1.)
            size_tokens = len(s.tokens)
            for it1 in range(size_tokens):
                for it2 in range(size_tokens):
                    if it1 != it2:
                        scores[it1 + 1][it2 + 1] = score_pred[index]
                        index = index + 1

            for it3 in range(size_tokens):
                scores[0][it3 + 1] = score_pred[index]
                index = index + 1

            heads = mst(scores.T)
            s.index_preds = heads[1:].tolist()

            head_pred = head_pred + s.index_preds
            head_label = head_label + s.index_labels

        print("test head acc: ", accuracy_score(head_label, head_pred))

        for s in data.sentences:
            s.extract_relation_features()

        feat = []
        label = []
        for s in data.sentences:
            feat = feat + s.relation_features
            label = label + s.relation_labels

        y_pred = self.model_relation.predict(feat)
        y_pred = y_pred.tolist()

        idx_sum = 0
        for s in data.sentences:
            idx = len(s.tokens)
            s.relation_preds = y_pred[idx_sum : idx_sum + idx]
            idx_sum = idx_sum + idx

        print('test relation acc: ', accuracy_score(label, y_pred))
        print()

        str_parsed = []
        str_golden = []
        for s in data.sentences:
            _str_parsed = ''
            _str_golden = ''
            for i in range(len(s.tokens)):
                _str_parsed = _str_parsed + s.tokens[i] + '\t' + s.pos_tags[i] + '\t' + str(s.index_preds[i]) + '\t' + s.relation_preds[i] + '\n'
                _str_golden = _str_golden + s.tokens[i] + '\t' + s.pos_tags[i] + '\t' + str(s.index_labels[i]) + '\t' + s.relation_labels[i] + '\n'
            str_parsed.append(DependencyGraph(_str_parsed))
            str_golden.append(DependencyGraph(_str_golden))

        de = DependencyEvaluator(str_parsed, str_golden)
        uas, las = de.eval()
        return uas, las

    def parse_sentence(self, sentence):
        '''
        tag a single tokenized and pos tagged sentence
        sentence: [[tkn1, tag1], [tkn2, tag2], ...]
        return the parsed with the same data format
            4 tab separated fields per line
        '''
        sent = Sentence(sentence)
        sent.extract_edge_features(type='test')

        dec = self.model_head.decision_function(sent.edge_features)

        min_values = np.amin(dec, axis=0)
        max_values = np.amax(dec, axis=0)
        score_pred = (dec - min_values)  / (max_values - min_values)

        index = 0
        scores = np.full([len(sent.tokens) + 1, len(sent.tokens) + 1], -1.)
        size_tokens = len(sent.tokens)

        for it1 in range(size_tokens):
            for it2 in range(size_tokens):
                if it1 != it2:
                    scores[it1 + 1][it2 + 1] = score_pred[index]
                    index = index + 1

        for it3 in range(size_tokens):
            scores[0][it3 + 1] = score_pred[index]
            index = index + 1

        heads = mst(scores.T)
        sent.index_preds = heads[1:]

        sent.extract_relation_features()
        y_pred = self.model_relation.predict(sent.relation_features)
        sent.relation_preds = y_pred.tolist()

        _str = ''
        for i in range(len(sent.tokens)):
            _str = _str + sent.tokens[i] + '\t' + sent.pos_tags[i] + '\t' + str(sent.index_preds[i]) + '\t' + sent.relation_preds[i] + '\n'

        return _str

    def parse(self, sentences):
        '''
        tag a list of sentences
        return a list of parsed sentences with same data format
        '''
        str_list = []
        for s in sentences:
            str_list = str_list + [self.parse_sentence(s)]

    def _mst(self, scores):
        '''
        '''

    def save(self, file_name):
        '''
        save the trained models to file_name
        '''
        with open(file_name, 'wb') as file:
            pickle.dump((self.model_head, self.model_relation), file, pickle.HIGHEST_PROTOCOL)

        # self.logger.info('Model saved, file name is ' + file_name)


    def load(self, file_name):
        '''
        load the trained models from file_name
        '''
        with open(file_name, 'rb') as file:
            self.model_head, self.model_relation = pickle.load(file)

        # self.logger.info('Model load, from ' + file_name)


    def init_logging(self, log_level):
        '''
        logging config and init
        '''
        if not self.logger:
            logging.basicConfig(
                format='%(asctime)s-|%(name)20s:%(funcName)12s|'
                       '-%(levelname)8s-> %(message)s')
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.setLevel(log_level)


if __name__ == '__main__':

    # data_dir = '../data/'
    # data_dir = '../data_small/'
    data_dir = '../data_mini/'

    parser = Parser()
    parser.train(data_dir + 'en-ud-train.conllu')

    uas, las = parser.test(data_dir + 'en-ud-dev.conllu')
    print()
    print('dev set results: ', uas, las)

    # if you use data directory comment out test set  because there is no test set
    # uas, las = parser.test(data_dir + 'en-ud-test.conllu')
    # print('test set results: ', uas, las)

    directory = '../models'
    if not os.path.exists(directory):
        os.makedirs(directory)

    parser.save('../models/dep_parser.pickle')

    parser = Parser()
    parser.load('../models/dep_parser.pickle')

    sentence = [
        ['From', 'IN'],
        ['the', 'DT'],
        ['AP', 'NNP'],
        ['comes', 'VBZ'],
        ['this', 'DT'],
        ['story', 'NN'],
        [':', ':']
    ]

    print()
    print(parser.parse_sentence(sentence))
