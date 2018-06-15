# -*- coding:utf-8 -*-

import logging
from sentence import Sentence

LOG_LEVEL = logging.INFO


class Data():
    '''
    Read data from a file
    '''

    def __init__(self, file_name, log_level=LOG_LEVEL):
        '''
        Constructor
        '''
        self.logger = None
        self.init_logging(log_level)
        # self.logger.info('Data class begin')

        self.sentences = []

        self.read_graphs(file_name)

    def read_graphs(self, file_name):

        with open(file_name, 'r') as file:
            lines = file.readlines()
            words = []
            for line in lines:

                if line.strip():
                    line_splitted = line.rstrip().split('\t')
                    words.append(line_splitted)
                else:
                    sentence = Sentence(words)
                    self.sentences.append(sentence)
                    words = []

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

    deps = Data(data_dir + 'en-ud-dev.conllu')