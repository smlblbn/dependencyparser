# -*- coding:utf-8 -*-

import logging

LOG_LEVEL = logging.INFO


class Sentence():
    '''
    Read a single sentence
    '''

    def create_relation_features(self, pair):
        return {
            'distance': abs(pair[0][0] - pair[1][0]),
            'is_left': pair[0][0] - pair[1][0] < 0,
            'is_right': pair[0][0] - pair[1][0] > 0,
            'is_left_next': pair[0][0] - pair[1][0] == -1,
            'is_right_next': pair[0][0] - pair[1][0] == 1,
            'is_root': pair[0][0] == 0,
            'index': pair[1][0],

            #'is_capitalized': pair[0][1][0].upper() == pair[0][1][0],
            #'is_all_caps': pair[0][1].upper() == pair[0][1],
            #'is_all_lower': pair[0][1].lower() == pair[0][1],
            #'capitals_inside': pair[0][1][1:].lower() != pair[0][1][1:],

            'is_first': pair[0][0] - 1 == 0,
            'is_last': pair[0][0] == len(self.tokens),
            'is_numeric': pair[0][1].isdigit(),

            'prefix-1': pair[0][1][0],
            'prefix-2': pair[0][1][:2],
            'prefix-3': pair[0][1][:3],
            'suffix-1': pair[0][1][-1],
            'suffix-2': pair[0][1][-2:],
            'suffix-3': pair[0][1][-3:],

            'word': pair[0][1],
            'pos_tag': '' if pair[0][0] == 0 else self.pos_tags[pair[0][0] - 1],

            'prev_word-1': '' if pair[0][0]-1 <= 0 else self.tokens[pair[0][0] - 2],
            'prev_word-2': '' if pair[0][0]-1 <= 1 else self.tokens[pair[0][0] - 3],
            'prev_word-3': '' if pair[0][0]-1 <= 2 else self.tokens[pair[0][0] - 4],
            'next_word-1': '' if pair[0][0]-1 >= len(self.tokens) - 1 else self.tokens[pair[0][0]],
            'next_word-2': '' if pair[0][0]-1 >= len(self.tokens) - 2 else self.tokens[pair[0][0] + 1],
            'next_word-3': '' if pair[0][0]-1 >= len(self.tokens) - 3 else self.tokens[pair[0][0] + 2],

            'prev_word-1_pos_tag': '' if pair[0][0] - 1 <= 0 else self.pos_tags[pair[0][0] - 2],
            'prev_word-2_pos_tag': '' if pair[0][0] - 1 <= 1 else self.pos_tags[pair[0][0] - 3],
            'prev_word-3_pos_tag': '' if pair[0][0] - 1 <= 2 else self.pos_tags[pair[0][0] - 4],
            'next_word-1_pos_tag': '' if pair[0][0] - 1 >= len(self.pos_tags) - 1 else self.pos_tags[pair[0][0]],
            'next_word-2_pos_tag': '' if pair[0][0] - 1 >= len(self.pos_tags) - 2 else self.pos_tags[pair[0][0] + 1],
            'next_word-3_pos_tag': '' if pair[0][0] - 1 >= len(self.pos_tags) - 3 else self.pos_tags[pair[0][0] + 2],

            #'is_capitalized_dependent': pair[1][1][0].upper() == pair[1][1][0],
            #'is_all_caps_dependent': pair[1][1].upper() == pair[1][1],
            #'is_all_lower_dependent': pair[1][1].lower() == pair[1][1],
            #'capitals_inside_dependent': pair[1][1][1:].lower() != pair[1][1][1:],

            'is_first_dependent': pair[1][0] - 1 == 0,
            'is_last_dependent': pair[1][0] == len(self.tokens),
            'is_numeric_dependent': pair[1][1].isdigit(),

            'prefix-1_dependent': pair[1][1][0],
            'prefix-2_dependent': pair[1][1][:2],
            'prefix-3_dependent': pair[1][1][:3],
            'suffix-1_dependent': pair[1][1][-1],
            'suffix-2_dependent': pair[1][1][-2:],
            'suffix-3_dependent': pair[1][1][-3:],

            'word_dependent': pair[1][1],
            'pos_tag_dependent': self.pos_tags[pair[1][0] -1],

            'prev_word-1_dependent': '' if pair[1][0] - 1 <= 0 else self.tokens[pair[1][0] - 2],
            'prev_word-2_dependent': '' if pair[1][0] - 1 <= 1 else self.tokens[pair[1][0] - 3],
            'prev_word-3_dependent': '' if pair[1][0] - 1 <= 2 else self.tokens[pair[1][0] - 4],
            'next_word-1_dependent': '' if pair[1][0] - 1 >= len(self.tokens) - 1 else self.tokens[pair[1][0]],
            'next_word-2_dependent': '' if pair[1][0] - 1 >= len(self.tokens) - 2 else self.tokens[pair[1][0] + 1],
            'next_word-3_dependent': '' if pair[1][0] - 1 >= len(self.tokens) - 3 else self.tokens[pair[1][0] + 2],

            'prev_word-1_dependent_pos_tag': '' if pair[1][0] - 1 <= 0 else self.pos_tags[pair[1][0] - 2],
            'prev_word-2_dependent_pos_tag': '' if pair[1][0] - 1 <= 1 else self.pos_tags[pair[1][0] - 3],
            'prev_word-3_dependent_pos_tag': '' if pair[1][0] - 1 <= 2 else self.pos_tags[pair[1][0] - 4],
            'next_word-1_dependent_pos_tag': '' if pair[1][0] - 1 >= len(self.pos_tags) - 1 else self.pos_tags[pair[1][0]],
            'next_word-2_dependent_pos_tag': '' if pair[1][0] - 1 >= len(self.pos_tags) - 2 else self.pos_tags[pair[1][0] + 1],
            'next_word-3_dependent_pos_tag': '' if pair[1][0] - 1 >= len(self.pos_tags) - 3 else self.pos_tags[pair[1][0] + 2]
        }

    def create_edge_features(self, pair):
        return {
            'distance': abs(pair[0][0] - pair[1][0]),
            'is_left': pair[0][0] - pair[1][0] < 0,
            'is_right': pair[0][0] - pair[1][0] > 0,
            'is_left_next': pair[0][0] - pair[1][0] == -1,
            'is_right_next': pair[0][0] - pair[1][0] == 1,
            'is_root': pair[0][0] == 0,

            #'is_capitalized': pair[0][1][0].upper() == pair[0][1][0],
            #'is_all_caps': pair[0][1].upper() == pair[0][1],
            #'is_all_lower': pair[0][1].lower() == pair[0][1],
            #'capitals_inside': pair[0][1][1:].lower() != pair[0][1][1:],

            'is_first': pair[0][0] - 1 == 0,
            'is_last': pair[0][0] == len(self.tokens),
            'is_numeric': pair[0][1].isdigit(),

            'prefix-1': pair[0][1][0],
            'prefix-2': pair[0][1][:2],
            'prefix-3': pair[0][1][:3],
            'suffix-1': pair[0][1][-1],
            'suffix-2': pair[0][1][-2:],
            'suffix-3': pair[0][1][-3:],

            'word': pair[0][1],
            'pos_tag': '' if pair[0][0] == 0 else self.pos_tags[pair[0][0] - 1],

            'prev_word-1': '' if pair[0][0]-1 <= 0 else self.tokens[pair[0][0] - 2],
            'prev_word-2': '' if pair[0][0]-1 <= 1 else self.tokens[pair[0][0] - 3],
            'prev_word-3': '' if pair[0][0]-1 <= 2 else self.tokens[pair[0][0] - 4],
            'next_word-1': '' if pair[0][0]-1 >= len(self.tokens) - 1 else self.tokens[pair[0][0]],
            'next_word-2': '' if pair[0][0]-1 >= len(self.tokens) - 2 else self.tokens[pair[0][0] + 1],
            'next_word-3': '' if pair[0][0]-1 >= len(self.tokens) - 3 else self.tokens[pair[0][0] + 2],

            'prev_word-1_pos_tag': '' if pair[0][0] - 1 <= 0 else self.pos_tags[pair[0][0] - 2],
            'prev_word-2_pos_tag': '' if pair[0][0] - 1 <= 1 else self.pos_tags[pair[0][0] - 3],
            'prev_word-3_pos_tag': '' if pair[0][0] - 1 <= 2 else self.pos_tags[pair[0][0] - 4],
            'next_word-1_pos_tag': '' if pair[0][0] - 1 >= len(self.pos_tags) - 1 else self.pos_tags[pair[0][0]],
            'next_word-2_pos_tag': '' if pair[0][0] - 1 >= len(self.pos_tags) - 2 else self.pos_tags[pair[0][0] + 1],
            'next_word-3_pos_tag': '' if pair[0][0] - 1 >= len(self.pos_tags) - 3 else self.pos_tags[pair[0][0] + 2],

            #'is_capitalized_dependent': pair[1][1][0].upper() == pair[1][1][0],
            #'is_all_caps_dependent': pair[1][1].upper() == pair[1][1],
            #'is_all_lower_dependent': pair[1][1].lower() == pair[1][1],
            #'capitals_inside_dependent': pair[1][1][1:].lower() != pair[1][1][1:],

            'is_first_dependent': pair[1][0] - 1 == 0,
            'is_last_dependent': pair[1][0] == len(self.tokens),
            'is_numeric_dependent': pair[1][1].isdigit(),

            'prefix-1_dependent': pair[1][1][0],
            'prefix-2_dependent': pair[1][1][:2],
            'prefix-3_dependent': pair[1][1][:3],
            'suffix-1_dependent': pair[1][1][-1],
            'suffix-2_dependent': pair[1][1][-2:],
            'suffix-3_dependent': pair[1][1][-3:],

            'word_dependent': pair[1][1],
            'pos_tag_dependent': self.pos_tags[pair[1][0] -1],

            'prev_word-1_dependent': '' if pair[1][0] - 1 <= 0 else self.tokens[pair[1][0] - 2],
            'prev_word-2_dependent': '' if pair[1][0] - 1 <= 1 else self.tokens[pair[1][0] - 3],
            'prev_word-3_dependent': '' if pair[1][0] - 1 <= 2 else self.tokens[pair[1][0] - 4],
            'next_word-1_dependent': '' if pair[1][0] - 1 >= len(self.tokens) - 1 else self.tokens[pair[1][0]],
            'next_word-2_dependent': '' if pair[1][0] - 1 >= len(self.tokens) - 2 else self.tokens[pair[1][0] + 1],
            'next_word-3_dependent': '' if pair[1][0] - 1 >= len(self.tokens) - 3 else self.tokens[pair[1][0] + 2],

            'prev_word-1_dependent_pos_tag': '' if pair[1][0] - 1 <= 0 else self.pos_tags[pair[1][0] - 2],
            'prev_word-2_dependent_pos_tag': '' if pair[1][0] - 1 <= 1 else self.pos_tags[pair[1][0] - 3],
            'prev_word-3_dependent_pos_tag': '' if pair[1][0] - 1 <= 2 else self.pos_tags[pair[1][0] - 4],
            'next_word-1_dependent_pos_tag': '' if pair[1][0] - 1 >= len(self.pos_tags) - 1 else self.pos_tags[pair[1][0]],
            'next_word-2_dependent_pos_tag': '' if pair[1][0] - 1 >= len(self.pos_tags) - 2 else self.pos_tags[pair[1][0] + 1],
            'next_word-3_dependent_pos_tag': '' if pair[1][0] - 1 >= len(self.pos_tags) - 3 else self.pos_tags[pair[1][0] + 2]
        }

    def __init__(self, sentence=None, type='list', log_level=LOG_LEVEL):
        '''
        type: list or str
        '''
        self.logger = None
        self.init_logging(LOG_LEVEL)
        #self.logger.info('Sentence class begin')

        self.tokens_with_root = []
        self.tokens = []
        self.pos_tags = []

        self.edge_features = []
        self.edge_labels = []

        self.index_preds = []
        self.index_labels = []

        self.relation_features = []
        self.relation_preds = []
        self.relation_labels = []

        self.read_graph(sentence, type)

    def read_graph(self, sentence, type='list'):

        if type == 'list':

            for s in sentence:
                self.tokens.append(s[0])
                self.pos_tags.append(s[1])
                if len(s) > 2:
                    self.index_labels.append(int(s[2]))
                    self.relation_labels.append(s[3])

            tokens_root = ['__root__'] + self.tokens

            for idx, value in enumerate(tokens_root):
                self.tokens_with_root.append((idx, value))

        elif type == 'str':

            lines = sentence.split('\n')

            for line in lines:

                if line.strip():
                    line_splitted = line.rstrip().split('\t')
                    self.tokens.append(line_splitted[0])
                    self.pos_tags.append(line_splitted[1])
                    if len(line_splitted) > 2:
                        self.index_labels.append(int(line_splitted[2]))
                        self.relation_labels.append(line_splitted[3])
                else:
                    tokens_root = ['__root__'] + self.tokens

                    for idx, value in enumerate(tokens_root):
                        self.tokens_with_root.append((idx, value))

        #self.logger.info('Read graph finished')

    def extract_edge_features(self, type='test'):
        '''
        add a dummy root node and extract featuers for all possible edges
        '''

        for t1 in self.tokens_with_root[1:]:
            for t2 in self.tokens_with_root[1:]:
                if t1 is not t2:

                    self.edge_features.append(self.create_edge_features((t1, t2)))

                    if type == 'train':
                        if t1[0] == self.index_labels[t2[0] - 1]:
                            self.edge_labels.append(1)
                        else:
                            self.edge_labels.append(0)

        for t3 in self.tokens_with_root[1:]:

            self.edge_features.append(self.create_edge_features((self.tokens_with_root[0], t3)))

            if type == 'train':
                if self.tokens_with_root[0][0] == self.index_labels[t3[0] - 1]:
                    self.edge_labels.append(1)
                else:
                    self.edge_labels.append(0)

        #self.logger.info('Extract edge feature finished')

    def extract_relation_features(self):
        '''
        add a dummy root node and extract featuers for all possible edges
        '''

        for t1 in self.tokens_with_root[1:]:
            self.relation_features.append(self.create_relation_features((t1, self.tokens_with_root[self.index_preds[t1[0]-1]])))

        #self.logger.info('Extract relation feature finished')

    def __repr__(self):
        _str = ''
        for i in range(len(self.tokens)):
            _str = _str + self.tokens[i] + '\t' + self.pos_tags[i] + '\t' + str(self.index_preds[i]) + '\t' + \
                   self.relation_preds[i] + '\n'

        return _str

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

    sentence = str(
        'From\tIN \t3\tcase\n'
        'the\tDT\t3\tdet\n'
        'AP\tNNP\t4\tobl\n'
        'comes\tVBZ\t0\troot\n'
        'this\tDT\t6\tdet\n'
        'story\tNN\t4\tnsubj\n'
        ':\t:\t4\tpunct\n'
    )

    deps = Sentence(sentence, type='str')

    sentence = [
        ['From', 'IN', '3', 'case'],
        ['the', 'DT', '3', 'det'],
        ['AP', 'NNP', '4', 'obl'],
        ['comes', 'VBZ', '0', 'root'],
        ['this', 'DT', '6', 'det'],
        ['story', 'NN', '4', 'nsubj'],
        [':', ':', '4', 'punct']
    ]

    deps = Sentence(sentence, type='list')

    sentence = [
        ['From', 'IN'],
        ['the', 'DT'],
        ['AP', 'NNP'],
        ['comes', 'VBZ'],
        ['this', 'DT'],
        ['story', 'NN'],
        [':', ':']
    ]

    deps = Sentence(sentence, type='list')