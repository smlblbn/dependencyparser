import logging
import numpy as np
from collections import defaultdict

logging.basicConfig(
    format='%(asctime)s-|%(name)20s:%(funcName)12s|'
           '-%(levelname)8s-> %(message)s')
logger = logging.getLogger('mst')
logger.setLevel(logging.INFO)


def mst(scores):
    """
    Chu-Liu-Edmonds' algorithm
    for finding minimum spanning arborescence in graphs.
    Calculates the arborescence with node 0 as root.
    Source: https://github.com/chantera/biaffineparser/blob/master/utils.py

    WARNING: mind the comment below.
    This mst function expects scores[i][j] to be the score from j to i,
    not from i to j (as you would probably expect!).
    If you use a graph where you have the convention
    that a head points to its dependent,
    you will need to transpose it before calling this function.
    That is, call `mst(scores.T)` instead of `mst(scores)`.

    :param scores: `scores[i][j]`
        is the weight of edge from node `j` to node `i`
    :returns an array containing the head node
        (node with edge pointing to current node) for each node,
        with head[0] fixed as 0
    """
    length = scores.shape[0]
    scores = scores * (1 - np.eye(length))
    heads = np.argmax(scores, axis=1)
    heads[0] = 0
    tokens = np.arange(1, length)
    roots = np.where(heads[tokens] == 0)[0] + 1

    # print initial heads
    logger.debug("initial heads: " + str(heads))

    # deal with roots
    if len(roots) < 1:
        logger.debug("no node is pointing to root, choosing one")
        root_scores = scores[tokens, 0]
        head_scores = scores[tokens, heads[tokens]]
        new_root = tokens[np.argmax(root_scores / head_scores)]
        logger.debug("new root is:" + str(new_root))
        heads[new_root] = 0
    elif len(roots) > 1:
        logger.debug("multiple nodes are pointing to root, choosing one")
        root_scores = scores[roots, 0]
        scores[roots, 0] = 0
        new_heads = np.argmax(scores[roots][:, tokens], axis=1) + 1
        new_root = roots[np.argmin(scores[roots, new_heads] / root_scores)]
        logger.debug("new root is: " + str(new_root))
        heads[roots] = new_heads
        heads[new_root] = 0

    # construct edges and vertices
    edges = defaultdict(set)
    vertices = set((0,))
    for dep, head in enumerate(heads[tokens]):
        vertices.add(dep + 1)
        edges[head].add(dep + 1)

    # identify cycles & contract
    for cycle in _find_cycle(vertices, edges):
        logger.debug("Found cycle! " + str(cycle))
        dependents = set()
        to_visit = set(cycle)
        while len(to_visit) > 0:
            node = to_visit.pop()
            logger.debug("Contraction, visiting node: " + str(node))
            if node not in dependents:
                dependents.add(node)
                to_visit.update(edges[node])
        cycle = np.array(list(cycle))
        old_heads = heads[cycle]
        old_scores = scores[cycle, old_heads]
        non_heads = np.array(list(dependents))
        scores[np.repeat(cycle, len(non_heads)),
               np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
        new_heads = np.argmax(scores[cycle][:, tokens], axis=1) + 1
        new_scores = scores[cycle, new_heads] / old_scores
        change = np.argmax(new_scores)
        changed_cycle = cycle[change]
        old_head = old_heads[change]
        new_head = new_heads[change]
        heads[changed_cycle] = new_head
        edges[new_head].add(changed_cycle)
        edges[old_head].remove(changed_cycle)

    return heads


def _find_cycle(vertices, edges):
    """
    Finds cycles in given graph, where the graph is provided as (vertices, edges).
    https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm  # NOQA
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/etc/tarjan.py  # NOQA
    """
    _index = [0]
    _stack = []
    _indices = {}
    _lowlinks = {}
    _onstack = defaultdict(lambda: False)
    _SCCs = []

    def _strongconnect(v):
        _indices[v] = _index[0]
        _lowlinks[v] = _index[0]
        _index[0] += 1
        _stack.append(v)
        _onstack[v] = True

        for w in edges[v]:
            if w not in _indices:
                _strongconnect(w)
                _lowlinks[v] = min(_lowlinks[v], _lowlinks[w])
            elif _onstack[w]:
                _lowlinks[v] = min(_lowlinks[v], _indices[w])

        if _lowlinks[v] == _indices[v]:
            SCC = set()
            while True:
                w = _stack.pop()
                _onstack[w] = False
                SCC.add(w)
                if not (w != v):
                    break
            _SCCs.append(SCC)

    for v in vertices:
        if v not in _indices:
            _strongconnect(v)

    return [SCC for SCC in _SCCs if len(SCC) > 1]


if __name__ == '__main__':

    w2i = defaultdict(lambda: len(w2i))
    sentence = "root john saw mary".split()
    sentence_ids = [w2i[token] for token in sentence]
    num_words = len(sentence)
    i2w = {i: w for w, i in w2i.items()}

    print(sentence)
    print(sentence_ids)
    print(num_words)

    # define the scores
    scores = np.full([num_words, num_words], -1.)
    scores[w2i['root']][w2i['saw']] = 10.
    scores[w2i['root']][w2i['mary']] = 9.
    scores[w2i['root']][w2i['john']] = 9.
    scores[w2i['john']][w2i['saw']] = 20.
    scores[w2i['john']][w2i['mary']] = 3.
    scores[w2i['saw']][w2i['mary']] = 30.
    scores[w2i['saw']][w2i['john']] = 30.
    scores[w2i['mary']][w2i['john']] = 11.
    scores[w2i['mary']][w2i['saw']] = 0.

    print(scores)

    heads = mst(scores.T)
    print("final heads:", heads)

    # define a dummy sentence
    w2i = defaultdict(lambda: len(w2i))
    sentence = np.arange(10)
    sentence_ids = sentence
    num_words = len(sentence)
    i2w = {i: w for w, i in w2i.items()}

    print(sentence)
    print(sentence_ids)
    print(num_words)

    # define the scores
    np.random.seed(seed=42)
    scores = 100 * np.random.rand(num_words, num_words)
    print(scores)

    heads = mst(scores.T)
    print("final heads:", heads)
