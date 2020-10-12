## -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import argparse

# load word2vec model from the file
def load_model(fi):
    fi = open(fi, 'r')

    # vocab size and dim
    line = fi.readline().strip().split()
    vocab_size, dim = int(line[0]), int(line[1])
    print 'vocab_size=%d, dim=%d' % (vocab_size, dim)

    vocab_vecs = [] # each item in the list is the word vector
    vocab_hash = {} # each item in the dict is word:index
    n = 0
    for line in fi:
        tokens = line.strip().split()
        word = tokens[0]
        vec = [float(v) for v in tokens[1:]]
        vec = vec / np.linalg.norm(vec) # vector normalization
        vocab_hash[word] = n
        vocab_vecs.append(vec)
        n += 1

    assert n == vocab_size, 'load model error: unique word count loaded is not equal to vocab_size!'
    fi.close()

    return vocab_vecs, vocab_hash


# find the best analogy word according the model
# for w1 w2 w3 w4 there is: w1 - w2 :: w3 - w4 (:: means "analogy to")
# for example: Beijing - China :: Toyko - Japan
def find_analogy(w1, w2, w3, vocab_vecs, vocab_hash):
    dim = len(vocab_vecs[0])
    w1 = w1.lower()
    w2 = w2.lower()
    w3 = w3.lower()
    v1 = np.zeros(dim)
    v2 = np.zeros(dim)
    v3 = np.zeros(dim)
    if w1 in vocab_hash: v1 = np.array(vocab_vecs[vocab_hash[w1]], dtype=np.float)
    if w2 in vocab_hash: v2 = np.array(vocab_vecs[vocab_hash[w2]], dtype=np.float)
    if w3 in vocab_hash: v3 = np.array(vocab_vecs[vocab_hash[w3]], dtype=np.float)
    v4 = v2 - v1 + v3
    nv4 = v4 / np.linalg.norm(v4)

    # find the most similar vector and the word
    max_sim = 0.0
    max_word = 'unknown'
    for w in vocab_hash:
        if w == w1 or w == w2 or w == w3: continue
        v = np.array(vocab_vecs[vocab_hash[w]], dtype=np.float)
        sim = np.dot(v, nv4) # Cosine similarity
        #euc = np.linalg.norm(v - nv4) # Euclidean distance
        if sim > max_sim:
            max_sim = sim
            max_word = w

    return max_word, max_sim


# compute spearman correlation coefficent according wordsim353 test file
def test_wordsim353(fi, vocab_vecs, vocab_hash):
    fi = open(fi, 'r')
    assert fi, 'open file error!'

    vec1 = []
    vec2 = []
    for line in fi:
        tokens = line.strip().split()
        word1 = tokens[0].lower()
        word2 = tokens[1].lower()
        vec1.append(float(tokens[2]))

        sim = 0.0
        if word1 in vocab_hash and word2 in vocab_hash:
            v1 = vocab_vecs[vocab_hash[word1]]
            v2 = vocab_vecs[vocab_hash[word2]]
            sim = np.dot(v1, v2) # consine similarity
        else:
            print 'Warning: %s or %s not found in vocab.' % (word1, word2)
        print word1, '\t', word2, '\t', tokens[2], '\t', sim

        vec2.append(sim)

    X1 = pd.Series(vec1)
    Y1 = pd.Series(vec2)
    r = X1.corr(Y1, method='spearman')
    print 'spearman correlation for wordsim353 is %.4f' % r

    fi.close()


# compute accuracy according to analogy test file
def test_analogy(fi, vocab_vecs, vocab_hash):
    fi = open(fi, 'r')
    assert fi, 'open file error!'

    acc_num = 0
    total_num = 0
    for line in fi:
        tokens = line.strip().split()
        if len(tokens) != 4: continue
        w, s = find_analogy(tokens[0], tokens[1], tokens[2], vocab_vecs, vocab_hash)
        if w == tokens[3].lower(): acc_num += 1
        total_num += 1
        print '[ %d/%d ] %s %s %s \t\t--> %s' % (acc_num, total_num, tokens[0], tokens[1], tokens[2], w)

    print 'Accuracy for Google analogy is %d/%d (%.4f)' % (acc_num, total_num, float(acc_num) / total_num)
    fi.close()


def test_model(model, et, test):
    vocab_vecs, vocab_hash = load_model(model)
    if et == 0:
        test_wordsim353(test, vocab_vecs, vocab_hash)
    else:
        test_analogy(test, vocab_vecs, vocab_hash)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', help='model file', dest='model', required=True)
    parser.add_argument('-et', help='evaluation type: 0 for wordsim353, 1 for Google Analogy', dest='et', required=True, type=int)
    parser.add_argument('-test', help='test file to be evaluated', dest='test', required=True)
    args = parser.parse_args()

    test_model(args.model, args.et, args.test)
