#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections

import numpy as np
import scipy.stats as stats
import nltk
from nltk.util import ngrams
from utils import stemmer, tokenizer, stopset, normalize_word

import math

def is_ngram_content(ngram):
	for gram in ngram:
		if not(gram in stopset):
			return True
	return False

def get_all_content_words(sentences, stem):
    all_words = []
    for s in sentences:
        if stem:
            all_words.extend([stemmer.stem(r) for r in tokenizer.tokenize(s)])
        else:
            all_words.extend(tokenizer.tokenize(s))

    normalized_content_words = map(normalize_word, all_words)
    return normalized_content_words

def pre_process_summary(summary, stem=True):
    summary_ngrams = get_all_content_words(summary, stem)
    return summary_ngrams

def _ngrams(words, n):
    queue = collections.deque(maxlen=n)
    for w in words:
        queue.append(w)
        if len(queue) == n:
            if is_ngram_content(tuple(queue)):
                yield tuple(queue)

def _ngram_counts(words, n):
	c = collections.Counter(_ngrams(words, n))
	norm = float(sum(c.values()))
	c = dict([(k, v / norm) for (k, v) in c.items()])
	return c

def KL_Divergence(P, Q):
    support = Q.keys()
    P_ = []
    for k in support:
    	tup = ((k, P[k]) if k in P else (k, 0.))
    	P_.append(tup)
    Q_ = [(k,v) for k,v in Q.items() if k in support]
    P_values = np.array([v for k,v in sorted(P_, key=lambda t: t[0])])
    Q_values = np.array([v for k,v in sorted(Q_, key=lambda t: t[0])])
    return stats.entropy(pk=P_values, qk=Q_values)

def distribution_average(P, Q):
    new_dict = collections.defaultdict(float)
    for k, v in P.items():
        new_dict[k] += v / 2.
    for k, v in Q.items():
        new_dict[k] += v / 2.
    return new_dict

def JS_Divergence(P, Q):
    M = distribution_average(P, Q)
    return 0.5 * KL_Divergence(P, M) + 0.5 * KL_Divergence(Q, M)

def JS(peer, models, n, stem=True):
    peer = pre_process_summary(peer, stem)
    models = [pre_process_summary(model, stem) for model in models]

    peer_counts = _ngram_counts(peer, n)
    models_counts = [_ngram_counts(model, n) for model in models]

    js_avg = [JS_Divergence(peer_counts, model) for model in models_counts]
    return sum(js_avg) / float(len(models_counts))

if __name__ == '__main__':
	s = ['a  a b b c a a b']
	model = [['a a b c f g a b c'], ['d e f a a b c'], ['b c a b e f a e']]

	print JS(s, model, 1)


