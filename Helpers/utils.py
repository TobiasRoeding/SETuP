#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, unicode_literals

# Get the python version (used to try decode an unknow instance to unicode)
from sys import version_info

PY3 = version_info[0] == 3

# Use classical Snowball stemmer for english
import nltk
from nltk.util import ngrams

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

from nltk.corpus import stopwords
stopset = frozenset(stopwords.words('english'))

from itertools import chain

import string
from nltk.corpus import stopwords
from nltk.tag import StanfordNERTagger
from collections import Counter

# Convert an object to its unicode representation (if possible)
def to_unicode(object):
	if isinstance(object, unicode):
		return object
	elif isinstance(object, bytes):
		return object.decode("utf8")
	else:
		print str(object)
		if PY3:
			if hasattr(instance, "__str__"):
				return unicode(instance)
			elif hasattr(instance, "__bytes__"):
				return bytes(instance).decode("utf8")
		else:
			if hasattr(instance, "__unicode__"):
				return unicode(instance)
			elif hasattr(instance, "__str__"):
				return bytes(instance).decode("utf8")

# normalize and stem the word
def stem_word(word):
	return stemmer.stem(normalize_word(word))

# convert to unicode and convert to lower case
def normalize_word(word):
	return to_unicode(word).lower()

def get_len(element):
	return len(tokenizer.tokenize(element))

def get_ngrams(sentence, N):
	tokens = tokenizer.tokenize(sentence.lower())
	clean = [stemmer.stem(token) for token in tokens]
	return [gram for gram in ngrams(clean, N)]

def get_words(sentence, stem=True):
	if stem:
		words = [stemmer.stem(r) for r in tokenizer.tokenize(sentence)]
		return map(normalize_word, words)
	else:
		return map(normalize_word, tokenizer.tokenize(sentence))

def createListOfStringsFromArray(array):
    if isinstance(array[0], list):
        array = list(chain.from_iterable(array))
        return createListOfStringsFromArray(array)
    else:
        return array

def get_list_of_words(array):
	return list(chain.from_iterable(map(tokenizer.tokenize, array)))

def createListOfWordsFromArray(array):
    array = createListOfStringsFromArray(array)
    return get_list_of_words(array)
    
def calculateAverageElementLength(elements):
    elementLengths = [len(element) for element in elements]
    if len(elementLengths) > 0:
        return sum(elementLengths) / float(len(elementLengths))
    else: 
        return 0

def removeStopwords(array):
	stop = stopwords.words('english') + list(string.punctuation)
	filtered = [w for w in array if not w in stop]
	return filtered



def getNamedEntities(nerTagger, document):
    words = createListOfWordsFromArray(document)
    taggedWords = nerTagger.tag(words)
    nerWords = [word for (word,tag) in taggedWords if tag != "O"]
    return nerWords

def counts(words):
    c = Counter(words)
    norm = float(sum(c.values()))
    c = dict([(k, v / norm) for (k, v) in c.items()])
    return c