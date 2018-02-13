#!/usr/bin/env python
# -*- coding: utf-8 -*-

import string
from nltk.stem.porter import PorterStemmer
import utils as utils
from gensim.similarities import Similarity
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

def stemListOfTokens(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def stemAndTokenizeArray(arrayOfText):
    tokens = utils.createListOfWordsFromArray(arrayOfText)
    filtered = utils.removeStopwords(tokens)
    stems = stemListOfTokens(filtered)
    return stems

def calculateAverageSimilarity(singleDocument, arrayOfDocuments):
    gen_docs = [stemAndTokenizeArray(document) for document in arrayOfDocuments]
    dictionary = Dictionary(gen_docs)
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
    tf_idf = TfidfModel(corpus)
    sims = Similarity('/tmp/',tf_idf[corpus], num_features=len(dictionary))
    query_doc = stemAndTokenizeArray(singleDocument)
    query_doc_bow = dictionary.doc2bow(query_doc)
    query_doc_tf_idf = tf_idf[query_doc_bow]

    similarities = sims[query_doc_tf_idf]
    return sum(similarities) / len(similarities)