#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import listdir
from os.path import isfile, join
import nltk

def getFileNames(path):
    return [f for f in listdir(path) if isfile(join(path, f))]

def loadTextFiles(path):
    fileNames = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".txt")]
    fileContents = []
    for fileName in fileNames:
        f = open(join(path, fileName), 'r')
        fileContents.append(f.read().strip())
        f.close()
    return fileContents

def tokenizeDocuments(documents):
    return [nltk.sent_tokenize(document) for document in documents]

def convertScores(documents):
    return [int(float(document)) for document in documents]

def loadClassificationData(pathSourceDocuments, pathSystemSummaries, pathReferenceSummaries):
    sourceDocuments = tokenizeDocuments(loadTextFiles(pathSourceDocuments))
    systemSummaries = tokenizeDocuments(loadTextFiles(pathSystemSummaries))
    referenceSummaries = tokenizeDocuments(loadTextFiles(pathReferenceSummaries))
    return [sourceDocuments, systemSummaries, referenceSummaries]

def loadTrainingData(pathSourceDocuments, pathSystemSummaries, pathReferenceSummaries, pathSystemSummaryScores):
    classificationData = loadClassificationData(pathSourceDocuments,pathSystemSummaries,pathReferenceSummaries)
    systemSummaryScores = convertScores(loadTextFiles(pathSystemSummaryScores))
    classificationData.append(systemSummaryScores)
    return classificationData