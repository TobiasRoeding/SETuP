#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Helpers.FeatureExtractors import FeatureExtractors
import json
import os.path
import datetime
from gensim.models.keyedvectors import KeyedVectors
import Config

class FeatureGenerator:
    singleFeatureExtractors = {}
    combinedFeatureExtractors = {}

    def printWithTime(self, string):
        print(str(datetime.datetime.now()) + " - " + string)

    def generateSingleFeatureScores(self, feature, sourceDocuments, systemSummaries, referenceSummaries):
        scores = []
        for systemSummary in systemSummaries:
            scores.append(self.singleFeatureExtractors[feature](sourceDocuments, systemSummary, referenceSummaries))
        return scores

    def generateCombinedFeatureScores(self, feature, sourceDocuments, systemSummaries, referenceSummaries):
        scores = []
        for systemSummaryA in systemSummaries:
            for systemSummaryB in systemSummaries:
                scores.append(self.combinedFeatureExtractors[feature](sourceDocuments, systemSummaryA, systemSummaryB, referenceSummaries))
        return scores

    def generateFeatureScores(self, sourceDocuments, systemSummaries, referenceSummaries, outputFilename):
        featureData = self.loadFeatureData(outputFilename)

        for feature in self.singleFeatureExtractors:
            if feature not in featureData['singleFeatures']:
                self.printWithTime("generating scores for " + feature)
                featureData['singleFeatures'][feature] = self.generateSingleFeatureScores(feature, sourceDocuments, systemSummaries, referenceSummaries)
                if outputFilename:
                    self.saveFeatureData(featureData, outputFilename)
            else:
                self.printWithTime("scores for " + feature + " already exist")
        
        for feature in self.combinedFeatureExtractors:
            if feature not in featureData['combinedFeatures']:
                self.printWithTime("generating scores for " + feature)
                featureData['combinedFeatures'][feature] = self.generateCombinedFeatureScores(feature, sourceDocuments, systemSummaries, referenceSummaries)
                if outputFilename:
                    self.saveFeatureData(featureData, outputFilename)
            else:
                self.printWithTime("scores for " + feature + " already exist")
        
        return featureData

    def loadFeatureData(self, fileName):
        blank = {
            "singleFeatures": {},
            "combinedFeatures": {}
        }
        if fileName is not None:
            try:
                featureDataFileReader = open(fileName, 'r')
            except Exception:
                featureDataFileWriter = open(fileName, 'w')
                featureDataFileWriter.write(json.dumps(blank, indent=4, sort_keys=True))
                featureDataFileWriter.close()
                featureDataFileReader = open(fileName, 'r')
            featureData = json.loads(featureDataFileReader.read())
            featureDataFileReader.close()
        else:
            featureData = blank
        return featureData

    def saveFeatureData(self, featureData, fileName):
        featureDataFileWriter = open(fileName, 'w')
        featureDataFileWriter.write(json.dumps(featureData))
        featureDataFileWriter.close()

    def __init__(self):

        self.featureData = {
            "singleFeatures": {},
            "combinedFeatures": {}
        }

        featureExtractors = FeatureExtractors()

        self.singleFeatureExtractors = {
            "Rouge1Alpha0": featureExtractors.rouge1Alpha0FeatureExtractor,
            "Rouge1Alpha05": featureExtractors.rouge1Alpha05FeatureExtractor,
            "Rouge1Alpha1": featureExtractors.rouge1Alpha1FeatureExtractor,
            "Rouge2Alpha0": featureExtractors.rouge2Alpha0FeatureExtractor,
            "Rouge2Alpha05": featureExtractors.rouge2Alpha05FeatureExtractor,
            "Rouge2Alpha1": featureExtractors.rouge2Alpha1FeatureExtractor,
            "JS-1": featureExtractors.jenson1FeatureExtractor,
            "JS-2": featureExtractors.jenson2FeatureExtractor,
            "JS-3": featureExtractors.jenson3FeatureExtractor,
            "AverageWordLength": featureExtractors.averageWordLengthFeatureExtractor,
            "AverageNumberOfWords": featureExtractors.averageNumberOfWordsFeatureExtractor,
            "AverageSentenceLength": featureExtractors.averageSentenceLengthFeatureExtractor,
            "AverageSimilarity": featureExtractors.averageSimilarityFeatureExtractor,
            "RepetitionVsUniqueWords": featureExtractors.repetitionVsUniqueWordsFeatureExtractor,
            "JaccardDistance": featureExtractors.jaccardDistanceFeatureExtractor
        }
        self.combinedFeatureExtractors = {
            "FlagBigRouge1Alpha0": featureExtractors.rougeBinaryFlagForBigRouge1Recall
        }

        if Config.pathWord2Vec:
            wordVectorsParam = KeyedVectors.load_word2vec_format(Config.pathWord2Vec, binary=True) 
            featureExtractors.setWord2VecParam(wordVectorsParam)
            self.singleFeatureExtractors.update({
                "RougeWe1Alpha0": featureExtractors.rougeWe1Alpha0FeatureExtractor,
                "RougeWe1Alpha05": featureExtractors.rougeWe1Alpha05FeatureExtractor,
                "RougeWe1Alpha1": featureExtractors.rougeWe1Alpha1FeatureExtractor,
                "RougeWe2Alpha0": featureExtractors.rougeWe2Alpha0FeatureExtractor,
                "RougeWe2Alpha05": featureExtractors.rougeWe2Alpha05FeatureExtractor,
                "RougeWe2Alpha1": featureExtractors.rougeWe2Alpha1FeatureExtractor
            })
        else: 
            self.printWithTime("No word vectors configured -> Skipping feature ROUGE with word embeddings")

        if Config.pathNerJar and Config.pathNerModel:
            nerJarParam = Config.pathNerJar
            nerModelParam = Config.pathNerModel
            featureExtractors.setNerParams(nerJarParam, nerModelParam)
            self.singleFeatureExtractors.update({
                "NamedEntitiesSimilarity": featureExtractors.namedEntitySimilarityFeatureExtractor
            })
        else: 
            self.printWithTime("No jar and/or model for ner tagging configured -> Skipping feature Named Entity Similarity")
