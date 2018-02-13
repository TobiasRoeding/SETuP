#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime

class VectorGenerator:

    def printWithTime(self, string):
        print(str(datetime.datetime.now()) + " - " + string) 

    def getAllSingleFeatures(self, featureData):
        return [featureType for featureType in featureData['singleFeatures']]
    
    def getAllCombinedFeatures(self, featureData):
        return [featureType for featureType in featureData['combinedFeatures']]

    def generateFeatureVector(self, featureData, numberOfSummaries, i, j):
        featureVector = []
        for featureType in featureData['singleFeatures']:
            featureVector.extend(featureData['singleFeatures'][featureType][i])
            featureVector.extend(featureData['singleFeatures'][featureType][j])
        for featureType in featureData['combinedFeatures']:
            featureVector.extend(featureData['combinedFeatures'][featureType][i * numberOfSummaries + j])
        return featureVector

    def generateFeatureVectors(self, featureData):
        featureVectors = []
        singleFeatures = [feature for feature in featureData['singleFeatures']]
        numberOfSummaries = len(featureData['singleFeatures'][singleFeatures[0]])
        for i in xrange(numberOfSummaries):
            for j in xrange(numberOfSummaries):
                featureVectors.append(self.generateFeatureVector(featureData, numberOfSummaries, i, j))
        return featureVectors

    def generateTargetScore(self, scoreA, scoreB):
        if scoreA > scoreB:
            return 1
        elif scoreA < scoreB:
            return -1
        else:
            return 0

    def generateTargetVector(self, targetData):
        targetVector = []
        numberOfSummaries = len(targetData)
        for i in xrange(numberOfSummaries):
            for j in xrange(numberOfSummaries):
                targetVector.append(self.generateTargetScore(targetData[i], targetData[j]))
        return targetVector

    def filterVectors(self, featureVectors, targetVector):
        filteredFeatureVectors = []
        filteredTargetVector = []
        for i in xrange(len(targetVector)):
            if targetVector[i] != 0:
                filteredFeatureVectors.append(featureVectors[i])
                filteredTargetVector.append(targetVector[i])

        return [filteredFeatureVectors, filteredTargetVector]