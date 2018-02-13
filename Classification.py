#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from FeatureGenerator import FeatureGenerator
from VectorGenerator import VectorGenerator

def saveModel(classifier, path):
    joblib.dump(classifier, path)

def loadModel(path):
    return joblib.load(path)

def classifyData(model, sourceDocuments, systemSummaries, referenceSummaries):
    featureGenerator = FeatureGenerator()
    featureData = featureGenerator.generateFeatureScores(sourceDocuments,systemSummaries,referenceSummaries, None)

    vectorGenerator = VectorGenerator()
    featureVector = vectorGenerator.generateFeatureVector(featureData, 0 , 0, 1)

    return model.predict([featureVector])

def trainModel(fileName, sourceDocuments, systemSummaries, referenceSummaries, systemSummaryScores):
    featureGenerator = FeatureGenerator()
    featureData = featureGenerator.generateFeatureScores(sourceDocuments,systemSummaries,referenceSummaries, "Feature Data/" + fileName + ".json")
    
    vectorGenerator = VectorGenerator()
    featureVectors = vectorGenerator.generateFeatureVectors(featureData)
    targetVector = vectorGenerator.generateTargetVector(systemSummaryScores)
    [filteredFeatureVectors, filteredTargetVector] = vectorGenerator.filterVectors(featureVectors, targetVector)

    classifier = RandomForestClassifier(n_jobs=-1,n_estimators=100, min_samples_split=12, min_samples_leaf=25)
    classifier.fit(filteredFeatureVectors, filteredTargetVector)

    return classifier