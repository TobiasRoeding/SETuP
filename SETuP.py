#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Option to train a model with new data. Split in two steps: feature generation and training
Option to classifiy two summaries using the existing model
Option to classify two summaries using a new model


'''
import click
import Classification as classification
import DataLoader as loader

modelFiles = loader.getFileNames("./Models")

@click.group()
def cli():
    pass

@cli.command()
@click.option('--model', prompt='Enter one of the following model names:\n' + ' \n'.join(modelFiles) + '\n', 
                help='Select one of the models in the folder \"Models\" to predict your data')
@click.option('--path-source-documents', prompt='Enter the path to the source documents', 
                help='Enter the path to the source documents. Each document has to have its own file.')
@click.option('--path-system-summaries', prompt='Enter the path to the system summaries', 
                help='Enter the path to the system summaries. There have to be exactly two system summaries, and each summary has to have its own file.')
@click.option('--path-reference-summaries', prompt='Enter the path to the reference summaries', 
                help='Enter the path to the reference summaries. Each summary has to have its own file.')
def classify(model, path_source_documents, path_system_summaries, path_reference_summaries):
    [sourceDocuments, systemSummaries, referenceSummaries] = loader.loadClassificationData(path_source_documents, path_system_summaries, path_reference_summaries)
    if len(systemSummaries) > 2:
        raise Exception('More than two system summaries provided')
    elif model not in modelFiles:
        raise Exception('Model does not exist')
    else:
        model = classification.loadModel('Models/' + model)
        prediction = classification.classifyData(model, sourceDocuments, systemSummaries, referenceSummaries)
        print(prediction)

@cli.command()
@click.option('--model', prompt='Enter the filename for the finished model', 
                help='Enter the name of the model, which will be saved in the folder \"Models\"')
@click.option('--path-source-documents', prompt='Enter the path to the source documents', 
                help='Enter the path to the source documents. Each document has to have its own file.')
@click.option('--path-system-summaries', prompt='Enter the path to the system summaries', 
                help='Enter the path to the system summaries. Each summary has to have its own file')
@click.option('--path-reference-summaries', prompt='Enter the path to the reference summaries', 
                help='Enter the path to the reference summaries. Each summary has to have its own file.')
@click.option('--path-system-summary-scores', prompt='Enter the path to the system summary scores', 
                help='Enter the path to the system summary scores. Each summary score has to have its own file with exactly the same name as the actual system summary.')
def train(model, path_source_documents, path_system_summaries, path_reference_summaries, path_system_summary_scores):
    [sourceDocuments, systemSummaries, referenceSummaries, systemSummaryScores] = loader.loadTrainingData(path_source_documents, path_system_summaries, path_reference_summaries, path_system_summary_scores)
    trainedModel = classification.trainModel(model, sourceDocuments,systemSummaries,referenceSummaries, systemSummaryScores)
    classification.saveModel(trainedModel, "Models/" + model + ".pkl")

if __name__ == '__main__':
    cli()