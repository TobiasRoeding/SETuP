import utils
import ROUGE as r
import JS as js
import Similarity as s
from nltk.tag import StanfordNERTagger

class FeatureExtractors:

    def setWord2VecParam(self, wordVectorsParam):
        self.wordVectors = wordVectorsParam

    def setNerParams(self, nerJarParam, nerModelParam):
        self.nerTagger = StanfordNERTagger(nerModelParam, nerJarParam)

    ## single feature extractors

    def averageWordLengthFeatureExtractor(self, originalDocuments, machineSummary, humanSummaries):
        listOfWords = utils.createListOfWordsFromArray(originalDocuments)
        avdWordLengthOD = utils.calculateAverageElementLength(listOfWords)
        
        listOfWords = utils.createListOfWordsFromArray(machineSummary)
        avdWordLengthMS = utils.calculateAverageElementLength(listOfWords)
        
        listOfWords = utils.createListOfWordsFromArray(humanSummaries)
        avdWordLengthHS = utils.calculateAverageElementLength(listOfWords)
        
        return [avdWordLengthOD, avdWordLengthMS, avdWordLengthHS]

    def averageNumberOfWordsFeatureExtractor(self, originalDocuments, machineSummary, humanSummaries):
        wordsInDocuments = [utils.get_list_of_words(document) for document in originalDocuments]
        numberOfWordsOD = utils.calculateAverageElementLength(wordsInDocuments)
        numberOfWordsMS = len(utils.createListOfWordsFromArray(machineSummary))
        wordsInSummaries = [utils.get_list_of_words(summary) for summary in humanSummaries]
        numberOfWordsHS = utils.calculateAverageElementLength(wordsInSummaries)
        return [numberOfWordsOD, numberOfWordsMS, numberOfWordsHS]

    def averageSentenceLengthFeatureExtractor(self, originalDocuments, machineSummary, humanSummaries):
        avdSentenceLengthOD = utils.calculateAverageElementLength(utils.createListOfStringsFromArray(originalDocuments))
        avdSentenceLengthMS = utils.calculateAverageElementLength(machineSummary)
        avdSentenceLengthHS = utils.calculateAverageElementLength(utils.createListOfStringsFromArray(humanSummaries))
        return [avdSentenceLengthOD, avdSentenceLengthMS, avdSentenceLengthHS]

    def jensonFeatureExtractor(self, originalDocuments, machineSummary, humanSummaries, n):
        score = js.JS(machineSummary, humanSummaries, n)
        if score > 1:
            score = 1
        return [score]

    def jenson1FeatureExtractor(self, originalDocuments, machineSummary, humanSummaries):
        return self.jensonFeatureExtractor(originalDocuments, machineSummary, humanSummaries, 1)
    def jenson2FeatureExtractor(self, originalDocuments, machineSummary, humanSummaries):
        return self.jensonFeatureExtractor(originalDocuments, machineSummary, humanSummaries, 2)
    def jenson3FeatureExtractor(self, originalDocuments, machineSummary, humanSummaries):
        return self.jensonFeatureExtractor(originalDocuments, machineSummary, humanSummaries, 3)

    def rougeFeatureExtractor(self, originalDocuments, machineSummary, humanSummaries, n, alpha):
        return [r.rouge_n(machineSummary, humanSummaries, n, alpha)]

    def rouge1Alpha0FeatureExtractor(self, originalDocuments, machineSummary, humanSummaries):
        return self.rougeFeatureExtractor(originalDocuments, machineSummary, humanSummaries, 1, 0)
    def rouge1Alpha05FeatureExtractor(self, originalDocuments, machineSummary, humanSummaries):
        return self.rougeFeatureExtractor(originalDocuments, machineSummary, humanSummaries, 1, 0.5)
    def rouge1Alpha1FeatureExtractor(self, originalDocuments, machineSummary, humanSummaries):
        return self.rougeFeatureExtractor(originalDocuments, machineSummary, humanSummaries, 1, 1)
    def rouge2Alpha0FeatureExtractor(self, originalDocuments, machineSummary, humanSummaries):
        return self.rougeFeatureExtractor(originalDocuments, machineSummary, humanSummaries, 2, 0)
    def rouge2Alpha05FeatureExtractor(self, originalDocuments, machineSummary, humanSummaries):
        return self.rougeFeatureExtractor(originalDocuments, machineSummary, humanSummaries, 2, 0.5)
    def rouge2Alpha1FeatureExtractor(self, originalDocuments, machineSummary, humanSummaries):
        return self.rougeFeatureExtractor(originalDocuments, machineSummary, humanSummaries, 2, 1)

    def rougeWeFeatureExtractor(self, originalDocuments, machineSummary, humanSummaries, n, alpha):
        return [r.rouge_n_we(machineSummary, humanSummaries, self.wordVectors, n, alpha)]

    def rougeWe1Alpha0FeatureExtractor(self, originalDocuments, machineSummary, humanSummaries):
        return self.rougeWeFeatureExtractor(originalDocuments, machineSummary, humanSummaries, 1, 0)
    def rougeWe1Alpha05FeatureExtractor(self, originalDocuments, machineSummary, humanSummaries):
        return self.rougeWeFeatureExtractor(originalDocuments, machineSummary, humanSummaries, 1, 0.5)
    def rougeWe1Alpha1FeatureExtractor(self, originalDocuments, machineSummary, humanSummaries):
        return self.rougeWeFeatureExtractor(originalDocuments, machineSummary, humanSummaries, 1, 1)
    def rougeWe2Alpha0FeatureExtractor(self, originalDocuments, machineSummary, humanSummaries):
        return self.rougeWeFeatureExtractor(originalDocuments, machineSummary, humanSummaries, 2, 0)
    def rougeWe2Alpha05FeatureExtractor(self, originalDocuments, machineSummary, humanSummaries):
        return self.rougeWeFeatureExtractor(originalDocuments, machineSummary, humanSummaries, 2, 0.5)
    def rougeWe2Alpha1FeatureExtractor(self, originalDocuments, machineSummary, humanSummaries):
        return self.rougeWeFeatureExtractor(originalDocuments, machineSummary, humanSummaries, 2, 1)

    def averageSimilarityFeatureExtractor(self, originalDocuments, machineSummary, humanSummaries):
        simWithOD = s.calculateAverageSimilarity(machineSummary, originalDocuments)
        simWithHS = s.calculateAverageSimilarity(machineSummary, humanSummaries)
        return [simWithOD, simWithHS]

    def namedEntitySimilarityFeatureExtractor(self, originalDocuments, machineSummary, humanSummaries):
        peer = utils.removeStopwords(utils.getNamedEntities(self.nerTagger, machineSummary))
        modelsOD = [utils.removeStopwords(utils.getNamedEntities(self.nerTagger, document)) for document in originalDocuments]
        modelsHS = [utils.removeStopwords(utils.getNamedEntities(self.nerTagger, document)) for document in humanSummaries]
        
        peerCount = utils.counts(peer)
        modelsODCount = [utils.counts(model) for model in modelsOD]
        modelsHSCount = [utils.counts(model) for model in modelsHS]
        
        jsAvgOD = [js.JS_Divergence(peerCount, model) for model in modelsODCount]
        jsAvgHS = [js.JS_Divergence(peerCount, model) for model in modelsHSCount]

        resultOD = sum(jsAvgOD) / float(len(modelsODCount))
        resultHS = sum(jsAvgHS) / float(len(modelsHSCount))
        
        if resultOD > 1:
            resultOD = 1
        if resultHS > 1:
            resultHS = 1
            
        return [resultOD, resultHS]

    def repetitionVsUniqueWordsFeatureExtractor(self, originalDocuments, machineSummary, humanSummaries):
        words = utils.createListOfWordsFromArray(machineSummary)
        wordsLowercase = [word.lower() for word in words]
        setOfWords = set(wordsLowercase)
        if len(wordsLowercase) > 0:
            percentage = len(setOfWords)/float(len(wordsLowercase))
        else:
            percentage = 0
        return [percentage]

    def jaccardDistanceFeatureExtractor(self, originalDocuments, machineSummary, humanSummaries):
        setMS = set(utils.createListOfWordsFromArray(machineSummary))
        jaccardDistances = []
        for humanSummary in humanSummaries:
            setHS = set(utils.createListOfWordsFromArray(humanSummary))
            distance = 1 - len(setMS.intersection(setHS)) / float(len(setMS.union(setHS)))
            jaccardDistances.append(distance)
        averageJaccardDistance = sum(jaccardDistances) / float(len(jaccardDistances))
        return [averageJaccardDistance]

    ## combined feature extractors

    def rougeBinaryFlagForBigRouge1Recall(self, originalDocuments, machineSummary1, machineSummary2, humanSummaries):
        rougeMS1 = self.rougeFeatureExtractor(originalDocuments, machineSummary1, humanSummaries, 1, 0)
        rougeMS2 = self.rougeFeatureExtractor(originalDocuments, machineSummary2, humanSummaries, 1, 0)
        distance = rougeMS1[0] - rougeMS2[0]
        if distance > 0 and abs(distance) > 0.001:
            return [1]
        elif distance < 0 and abs(distance) > 0.001:
            return [-1]
        else:
            return [0]