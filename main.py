from pprint import pprint
from Parser import Parser, JiebaParser
import util
from typing import List, Dict
import os
import math
import numpy as np
from itertools import chain
import nltk
from nltk import word_tokenize, pos_tag


def get_documents(folder_path: str) -> Dict[str, str]:
    files_content = {}
    # List all files in the directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Ensure it's a file (not a directory)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                files_content[filename.split(".")[0]] = file.read()
    
    return files_content

class DocumentVec:
    def __init__(self, tf, tf_idf):
        self.tf = tf
        self.tf_idf = tf_idf

def makeTfIdfVector(tf_vector: np.ndarray, idf_vector: np.ndarray) -> np.ndarray:
    return tf_vector * idf_vector

class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Collection of document term vectors
    documentVectors: Dict[str, DocumentVec] = {}

    tokenized: Dict[str, List[str]] = {}

    #Mapping of vector index to keyword
    vectorKeywordIndex=[]

    #Tidies terms
    parser=None


    def __init__(self, documents={}, parser=None):
        self.documentVectors={}
        self.parser = Parser() if parser is None else parser
        if(len(documents)>0):
            self.build(documents)

    def build(self,documents: Dict[str, str]):
        """ Create the vector space for the passed document strings """
        self.tokenized = self.tokenize(documents)
        self.vectorKeywordIndex = self.getVectorKeywordIndex(self.tokenized)
        self.idfVector = self.makeIdfVector(documents, self.vectorKeywordIndex)
        self.documentVectors = {key: self.makeVector(document, self.idfVector) for key, document in documents.items()}

    def tokenize(self, documents: Dict[str, str]) -> Dict[str, List[str]]:
        """ Tokenize the documents """

        return {key: self.parser.tokenise(doc) for key, doc in documents.items()}

    def getVectorKeywordIndex(self, tokenized: Dict[str, List[str]]) -> Dict[str, int]:
        """ create the keyword associated to the position of the elements within the document vectors """
        #Mapped documents into a single word string
        vocabularyList = list(chain(*tokenized.values()))
        #Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1
        return vectorIndex  #(keyword:position)

    def makeVector(self, document, idfVector) -> DocumentVec:
        """ Convert a document into a vector """
        tf = self.makeTfVector(document)
        tf_idf = makeTfIdfVector(tf, idfVector)
        return DocumentVec(tf, tf_idf)
    

    
    def makeTfVector(self, document) -> np.ndarray:
        vector = np.zeros(len(self.vectorKeywordIndex))
    
        wordList = self.parser.tokenise(document)
        wordList = self.parser.removeStopWords(wordList)
    
        for word in wordList:
            if word not in self.vectorKeywordIndex:
                continue
            vector[self.vectorKeywordIndex[word]] += 1  # Term frequency count
    
        return vector

    def makeIdfVector(self, documentDict: Dict[str, str], vectorKeywordIndex) -> np.ndarray:
        """ create a vector of idf values for the vector space"""
        idfVector = np.zeros(len(vectorKeywordIndex))
    
        for doc in documentDict.values():
            wordList = self.parser.tokenise(doc)
            wordList = self.parser.removeStopWords(wordList)
            for word in util.removeDuplicates(wordList):
                idfVector[vectorKeywordIndex[word]] += 1
        idfVector = np.log(len(documentDict) / idfVector)
    
        return idfVector
    
    def makeFeedBackVector(self, tokenized_doc: List[int]) -> np.ndarray:
        tokenized_doc = self.parser.removeStopWords(tokenized_doc)
        vector = np.zeros(len(self.vectorKeywordIndex))
        tagged = pos_tag(tokenized_doc)
        for word, tag in tagged:
            if tag.startswith('NN') or tag.startswith('VB'):
                vector[self.vectorKeywordIndex[word]] += 1
        return vector

    def pfSearchTfIdfCos(self, queryVector, top=10) -> Dict[str, float]:
        best = self.searchTfIdfCos(queryVector, 1)
        tokenized_doc = self.tokenized[next(iter(best))]
        fb_vector = self.makeFeedBackVector(tokenized_doc)
        new_query = queryVector + 0.5 * fb_vector
        return self.searchTfIdfCos(new_query, top)

    def searchTfIdfCos(self, queryVector, top=10) -> Dict[str, float]:
        ratings = {
            doc_name: util.np_cos(queryVector, documentVector.tf_idf)
            for doc_name, documentVector in self.documentVectors.items()
        }
        sorted_ratings = dict(sorted(ratings.items(), key=lambda item: item[1], reverse=True))
        return dict(list(sorted_ratings.items())[:top])

    def searchTfCos(self, queryVector, top=10) -> Dict[str, float]:
        ratings = {
            doc_name: util.np_cos(queryVector, documentVector.tf)
            for doc_name, documentVector in self.documentVectors.items()
        }
        sorted_ratings = dict(sorted(ratings.items(), key=lambda item: item[1], reverse=True))
        return dict(list(sorted_ratings.items())[:top])
    
    def searchTfIdfEcld(self, queryVector, top=10) -> Dict[str, float]:
        ratings = {
            doc_name: util.np_euclidean_distance(queryVector, documentVector.tf_idf)
            for doc_name, documentVector in self.documentVectors.items()
        }
        sorted_ratings = dict(sorted(ratings.items(), key=lambda item: item[1], reverse=False))
        return dict(list(sorted_ratings.items())[:top])
    
    def searchTfEcld(self, queryVector, top=10) -> Dict[str, float]:
        ratings = {
            doc_name: util.np_euclidean_distance(queryVector, documentVector.tf)
            for doc_name, documentVector in self.documentVectors.items()
        }
        sorted_ratings = dict(sorted(ratings.items(), key=lambda item: item[1], reverse=False))
        return dict(list(sorted_ratings.items())[:top])
    

def print_sperate():
    print("\n" + "-"*50 + "\n")


def parse_ground_truth(file_path: str) -> Dict[str, List[str]]:
    # Initialize the result dictionary
    result = {}
    
    # Open and read the file
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into key and values
            key, values = line.strip().split('\t')
            
            # Convert the string of numbers into a list of integers
            value_list = [int(num) for num in values.strip('[]').split(',')]
            
            # Add to the dictionary
            result[key] = value_list
    
    return result


def prob4():
    collections = get_documents('smaller_dataset/collections')
    vector_space = VectorSpace(collections)
    queries = get_documents('smaller_dataset/queries')
    ground_truth = parse_ground_truth('smaller_dataset/rel.tsv')

    
    vector_space = VectorSpace(collections)
    
    results = {}
    for id, query in queries.items():
        query_vector = vector_space.makeVector(query, vector_space.idfVector)
        results[id] = vector_space.searchTfIdfCos(query_vector.tf_idf)
    
    # Calculate MRR
    mrr = 0
    for id, result in results.items():
        for i, (doc, _) in enumerate(result.items()):
            if int(doc[1:]) in ground_truth[id]:
                mrr += 1 / (i + 1)
                break

    print(f"MRR: {mrr/len(queries):.5f}")

    # Calculate MAP
    map = 0
    for id, result in results.items():
        avp = 0
        correct = 0
        for i, (doc, _) in enumerate(result.items()):
            if int(doc[1:]) in ground_truth[id]:
                correct += 1
                avp += correct / (i + 1)
        if correct == 0:
            continue
        avp /= correct
        map += avp
    
    print(f"MAP: {map/len(queries):.5f}")

    # Calculate Recall
    recall = 0
    for id, result in results.items():
        correct = 0
        for i, (doc, _) in enumerate(result.items()):
            if int(doc[1:]) in ground_truth[id]:
                correct += 1
        recall += correct / 10

    print(f"Recall: {recall/len(queries):.5f}")




if __name__ == '__main__':
    news = get_documents('EnglishNews')
    vector_space = VectorSpace(news)
    query_vector = vector_space.makeVector("Typhoon Taiwan war", vector_space.idfVector)
    
    print("TF-IDF Cosine:")
    results = vector_space.searchTfIdfCos(query_vector.tf_idf)
    for doc, score in results.items():
        print(f"{doc}: {score:.5f}")
    print_sperate()

    print("TF Cosine:")
    results = vector_space.searchTfCos(query_vector.tf)
    for doc, score in results.items():
        print(f"{doc}: {score:.5f}")
    print_sperate()

    print("TF-IDF Euclidean:")
    results = vector_space.searchTfIdfEcld(query_vector.tf_idf)
    for doc, score in results.items():
        print(f"{doc}: {score:.5f}")
    print_sperate()

    print("TF Euclidean:")
    results = vector_space.searchTfEcld(query_vector.tf)
    for doc, score in results.items():
        print(f"{doc}: {score:.5f}")
    print_sperate()

    
    results = vector_space.pfSearchTfIdfCos(query_vector.tf_idf)
    print("Relevance Feedback:")
    for doc, score in results.items():
        print(f"{doc}: {score:.5f}")

    # Chinese News
    news = get_documents('ChineseNews')
    vector_space = VectorSpace(news, parser=JiebaParser())
    query_vector = vector_space.makeVector("資安 遊戲", vector_space.idfVector)
    
    print("TF-IDF Cosine:")
    results = vector_space.searchTfIdfCos(query_vector.tf_idf)
    for doc, score in results.items():
        print(f"{doc}: {score:.5f}")
    print_sperate()

    print("TF Cosine:")
    results = vector_space.searchTfCos(query_vector.tf)
    for doc, score in results.items():
        print(f"{doc}: {score:.5f}")
    print_sperate()

    # Prob4
    prob4()

