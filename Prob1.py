from pprint import pprint
from Parser import Parser
import util
from typing import List, Dict
import os
import math
import numpy as np
from itertools import chain

def get_news(folder_path: str) -> Dict[str, str]:
    files_content = {}
    # List all files in the directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Ensure it's a file (not a directory)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                files_content[filename] = file.read()
    
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

    # tokenized documents
    tokenized: Dict[str, List[str]] = {}

    
    documentVectors: Dict[str, DocumentVec] = {}


    #Mapping of vector index to keyword
    vectorKeywordIndex=[]

    #Tidies terms
    parser=None


    def __init__(self, documents={}):
        self.documentVectors={}
        self.parser = Parser()
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
    


    def searchTfIdfCos(self, doc, top=10) -> Dict[str, float]:
        queryVector = self.makeVector(doc, self.idfVector)
        ratings = {
            doc_name: util.np_cos(queryVector.tf_idf, documentVector.tf_idf)
            for doc_name, documentVector in self.documentVectors.items()
        }
        sorted_ratings = dict(sorted(ratings.items(), key=lambda item: item[1], reverse=True))
        return dict(list(sorted_ratings.items())[:top])

    def searchTfCos(self, doc, top=10) -> Dict[str, float]:
        queryVector = self.makeVector(doc, self.idfVector)
        ratings = {
            doc_name: util.np_cos(queryVector.tf, documentVector.tf)
            for doc_name, documentVector in self.documentVectors.items()
        }
        sorted_ratings = dict(sorted(ratings.items(), key=lambda item: item[1], reverse=True))
        return dict(list(sorted_ratings.items())[:top])
    
    def searchTfIdfEcld(self, doc, top=10) -> Dict[str, float]:
        queryVector = self.makeVector(doc, self.idfVector)
        ratings = {
            doc_name: util.np_euclidean_distance(queryVector.tf_idf, documentVector.tf_idf)
            for doc_name, documentVector in self.documentVectors.items()
        }
        sorted_ratings = dict(sorted(ratings.items(), key=lambda item: item[1], reverse=False))
        return dict(list(sorted_ratings.items())[:top])
    
    def searchTfEcld(self, doc, top=10) -> Dict[str, float]:
        queryVector = self.makeVector(doc, self.idfVector)
        ratings = {
            doc_name: util.np_euclidean_distance(queryVector.tf, documentVector.tf)
            for doc_name, documentVector in self.documentVectors.items()
        }
        sorted_ratings = dict(sorted(ratings.items(), key=lambda item: item[1], reverse=False))
        return dict(list(sorted_ratings.items())[:top])

def print_sperate():
    print("\n" + "="*50 + "\n")


if __name__ == '__main__':
    news = get_news('EnglishNews/EnglishNews')
    vector_space = VectorSpace(news)

    results = vector_space.searchTfIdfCos("Typhoon Taiwan war")
    print("TF-IDF Cosine:")
    for doc, score in results.items():
        print(f"{doc}: {score:.5f}")
    print_sperate()

    results = vector_space.searchTfCos("Typhoon Taiwan war")
    print("TF Cosine:")
    for doc, score in results.items():
       print(f"{doc}: {score:.5f}")
    print_sperate()

    results = vector_space.searchTfEcld("Typhoon Taiwan war")
    print("TF Euclidean:")
    for doc, score in results.items():
       print(f"{doc}: {score:.5f}")
    print_sperate()

    results = vector_space.searchTfIdfEcld("Typhoon Taiwan war")
    print("TF-IDF Euclidean:")
    for doc, score in results.items():
       print(f"{doc}: {score:.5f}")



