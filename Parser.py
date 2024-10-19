#http://tartarus.org/~martin/PorterStemmer/python.txt
from PorterStemmer import PorterStemmer

class Parser:

	#A processor for removing the commoner morphological and inflexional endings from words in English
	stemmer=None

	stopwords=[]

	def __init__(self,):
		self.stemmer = PorterStemmer()

		#English stopwords from ftp://ftp.cs.cornell.edu/pub/smart/english.stop
		self.stopwords = open('english.stop', 'r').read().split()


	def clean(self, string):
		""" remove any nasty grammar tokens from string """
		string = string.replace(".","")
		string = string.replace(r"\s+"," ")
		string = string.lower()
		return string
	

	def removeStopWords(self,list):
		""" Remove common words which have no search value """
		return [word for word in list if word not in self.stopwords ]


	def tokenise(self, string):
		""" break string up into tokens and stem words """
		string = self.clean(string)
		words = string.split(" ")
		
		return [self.stemmer.stem(word,0,len(word)-1) for word in words]


import jieba
import jieba.posseg as pseg

class JiebaParser:
    stopwords = []

    def __init__(self):
        # Load Chinese stopwords from a file
        # Assuming you have a file named 'chinese.stop' with Chinese stopwords
        self.stopwords = [] if True else set(open('chinese.stop', 'r', encoding='utf-8').read().split())

        # Initialize jieba
        jieba.initialize()

    def clean(self, string):
        """ remove any nasty grammar tokens from string """
        string = string.replace("。", "")  # Remove Chinese full stop
        string = string.replace("，", "")  # Remove Chinese comma
        string = string.replace(r"\s+", " ")
        return string

    def removeStopWords(self, word_list):
        """ Remove common words which have no search value """
        return [word for word in word_list if word not in self.stopwords]

    def tokenise(self, string):
        """ break string up into tokens """
        string = self.clean(string)
        words = jieba.cut(string)
        return list(words)

    def tokenise_with_pos(self, string):
        """ break string up into tokens with part-of-speech tags """
        string = self.clean(string)
        words = pseg.cut(string)
        return [(word, flag) for word, flag in words]

    def get_nouns_and_verbs(self, string):
        """ Extract nouns and verbs from the string """
        words = self.tokenise_with_pos(string)
        nouns = [word for word, flag in words if flag.startswith('n')]
        verbs = [word for word, flag in words if flag.startswith('v')]
        return nouns, verbs