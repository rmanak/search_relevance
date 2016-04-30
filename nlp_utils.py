# -*- coding: utf-8 -*-
"""

__file__

    nlp_utils.py
    
__description__

     - Text cleaning utilities
     - Default global objects = (stopwords,tokenizer,stemmer)
     - Class: Stemmer (wrapper for popular stemmers)
     - Class: Tokenizer (wrapper for RegExpTokenizer)
     - Class: Normalizer (Stemmer + Tokenizer - stopwords)
     - Class: wrappers for sklearn:
            [TfidfVectorizer + Stemmer] 
            [CountVectorizer + Stemmer]
     - Defaults for TFIDF and CountVectorizers
     - Various tokenizer regexp patterns
     - Common set distances used in nlp

    
__author__

    Arman Akbarian
    
"""

import re
import csv
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np




####### Wrapper for WordNetLemmatizer #########


class WordNetStemmer(WordNetLemmatizer):
    def stem(self,word,pos=u'n'):
        return self.lemmatize(word,pos)


########  Wrapper for all  of the popular stemmers ###########


class Stemmer(object):
    def __init__(self,stemmer_type):
        self.stemmer_type = stemmer_type
        if (self.stemmer_type == 'porter'):
            self.stemmer = nltk.stem.PorterStemmer()
        elif (self.stemmer_type == 'snowball'):
            self.stemmer = nltk.stem.SnowballStemmer('english')
        elif (self.stemmer_type == 'lemmatize'):
            self.stemmer = WordNetStemmer()
        else:
            raise NameError("'"+stemmer_type +"'" + " not supported")
            


######## Simple wordreplacer object using a dictionary  ############

class WordReplacer(object):
    def __init__(self, words_map):
        self.words_map = words_map
    def replace(self, words):
        return [self.words_map.get(w, w) for w in words]

      
####### wordreplacer with csv file  for word replacement dictionary ########

     
class CSVWordReplacer(WordReplacer):
    def __init__(self, fname):
        words_map = {}
        for line in csv.reader(open(fname)):
            word, syn = line
            if word.startswith('#'):
                continue
            words_map[word] = syn
        super(CSVWordReplacer, self).__init__(words_map)


######### for now just a wrapper to RegexpTokenizer #########

class Tokenizer():
    def __init__(self,pattern):
        self.pattern = pattern 
        self.tokenizer = RegexpTokenizer(self.pattern)
        


######## defining a default stopwords set #############

stopwords = nltk.corpus.stopwords.words('english')
stopwords = set(stopwords)



######## defining a default stemmer ##########


stemmer_type = 'snowball' # optimum of speed and handling
stemmer = Stemmer(stemmer_type).stemmer

###### other options for stemmer: ##########

#stemmer_type = 'porter' # Fastest, doesn' handle as good as snowball
#stemmer_type = 'lemmatize' # Slowest (aka the best!)



######### default token pattern #############

token_pattern = r"(?u)\b\w\w+\b" # good enough and fast enough (from tfidf library)

####### other options for token pattern #########
#token_pattern = r"(?:[A-Za-z]\.)+|\w+(?:[']\w+)*|\$?\d+(?:\.\d+)?%?" # might be slow
#token_pattern= r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*"
#token_pattern = r'[\w']+'
#token_pattern = r'[ \t\n]+'
#token_pattern = r'\W+'
#token_pattern = r'\w+|\S\w*'



####### defining a default tokenizer ######

tokenizer = Tokenizer(token_pattern).tokenizer


######### Tokenizer + Stemmer - Stopwords ###########

class Normalizer(object):
    def __init__(self,stemmer,tokenizer,stop_words):
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.stop_words = stop_words
    def normalize(self, text):
        return [self.stemmer.stem(token) 
                for token in self.tokenizer.tokenize(text.lower()) 
                if token not in self.stop_words]
                    
######### defining a default normalizer ##########


normalizer = Normalizer(stemmer,tokenizer,stopwords)

########### Normalizer without Stemmer ##############

class NoStemNormalizer(object):
    def __init__(self,tokenizer,stop_words):
        self.tokenizer = tokenizer
        self.stop_words = stop_words
    def normalize(self, text):
        return [token for token in self.tokenizer.tokenize(text.lower()) 
                if token not in self.stop_words]
    

nostemnormalizer = NoStemNormalizer(tokenizer,stopwords)



########## Stemmer  + Tfidf wrapper ############


class StemmedTfidfVectorizer(TfidfVectorizer):
        
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (stemmer.stem(w) for w in analyzer(doc))
        


########## Stemmer + CountVectorizer wrapper #############


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(CountVectorizer, self).build_analyzer()
        return lambda doc: (stemmer.stem(w) for w in analyzer(doc))
        
        
########## Defaults TF-IDF & Count Vectorizers ########
        
        
#======== TF-IDF Vectorizer =========#
        
tfidf__norm = "l2"
tfidf__max_df = 0.75
tfidf__min_df = 3

def getTFV(token_pattern = token_pattern,
           norm = tfidf__norm,
           max_df = tfidf__max_df,
           min_df = tfidf__min_df,
           ngram_range = (1, 1),
           vocabulary = None,
           stop_words = 'english'):
    tfv =TfidfVectorizer(min_df=min_df, max_df=max_df, max_features=None, 
                         strip_accents='unicode', analyzer='word', 
                         token_pattern=token_pattern,
                         ngram_range=ngram_range, use_idf=True, 
                         smooth_idf=True, sublinear_tf=True,
                         stop_words = stop_words, norm=norm, vocabulary=vocabulary)
    return tfv   


#========= CountVectorizer =========#

bow__max_df = 0.75
bow__min_df = 3

def getBOW(token_pattern = token_pattern,
           max_df = bow__max_df,
           min_df = bow__min_df,
           ngram_range = (1, 1),
           vocabulary = None,
           stop_words = 'english'):
    bow =CountVectorizer(min_df=min_df, max_df=max_df, max_features=None, 
                         strip_accents='unicode', analyzer='word', 
                         token_pattern=token_pattern,
                         ngram_range=ngram_range,
                         stop_words = stop_words, vocabulary=vocabulary)
    return bow     


########################################################
        
# ------------------------------
# Simple text cleaning using 
#        
#     -replacement dict 
#        
#        or
#        
#     -WordReplacer object
#--------------------------------
        
def clean_text(text,replace_dict=None,words_replacer=None):
    
    text = text.lower()
    
    if replace_dict is not None:
        for k, v in replace_dict.items():
            text = re.sub(k,v,text)
    
    if words_replacer is not None:
        text = text.split(' ')
        text = words_replacer.replace(text)
        text = ' '.join(text)
    
    return text
    
    


####### Standard distance metrics ##########



def JaccardCoef(A, B):
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A.union(B))
    coef = safe_divide(intersect, union)
    return coef
    

def DiceDist(A, B):
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A) + len(B)
    d = safe_divide(2*intersect, union)
    return d


def compute_dist(A, B, dist="jaccard_coef"):
    if dist == "jaccard_coef":
        d = JaccardCoef(A, B)
    elif dist == "dice_dist":
        d = DiceDist(A, B)
    return d


def pairwise_jaccard_coef(A, B):
    coef = np.zeros((A.shape[0], B.shape[0]), dtype=float)
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            coef[i,j] = JaccardCoef(A[i], B[j])
    return coef


def pairwise_dice_dist(A, B):
    d = np.zeros((A.shape[0], B.shape[0]), dtype=float)
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            d[i,j] = DiceDist(A[i], B[j])
    return d
    

def pairwise_dist(A, B, dist="jaccard_coef"):
    if dist == "jaccard_coef":
        d = pairwise_jaccard_coef(A, B)
    elif dist == "dice_dist":
        d = pairwise_dice_dist(A, B)
    return d
    

###### other common tools #########

def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed

def safe_divide(x,y,res=0.0):
    if y != 0.0:
        res = float(x)/float(y)
    return res
    
def cosine_sim(x, y):
    try:
        d = cosine_similarity(x.reshape(1,-1), y.reshape(1,-1))
        d = d[0][0]
    except:
        d = 0.0
    return d

if __name__ == '__main__':
    print(__doc__)
    
