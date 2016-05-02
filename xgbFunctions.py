'''
Required functions for xgb.py

__Author__:
Ali Narimani

__Version__:
2.1
'''


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, make_scorer
import re
from nltk.stem.porter import *

### Keys:
Snow = True    # stemmer choice
####


if Snow:
    from nltk.stem.snowball import SnowballStemmer #0.003 improvement but takes twice as long as PorterStemmer
    stemmer = SnowballStemmer('english')
else:
    stemmer = PorterStemmer()


def DeepClean(word):
    word = word.replace('kholerhighland', 'kohler highline')
    word = word.replace('smart', ' smart ')
    word = word.replace('residential', ' residential ')
    word = word.replace('whirlpool', ' whirlpool ')
    word = word.replace('alexandrea',' alexandria ')
    word = word.replace('bicycle',' bicycle ')
    word = word.replace('non',' non ')
    word = word.replace('replacement',' replacement')
    word = word.replace('mowerectrical', 'mow electrical')
    word = word.replace('dishwaaher', 'dishwasher')
    word = word.replace('fairfield',' fairfield ')
    word = word.replace('hooverwindtunnel','hoover windtunnel')
    word = word.replace('airconditionerwith','airconditioner with ')
    word = word.replace('pfistersaxton', 'pfister saxton')
    word = word.replace('eglimgton','ellington')
    word = word.replace('chrome', ' chrome ')
    word = word.replace('foot', ' foot ')    
    word = word.replace('samsung', ' samsung ')
    word = word.replace('galvanised', ' galvanised ')
    word = word.replace('exhaust', ' exhaust ')
    word = word.replace('reprobramable', 'reprogramable')
    word = word.replace('rackcloset', 'rack closet ')
    word = word.replace('hamptonbay', ' hampton bay ')
    word = word.replace('cadet', ' cadet ')
    word = word.replace('weatherstripping', 'weather stripping')
    word = word.replace('poyurethane', 'polyurethane')
    word = word.replace('refrigeratorators','refrigerator')
    word = word.replace('baxksplash','backsplash')
    word = word.replace('inches',' inch ')
    word = word.replace('conditioner',' conditioner ')
    word = word.replace('landscasping',' landscaping ')
    word = word.replace('discontinuedbrown',' discontinued brown ')
    word = word.replace('drywall',' drywall ')
    word = word.replace('carpet', ' carpet ')
    word = word.replace('less', ' less ')
    word = word.replace('tub', ' tub')
    word = word.replace('tubs', ' tub ')
    word = word.replace('marble',' marble ')
    word = word.replace('replaclacemt',' replacement ')
    word = word.replace('non',' non ')
    word = word.replace('soundfroofing', 'sound proofing')
    
    return word

def str_stem(s): 
    if isinstance(s, str):
        s = s.lower()
        s = DeepClean(s)
	
	
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) 
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)    
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)    
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        

        s = s.replace(" x "," xby ")
        s = s.replace("*"," xby ")
        s = s.replace(" by "," xby")
        s = s.replace("x0"," xby 0")
        s = s.replace("x1"," xby 1")
        s = s.replace("x2"," xby 2")
        s = s.replace("x3"," xby 3")
        s = s.replace("x4"," xby 4")
        s = s.replace("x5"," xby 5")
        s = s.replace("x6"," xby 6")
        s = s.replace("x7"," xby 7")
        s = s.replace("x8"," xby 8")
        s = s.replace("x9"," xby 9")
        s = s.replace("0x","0 xby ")
        s = s.replace("1x","1 xby ")
        s = s.replace("2x","2 xby ")
        s = s.replace("3x","3 xby ")
        s = s.replace("4x","4 xby ")
        s = s.replace("5x","5 xby ")
        s = s.replace("6x","6 xby ")
        s = s.replace("7x","7 xby ")
        s = s.replace("8x","8 xby ")
        s = s.replace("9x","9 xby ")
    

        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)    
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)        
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)    
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)    
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)        
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)    
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)        
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)    
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)        

        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool")
        s = s.replace("whirlpoolstainless","whirlpool stainless")

        s = s.replace("  "," ")
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        if s == '':
            s = 'null'
        return s.lower()
    else:
        return "null"
    
def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            cnt+=1
    return cnt

def str_whole_word(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt

def jaccard(a, b):
    a = set(a.split())
    b = set(b.split()) 
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['id','relevance','search_term','product_title','product_description','attr','brand','Synonym',\
		'material','color']
        hd_searches = hd_searches.drop(d_col_drops,axis=1).values
        return hd_searches

class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].apply(str)

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)
