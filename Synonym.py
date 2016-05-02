'''
This code finds the synonyms of the words in search quesries.
Returns "ForgetIt" if the search query consists of stop words only
(e.g. search_query = 'or'), or no synomyms are found for the search query

INPUT FILES: 
Hamid_data_trimmed/training_data_trimmed.csv
Hamid_data_trimmed/test_data_trimmed.csv

OUTPUTS:
aliCleaned/train_SynonymDropBox.csv
aliCleaned/test_SynonymDropBox.csv

__Author__:
Ali Narimani

__Veresion__:
1.3
'''

import pandas as pd
from PyDictionary import PyDictionary
from nltk.corpus import stopwords
import sys


dictionary = PyDictionary()
stops = stopwords.words('english')

def synonfinder(text):
    forgetNum = forgetNum +1
    y = ''
    if (text == 'ForgetIt'):
        print  forgetNum+1,'th ForgetIt'
        sys.stdout.flush()
        return 'ForgetIt'
    else:
        try:
            for s in text.split():
                synonyms = dictionary.synonym(s)
                if (synonyms is not None):
                    for x in synonyms:
                        y = y + str(x) + ' '
            return y.rstrip()
        except UnicodeEncodeError:
            return 'ForgetIt'

def remove_stops(s):
    y = ''
    for word in s.split():
        if word not in stops:
            y = y + word + ' '
    return y.rstrip()

### Reading input files. ``Hamid_data_trimmed`` contains no numbers. 
### This makes the code a lot faster:
train = pd.read_csv('../data/Hamid_data_trimmed/training_data_trimmed.csv',encoding = 'ISO-8859-1')
test = pd.read_csv('../data/Hamid_data_trimmed/test_data_trimmed.csv',encoding = 'ISO-8859-1')

train.fillna('ForgetIt', inplace=True)
test.fillna('ForgetIt', inplace=True)

train['search_trimmed'] = train.search_term_trimmed.map(remove_stops)
test['search_trimmed'] = test.search_term_trimmed.map(remove_stops)

train['Synonym'] = train.search_trimmed.map(synonfinder)
forgetNum = 0
test['Synonym'] = test.search_trimmed.map(synonfinder)

### preparing files for submission:
trainSyn = train.ix[:,['id','Synonym']]
testSyn = test.ix[:,['id','Synonym']]

trainSyn2.to_csv("../data/aliCleaned/train_SynonymDropBox.csv", index=False, encoding = 'utf-8')
testSyn2.to_csv("../data/aliCleaned/test_SynonymDropBox.csv", index=False, encoding = 'utf-8')

# End of Code
