# -*- coding: utf-8 -*-
"""

__file__

    preprocess.py
    
__description__

    pre-processing the data:
      - text cleaning
      - merging synonyms 
      - stemming
      - cleaning attribute
      - building attribute_description
      - extracting brand and size for products
    
__author__

    Arman Akbarian
    
"""

from __future__ import print_function
from nlp_utils import *
import cPickle
import pandas as pd
import project_params as pp
import sys
from spell_corr import spell_check_dict
import re

def prog():
    print(".",end='')
    sys.stdout.flush()
    
def longprog():
    print("....",end='')
    sys.stdout.flush()
    
    
def clean_attributes(df):
    def cat_text(x):
        res = '%s %s' % (x['name'], x['value'])
        return res   
    
    
    df['attribute_description'] = list(df.apply(cat_text, axis=1)); prog()
    remove_bullet = lambda x: re.sub(r'(bullet\d+)', r' ', x) 
    df['attribute_description'] = df['attribute_description'].map(remove_bullet); prog()
    
   
    def has_size_attribute(x):
        if ('height' in x) | ('width' in x) | ('length' in x) | ('depth' in x):
            return True
        else:
            return False
    
    df['has_size'] = df['name'].map(has_size_attribute); prog()
    
    dfSize = df.loc[df.has_size, ['product_uid','value']]
    
    df = df.drop(['has_size'],axis=1)
    
    all_sizes =  dfSize.groupby('product_uid').agg(lambda x : ' '.join(x))
    indx = all_sizes.index.map(int)

    dfSize = pd.DataFrame({'product_uid':list(indx), 'size_attribute':list(all_sizes['value'])})
    
    prog()
    
    dfBrand = df.loc[df['name'] == 'MFG Brand Name',['product_uid','value']].rename(columns={"value": "brand"})
    
    dfBrand['brand']= dfBrand['brand'].map(lambda x: x.lower())
    
    all_descr = df[['product_uid','attribute_description']].groupby('product_uid').agg(lambda x: ' '.join(x))    
    indx = all_descr.index.map(int)
    
    prog()
    
    df = pd.DataFrame({'product_uid':list(indx), 'attribute_description':list(all_descr['attribute_description'])})
    
    df = pd.merge(df,dfSize,on='product_uid',how='left')
    
    df = df.fillna(u'unknownsize')
    
    df = pd.merge(df,dfBrand,on='product_uid',how='left')
    
    df = df.fillna(u'unknownbrand')
    
    
    return df
    
    
def extra_clean(word):
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
    
    
def str_clean_stem_lower(s): 
    try:
        s = s.lower()
        s = extra_clean(s)
    
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) 
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)    
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)    
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
     
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)    
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)        
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)    
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)    
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)        
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)    
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)        
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)    
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
    
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
    
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool")
        s = s.replace("whirlpoolstainless","whirlpool stainless")
    
        s = s.replace("  "," ")
        
        # using default stemmer from nlp_utils:
        s = (' ').join([stemmer.stem(z) for z in s.split(' ')])
        if s == '':
            s = u'null'
        return s.lower()
    except:
        return u'null'
    
if __name__ == '__main__':


    ######### reading csv files #############
    print("Loading data.",end='')
    
    
    dfTrain = pd.read_csv(pp.train_raw_file,encoding=pp.encoding); prog()
    
    dfTest = pd.read_csv(pp.test_raw_file,encoding=pp.encoding); prog()
    
    dfAttribute = pd.read_csv(pp.attribute_raw_file,encoding=pp.encoding); prog()
    
    dfProdDescription = pd.read_csv(pp.description_raw_file,encoding=pp.encoding); prog()
    
    dfSynTrain = pd.read_csv(pp.synonyms_train_raw_file,encoding=pp.encoding); prog()
    
    dfSynTest = pd.read_csv(pp.synonyms_test_raw_file,encoding=pp.encoding)
    
    print("Done.")
    
    ######## cleaning and adding synonyms ###############
    
    print("Pre-processing data.",end='')
    
    # very few dense NAN values in Attribute
    dfAttribute = dfAttribute.dropna(); prog()
    
    dfAttribute =  clean_attributes(dfAttribute) ; longprog()

    # fixing some typo's in search terms:

    dfTrain['search_term'].replace(spell_check_dict,inplace=True)
    
    dfTest['search_term'].replace(spell_check_dict,inplace=True)
    
    # adding synonyms
    
    dfTrain = pd.merge(dfTrain,dfSynTrain,on='id',how='left')
    
    dfTrain['Synonym'] = dfTrain['Synonym'].fillna(u'synonymnotfound')
    
    del dfSynTrain
    
    dfTest = pd.merge(dfTest,dfSynTest,on='id',how='left')
    
    del dfSynTest
    
    dfTest['Synonym'] = dfTest['Synonym'].fillna(u'synonymnotfound')
    
    # cleaning text contents:
    
    colNames = ['product_title', 'search_term', 'Synonym']

    for col in colNames:
        dfTrain[col] = dfTrain[col].map(str_clean_stem_lower); prog()
        dfTest[col] = dfTest[col].map(str_clean_stem_lower); prog()
        
    dfProdDescription['product_description'] = dfProdDescription['product_description'].map(str_clean_stem_lower)
    longprog()
    
    dfAttribute['attribute_description'] = dfAttribute['attribute_description'].map(str_clean_stem_lower); 
    
    
    print("Done.")

    
    print("Saving data.",end='')
    
    with open(pp.train_proccessed_file, "wb") as f:
        cPickle.dump(dfTrain, f, -1)
    prog()    
    
    with open(pp.test_processed_file, "wb") as f:
        cPickle.dump(dfTest, f, -1)
    prog()   
    
    with open(pp.attribute_processed_file, "wb") as f:
        cPickle.dump(dfAttribute, f, -1)
    prog()   
    
    with open(pp.product_descriptions_processed_file, "wb") as f:
        cPickle.dump(dfProdDescription, f, -1)
    print("Done.")
    


