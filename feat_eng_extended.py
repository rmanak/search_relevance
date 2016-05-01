# -*- coding: utf-8 -*-
"""

__file__


    feat_eng_extended.py
    
__description__

     Various feature engineering:
     - counting of grams(unigram,bigram,trigrams)
     - count of uniques , ration of uniques in all
     - couting of digits, ratio of counts in all
     - word intersection counts and ratios
     - word occurance positions statistics
     - set distances (Jaccard etc..)

    
__author__

    Arman Akbarian
    
"""
from __future__ import print_function
import project_params as pp

try:
	import cPickle

except:
	import pickle as cPickle
 
from nlp_utils import *
import pandas as pd
import sys
import ngrams
from copy import copy


def create_unigrams(df,col):
    # default Tokenizer - Stopwords from nlp_utils : 
    # no need to stem, it is done in preprocessing
    f = nostemnormalizer.normalize 
    df[col+'_unigrams'] = list(df[col].map(lambda x: f(x)))
    

def create_bigrams(df,col):
    f = ngrams.getBigram
    join_str = '_'
    df[col+'_bigrams'] = list(df[col+'_unigrams'].map(lambda x: f(x,join_str)))
    

def create_trigrams(df,col):
    f = ngrams.getTrigram
    join_str = '_'
    df[col+'_trigrams'] = list(df[col+'_unigrams'].map(lambda x: f(x,join_str)))

    
def create_digits(df,col):
    f = lambda x: list([w for w in x if w.isdigit()])
    df[col+'_digits'] = list(df[col+'_unigrams'].map(f))
    
def count_a_in_b(a,b):
    res = sum([1. for w in a if w in set(b)])
    if res is not None:
        return res
    else:
        return 0
    
def get_position_list(target, obs):
    pos_of_obs_in_target = [0]
    if len(obs) != 0:
        pos_of_obs_in_target = [j for j,w in enumerate(obs, start=1) if w in target]
        if len(pos_of_obs_in_target) == 0:
            pos_of_obs_in_target = [0]
    return pos_of_obs_in_target
     
                                              
    
def prog():
    print(".",end='')
    sys.stdout.flush()
    
def longprog():
    print("....",end='')
    sys.stdout.flush()
    
   
def create_features(df,dfName,grams,columns,):
    
        print("Working on data: " + dfName + '\n' )
        
        
        ##### grams #####
        
        print("Creating grams.",end='')
        for gram in grams:
            for col in columns:
                eval('create_'+gram)((df),col); prog()
    
        print("Done!")
    
        ####  Word count features ######
    
        print("Creating word count features.",end='')

        for gram in grams: 
            for col in columns:
                s1 = 'count_of_'+col+'_'+gram
                s2 = 'count_of_unique_'+col+'_'+gram 
                (df)[s1] = (df)[col+'_'+gram].map(lambda x: len(x))
                (df)[s2] = (df)[col+'_'+gram].map(lambda x: len(set(x)))
                (df)['ratio_of_unique_'+col+'_'+gram] = map(safe_divide,(df)[s2],(df)[s1])
                prog()
                   
    
        print("Done!")
    
        ####  Digit features ####
   
        print("Creating digit features.",end='')
        
        for col in columns:
            create_digits((df),col)
            s1 = 'count_of_digits_in_'
            (df)[s1+col] = (df)[col+'_digits'].map(lambda x: len(x))
            (df)['ratio_of_digits_in_'+col] = map(safe_divide,
                                                      (df)[s1+col],(df)['count_of_'+col+'_unigrams'])
            prog()            
    
        print("Done!")
    
        #### word intersects #####
    
        print("Creating word intersect count features.",end='')

        for gram in grams:
          for col_a in columns:
             for col_b in columns:
                if col_a != col_b:
                    
                    s1 = col_a+'_'+gram
                    s2 = col_b+'_'+gram
                    
                    (df)['count_of_'+s1+'_in_'+col_b] = map(count_a_in_b,(df)[s1],(df)[s2])
                    prog()
                    (df)['ratio_of_'+s1+'_in_'+col_b] = map(safe_divide,(df)['count_of_'+s1+'_in_'+col_b],
                                                               (df)['count_of_'+s1])
                    prog()
         
          (df)["title_%s_in_search_div_search_%s"%(gram,gram)] = map(safe_divide, (df)["count_of_product_title_%s_in_search_term"%gram], (df)["count_of_search_term_%s"%gram])
          (df)["title_%s_in_search_div_search_%s_in_title"%(gram,gram)] = map(safe_divide, (df)["count_of_product_title_%s_in_search_term"%gram], (df)["count_of_search_term_%s_in_product_title"%gram])
          (df)["description_%s_in_search_div_search_%s"%(gram,gram)] = map(safe_divide, (df)["count_of_product_description_%s_in_search_term"%gram], (df)["count_of_search_term_%s"%gram])
          (df)["description_%s_in_search_div_search_%s_in_description"%(gram,gram)] = map(safe_divide, (df)["count_of_product_description_%s_in_search_term"%gram], (df)["count_of_search_term_%s_in_product_description"%gram])
          prog()              

        print("Done!")
    
        ########## digit intersect ##############
    
        print("Creating digits intersect count features.",end='')
    
    
        for col_a in columns:
            for col_b in columns:
                if col_a != col_b:
                    s1 = col_a + '_digits'
                    s2 = col_b + '_digits'
                    (df)['count_of_'+s1+'_in_'+col_b] = map(count_a_in_b,(df)[s1],(df)[s2])
                    prog()
                    (df)['ratio_of_'+s1+'_in_'+col_b] = map(safe_divide,(df)['count_of_'+s1+'_in_'+col_b],
                                                                (df)['count_of_digits_in_'+col_a])
                    prog()
    
        print("Done!")
     

        ####### positions #########
    
        print("Creating word position features.",end='')
    
        
        for gram in grams:
          for target_name in columns:
              for obs_name in columns:
                 if target_name != obs_name:
                    pos = list((df).apply(lambda x: get_position_list(x[target_name+"_"+gram], obs=x[obs_name+"_"+gram]), axis=1))
                    # stats feat on pos                   
                    (df)["pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = map(np.min, pos)
                    (df)["pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = map(np.mean, pos)
                    prog()
                    (df)["pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] = map(np.median, pos)
                    (df)["pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = map(np.max, pos)
                    (df)["pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = map(np.std, pos)
                    prog()
                    # stats feat on normalized_pos
                    (df)["normalized_pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = map(safe_divide, (df)["pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)], (df)["count_of_%s_%s" % (obs_name, gram)])
                    (df)["normalized_pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = map(safe_divide, (df)["pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)], (df)["count_of_%s_%s" % (obs_name, gram)])
                    (df)["normalized_pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] = map(safe_divide, (df)["pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)], (df)["count_of_%s_%s" % (obs_name, gram)])
                    prog()
                    (df)["normalized_pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = map(safe_divide, (df)["pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)], (df)["count_of_%s_%s" % (obs_name, gram)])
                    (df)["normalized_pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = map(safe_divide, (df)["pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] , (df)["count_of_%s_%s" % (obs_name, gram)])
                    prog()
    
               
        print("Done!")
    
        ######## distances ###########

        print("Creating distance features.",end='')

        dists = ["jaccard_coef", "dice_dist"]
    
        for dist in dists:
          for gram in grams:
            for i in range(len(columns)-1):
                for j in range(i+1,len(columns)):
                    target_name = columns[i]
                    obs_name = columns[j]
                    (df)["%s_of_%s_between_%s_%s"%(dist,gram,target_name,obs_name)] = \
                            list((df).apply(lambda x: compute_dist(x[target_name+"_"+gram], x[obs_name+"_"+gram], dist), axis=1))
                    prog()
                    
        
        print("Done!")
        print("Done with ",dfName,'!')
        
        print("========================")
    
def find_new_features(df):
    new_features = [ \
            name for name in df.columns \
            if "count" in name \
            or "ratio" in name \
            or "div" in name \
            or "between" in name \
            or "pos_of" in name \
            ]       
    return new_features
        
        
    
if __name__ == "__main__":
    
    ##### loading data #########
    
    print("Loading data.",end="")
    
    with open(pp.train_proccessed_file, "rb") as f:
        dfTrain = cPickle.load(f)
    prog()
        
    with open(pp.test_processed_file, "rb") as f:
        dfTest = cPickle.load(f)
    prog()
    
    with open(pp.product_descriptions_processed_file, "rb") as f:
        dfProdDescription = cPickle.load(f)   
    prog()
    
    with open(pp.attribute_processed_file, "rb") as f:
        dfAttribute = cPickle.load(f)   
    
        
    print("Done!")
    
    
    ###### merging for further intersection features #####
    
    dfTrain = pd.merge(dfTrain, dfProdDescription , how='left', on='product_uid')
    dfTest = pd.merge(dfTest, dfProdDescription , how='left', on='product_uid')
    
    del dfProdDescription
    
    dfTrain = pd.merge(dfTrain, dfAttribute , how='left', on='product_uid')
    dfTest = pd.merge(dfTest, dfAttribute , how='left', on='product_uid')
    
    del dfAttribute
    
    dfTrain['attribute_description'] = dfTrain['attribute_description'].fillna(u'noattributedescription')
    dfTest['attribute_description'] = dfTest['attribute_description'].fillna(u'noattributedescription')
    
    dfTrain['size_attribute'] = dfTrain['size_attribute'].fillna(u'nosizeattributefound')
    dfTest['size_attribute'] = dfTest['size_attribute'].fillna(u'nosizeattributefound')
    
    dfTrain['brand'] = dfTrain['brand'].fillna(u'unknownbrand')
    dfTest['brand'] = dfTest['brand'].fillna(u'unknownbrand')
    
    

    grams = ['unigrams','bigrams']
    columns = ['product_title','search_term','Synonym','product_description', \
               'attribute_description']
    
    
    # Breaking down the data due to memory issues #PoorDataScientist
    
    # len dfTrain = 74067
    
    # len dfTest =  166693


    
    dfTrain1 = dfTrain[:37000].copy()
    dfTrain2 = dfTrain[37000:].copy()
    
    del dfTrain
    
    
    
    dfTest1 = dfTest[:40000].copy()
    dfTest2 = dfTest[40000:80000].copy()
    dfTest3 = dfTest[80000:120000].copy()
    dfTest4 = dfTest[120000:].copy()
    
    del dfTest
    
        
    # Train 1
    create_features(dfTrain1,'dfTrain1',grams,columns)
    new_features = find_new_features(dfTrain1)
    all_features = copy(new_features)
    all_features.extend(['id','product_uid','relevance'])
    dfTrain1 = dfTrain1[all_features]
    dfTrain1.to_csv(pp.output_root_dir + 'Train1' + '.csv' , index=False )
    del dfTrain1
    
    # Train 2
    create_features(dfTrain2,'dfTrain2',grams,columns)
    new_features = find_new_features(dfTrain2)
    all_features = copy(new_features)
    all_features.extend(['id','product_uid','relevance'])
    dfTrain2 = dfTrain2[all_features]
    dfTrain2.to_csv(pp.output_root_dir + 'Train2' + '.csv' , index=False )
    del dfTrain2    
    

    # Test 1
    create_features(dfTest1,'dfTest1',grams,columns)
    new_features = find_new_features(dfTest1)
    all_features = copy(new_features)
    all_features.extend(['id','product_uid'])
    dfTest1 = dfTest1[all_features]
    dfTest1.to_csv(pp.output_root_dir + 'Test1' + '.csv' , index=False )
    del dfTest1
    


    # Test 2
    create_features(dfTest2,'dfTest2',grams,columns)
    new_features = find_new_features(dfTest2)
    all_features = copy(new_features)
    all_features.extend(['id','product_uid'])
    dfTest2 = dfTest2[all_features]
    dfTest2.to_csv(pp.output_root_dir + 'Test2' + '.csv' , index=False )
    del dfTest2
    
    # Test 3
    create_features(dfTest3,'dfTest3',grams,columns)
    new_features = find_new_features(dfTest3)
    all_features = copy(new_features)
    all_features.extend(['id','product_uid'])
    dfTest3 = dfTest3[all_features]
    dfTest3.to_csv(pp.output_root_dir + 'Test3' + '.csv' , index=False )
    del dfTest3
    
    # Test 4
    create_features(dfTest4,'dfTest4',grams,columns)
    new_features = find_new_features(dfTest4)
    all_features = copy(new_features)
    all_features.extend(['id','product_uid'])
    dfTest4 = dfTest4[all_features]
    dfTest4.to_csv(pp.output_root_dir + 'Test4' + '.csv' , index=False )
    del dfTest4
    
    
