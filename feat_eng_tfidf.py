# -*- coding: utf-8 -*-
"""

__file__

    feat_eng_tfidf.py
    
__description__

     tf-idf features

__author__

    Arman Akbarian
    
"""

from __future__ import print_function
import project_params as pp
import cPickle
from nlp_utils import *
import pandas as pd
import sys
from copy import copy

def create_unigrams(df,col):
    # default Stemmer + Tokenizer - Stopwords from nlp_utils : 
    f = normalizer.normalize 
    df[col+'_unigrams'] = list(df[col].map(lambda x: f(x)))


def prog():
    print(".",end='')
    sys.stdout.flush()
    
def longprog():
    print("....",end='')
    sys.stdout.flush()
    
def cat_col(x,a,b):
        res = '%s %s' % (x[a], x[b])
        return res
    
    
def col_ainb(x,a,b):
    if x[a] in x[b]:
        return 1
    else:
        return 0
        
def find_new_features(df):
    new_features = [ \
            name for name in df.columns \
            if "tfidf" in name \
            or "bow" in name \
            or "brand_match" in name \
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
    
    
    
    ######## tf-idf and count vec features ############
    
    
    vec_types = ['tfidf','bow']
    ngram_range=(1,1)
    
    columns = ['product_title','search_term','Synonym','product_description', \
               'attribute_description']
    
    
    tfidf_pairs = [ ['search_term','product_title'], \
                    ['search_term','product_description'], \
                    ['search_term','attribute_description'], \
                    ['Synonym','product_title'], \
                    ['Synonym','product_description'], \
                    ['Synonym','attribute_description']
                  ]
                  
    for which_two_cols in tfidf_pairs:
        for vec_type in vec_types:
            
            col_a = which_two_cols[0]
            
            col_b = which_two_cols[1]

            
            print("Performing "+vec_type+" on "+col_a+ " " + col_b,end="")
            
            
            corpus = list(dfTrain.apply(lambda x: cat_col(x,col_a,col_b), axis=1)); prog()
            corpus.extend(list(dfTest.apply(lambda x: cat_col(x,col_a,col_b), axis=1))) ; prog()

                
            
            if vec_type == 'tfidf':
                vectorizer = getTFV(ngram_range=ngram_range)
        
            elif vec_type == 'bow':
                vectorizer = getBOW(ngram_range=ngram_range)
                
            feat_name = col_a + '_' + col_b + '_' + vec_type
                
            vectorizer.fit(corpus);  longprog()
            
            print("\n")
            
            
            # Doing things in batch due to low memory! #PoorDataScientist
            
            if ( (col_b == 'product_description') | (col_b == 'attribute_description')):
                batch_size = 1000
            else:
                batch_size = 5000
            
            print("Train",end='')
            ## Train ##
            N = len(dfTrain[col_a])      

            dd = np.zeros(N)
            
            i = 0
            while(i < N):
                start = i
                end = i + batch_size
                if (end > N):
                    end = N
                    
                X_a = vectorizer.transform(dfTrain[col_a][start:end]).toarray()   
                X_b = vectorizer.transform(dfTrain[col_b][start:end]).toarray()
                
                dd[start:end] = np.asarray(map(cosine_sim, X_a[:,:], X_b[:,:]))
                
                i = i + batch_size
                prog()

            dfTrain[feat_name] = dd
            
            print("\n")
            
            print("Test",end='')
            ## Test ##
            N = len(dfTest[col_a])      

            dd = np.zeros(N)
            
            i = 0
            while(i < N):
                start = i
                end = i + batch_size
                if (end > N):
                    end = N
                    
                X_a = vectorizer.transform(dfTest[col_a][start:end]).toarray()   
                X_b = vectorizer.transform(dfTest[col_b][start:end]).toarray()
                
                dd[start:end] = np.asarray(map(cosine_sim, X_a[:,:], X_b[:,:]))
                
                i = i + batch_size
                prog()

            
            
            dfTest[feat_name] = dd

            
            print("Done!")
            
    
    ######### adding a brand feature #############
    
    
    dfTrain['brand_match_count_as_one'] = list(dfTrain.apply(lambda x: \
                                               col_ainb(x,'brand','search_term'),\
                                               axis=1))
                                               
    dfTest['brand_match_count_as_one'] = list(dfTest.apply(lambda x: \
                                               col_ainb(x,'brand','search_term'),\
                                               axis=1))
    
    ###### Saving data ########                                           
    new_features = find_new_features(dfTrain)  

    all_features = copy(new_features)
    
    all_features.extend(['id','product_uid','relevance'])      

    dfTrain = dfTrain[all_features]

    dfTrain.to_csv(pp.output_root_dir +  'Train_tfidf_bow_brand'+ '.csv',index=False)                                  
      
      
    new_features = find_new_features(dfTest)  

    all_features = copy(new_features)
    
    all_features.extend(['id','product_uid'])      

    dfTest = dfTest[all_features]

    dfTest.to_csv(pp.output_root_dir +  'Test_tfidf_bow_brand'+ '.csv',index=False)                                  
                                               
    
    
    


