# -*- coding: utf-8 -*-
"""

__file__

    model_xgb.py
    
__description__

     XGB implementation

    
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
from copy import copy
import sys

from sklearn.cross_validation import train_test_split
import xgboost as xgb


                     
    
def prog():
    print(".",end='')
    sys.stdout.flush()
    
def longprog():
    print("....",end='')
    sys.stdout.flush()
    
def load_data():
    
    
    print("Loading data.",end="")
    
    with open(pp.train_features_file, "rb") as f:
        dfTrain = cPickle.load(f); prog()
    
    with open(pp.test_features_file, "rb") as f:
        dfTest = cPickle.load(f); prog()
        
    print("Done!")
  
    return dfTrain, dfTest
    
def append_dataframes(list_df):
    df = pd.DataFrame()
    for fname in list_df:
        df_tmp = pd.read_csv(fname)
        df = df.append(df_tmp,ignore_index=True)
    del df_tmp
    return df
    
def parse_textfile_todict(fname):
    fdict = {}
    with open(fname,"r") as f:
        for line in f:
            a, b = line.split()
            fdict[a] = b
            
    return fdict
    
def merge_dfs(df1,df2,common_col='id',ignore_col=list()):
    
    df1 = pd.merge(df1,df2.drop(ignore_list,axis=1),on=common_col)
    
    return df1
    
def redefine_tfidf_feats(df):
    
    #vec_types = ['tfidf','bow']
    
    tfidf_pairs = [ ['search_term','product_title'], \
                    ['search_term','product_description'], \
                    ['search_term','attribute_description'], \
                    ['Synonym','product_title'], \
                    ['Synonym','product_description'], \
                    ['Synonym','attribute_description']
                  ]
    accum_score = np.zeros(len(df))
    for which_two_cols in tfidf_pairs:
            
            col_a = which_two_cols[0]
            col_b = which_two_cols[1]
            
            
            feat_name1 = col_a + '_' + col_b + '_' + 'tfidf'
            
            feat_name2 = col_a + '_' + col_b + '_' + 'bow'
            
            
            x1 = df[feat_name1]
            
            x2 = df[feat_name2]
            
            idx1 = x1 > 0

            
            mm1 = x1[idx1].mean()
            
            
            idx1num = np.array(map(lambda x: int(x),idx1))
            
            idxabovemean1 = x1 > mm1
            
            idxabovemean1num = np.array(map(lambda x: int(x),idxabovemean1))
             
            feat_on1 = feat_name1+'ison'
            
            df[feat_on1] = idx1num
            
            feat_on1very = feat_name1+'isreallyon'
            
            df[feat_on1very] = idxabovemean1num
            
            accum_score = accum_score + idx1num + 2*idxabovemean1num
            
            feat_sub = feat_name1 + 'sub' + feat_name2
            
            feat_sum = feat_name1 + 'add' + feat_name2
            
            df[feat_sub] = x1 - x2
            
            df[feat_sum] = x1 + x2
     
    df['tfidf_accum_score'] = accum_score
       
    return df
    
def find_new_features(df):
    new_features = [ \
            name for name in df.columns \
            if "sub" in name \
            or "add" in name \
            or "ison" in name \
            or "isreallyon" in name \
            or "accum" in name
            ]       
    return new_features
    
def find_drop_features(df):
    new_features = [ \
            name for name in df.columns \
            if "sub" in name \
            or "add" in name 
            ]       
    return new_features
    
    
def find_basic_features(df):
    new_features = [ \
            name for name in df.columns \
            if 'ratio' in name \
            or 'count' in name \
            or 'div' in name \
            or 'between' in name \
            or 'pos_of' in name \
            ]       
    return new_features
    
if __name__ == "__main__":
    
    ###### loading all feature files ########
    
      
    
    train_tfidf_files = list()
    train_tfidf_files.append(pp.output_root_dir + 'Train_tfidf_bow_brand.csv')
    
    dfTrainTFIDF = append_dataframes(train_tfidf_files)
    
    test_tfidf_files = list()
    test_tfidf_files.append(pp.output_root_dir + 'Test_tfidf_bow_brand.csv')    
    
    dfTestTFIDF = append_dataframes(test_tfidf_files)

    
    vec_types = ['tfidf','bow']
    
    tfidf_pairs = [ ['search_term','product_title'], \
                    ['search_term','product_description'], \
                    ['search_term','attribute_description'], \
                    ['Synonym','product_title'], \
                    ['Synonym','product_description'], \
                    ['Synonym','attribute_description']
                  ]
    
    dfTrainTFIDF = redefine_tfidf_feats(dfTrainTFIDF)
    
    dfTestTFIDF = redefine_tfidf_feats(dfTestTFIDF)
    
    newft_train = find_new_features(dfTrainTFIDF)
    
    newft_test = find_new_features(dfTestTFIDF)
    
    newft_train == newft_test
    
    newft = copy(newft_train)
    
    newft.append('id')
    
    dfTestTFIDF = dfTestTFIDF[newft]
    
    newft.append('relevance')
    
    dfTrainTFIDF = dfTrainTFIDF[newft]
    
    
    ####################################
    
    new_feats = find_new_features(dfTrainTFIDF)
    
    new_feats.extend(['id','relevance'])
    
    dfTrainTFIDF[new_feats].to_csv(pp.output_root_dir + 'Train_tfidf_subadd_clean.csv',index=False)
    
    new_feats = find_new_features(dfTestTFIDF)
    
    new_feats.append('id')
    
    dfTestTFIDF[new_feats].to_csv(pp.output_root_dir + 'Test_tfidf_subadd_clean.csv',index=False)
    
    
    drop_feats = find_drop_features(dfTrainTFIDF)
    
    dfTrainTFIDF.drop(drop_feats,axis=1).to_csv(pp.output_root_dir + 'Train_tfidf_org.csv',index=False)
    
    drop_feats = find_drop_features(dfTestTFIDF)
    
    dfTestTFIDF.drop(drop_feats,axis=1).to_csv(pp.output_root_dir + 'Test_tfidf_org.csv',index=False)
    
    
    dfTrainTFIDF.to_csv(pp.output_root_dir + 'Train_tfidf_all.csv')
    
    dfTestTFIDF.to_csv(pp.output_root_dir + 'Test_tfidf_all.csv')
    ###############################################
    
    
    len(dfTrainTFIDF.columns.values)    
    
    len(dfTestTFIDF.columns.values)   
    
    train_files = list()
    train_files.append(pp.output_root_dir + 'Train1.csv')
    train_files.append(pp.output_root_dir + 'Train2.csv')
    
    dfTrain_rest = append_dataframes(train_files)
    
    test_files = list()
    test_files.append(pp.output_root_dir + 'Test1.csv')
    test_files.append(pp.output_root_dir + 'Test2.csv')
    test_files.append(pp.output_root_dir + 'Test3.csv')
    test_files.append(pp.output_root_dir + 'Test4.csv')
    
    dfTest_rest = append_dataframes(test_files)
               
    #good_col_dict= parse_textfile_todict('/Users/arman/Dropbox/kaggle/homedepot/goodColumns.txt')
    
    #good_columns = good_col_dict.keys()
    
    #good_columns.append('id')
    
    basic_feats_train = find_basic_features(dfTrain_rest)
    
    basic_feats_test = find_basic_features(dfTest_rest)
    
    basic_feats_train == basic_feats_test    
    
    basic_feats = copy(basic_feats_train)
    
    basic_feats.append('id')
    
    len(dfTrainTFIDF.columns.values)    
    
    len(dfTestTFIDF.columns.values)
    
    len(dfTrain_rest[basic_feats].columns.values)
    
    len(dfTest_rest[basic_feats].columns.values)
    
    'product_uid' in (dfTest_rest[basic_feats].columns.values)
    
    
    dfTrainTFIDF = pd.merge(dfTrainTFIDF,dfTrain_rest[basic_feats],on='id')
    
    del dfTrain_rest
    
    dfTestTFIDF = pd.merge(dfTestTFIDF,dfTest_rest[basic_feats],on='id')
    
    
    #dfTrainTFIDF = dfTrainTFIDF.drop('product_uid',axis=1)
    
    #dfTestTFIDF = dfTestTFIDF.drop('product_uid',axis=1)
    
    
    len(dfTrainTFIDF.columns.values)
    
    len(dfTestTFIDF.columns.values)
    
    train = dfTrainTFIDF.copy()
    
    del dfTrainTFIDF
    
    test = dfTestTFIDF
    
    
    train.to_csv(pp.output_root_dir + 'lastchance_train.csv',index=False)
    
    test.to_csv(pp.output_root_dir + 'lastchance_test.csv',index=False)
    
    features = list(train.columns.values)
    
    
    features.remove('relevance')
    
    features == list(test.columns.values)
    
    features.remove('id')
    
    
    #XGboost params: (set for quick run, not high accuracy)
    params = {"objective": "reg:linear",
              "booster" : "gbtree",
              "eta": 0.02,
              "max_depth": 12,
              "subsample": 0.8,
              "colsample_bytree": 0.4,
              "silent": 1,
              "thread": 2,
              "seed": 2015,
              "min_child_weight": 8
              }
    # For the final run should try 1000+ iteration at eta < 0.02
    num_boost_round = 600
    
    
    X_train, X_valid = train_test_split(train, test_size=0.1, random_state=10)
    y_train = np.array(X_train['relevance'])
    y_valid = np.array(X_valid['relevance'])
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)
    
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    
    def rmse(y, yhat):
        return np.sqrt(np.mean((yhat - y) ** 2))
    
    def rmse_xg(yhat, y):
        y = np.array(y.get_label())
        yhat = np.array(yhat)
        return "rmse", rmse(y,yhat)
    
    # Training the tree:
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,
                    early_stopping_rounds=50, feval=rmse_xg, verbose_eval=True)
    
    dtest = xgb.DMatrix(test[features])
    
    test_preds = gbm.predict(dtest)    
    
                  
    
    
    
    


