'''
Reads the raw or feature files and performs xgb regression

INPUT FILES: 
Train_tfidf_org.csv
Test_tfidf_org.csv
newFeatTrain.csv
newFeatTest.csv
 + (first time only): 
train.csv
test.csv
product_descriptions.csv
attributes.csv
train_SynonymDropBox.csv
test_SynonymDropBox.csv
(It proved to be more convinient to combine the xgb and cleaning processes in one code
 and provoke cleaning only once)

OUTPUTS:
a `.csv` submission file

__Authors__:
Ali Narimani

__Veresion__:
6.0
'''


import matplotlib.pyplot as plt

import time
start_time = time.time()

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import pipeline, grid_search
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from spell_corr import *
from xgbFunctions import * # check this file for choosing the right `stemmer`

#### Keys : ####
doStem = True # Set this to True only for the first time
dateTag = 'Apr7'   # data tag for (reading/writing) the latest cleaned file
ResuleDateTag = 'Apr25' # the submission file date tag
################


if doStem:
    df_train = pd.read_csv('../data/train.csv', encoding="ISO-8859-1")
    df_test = pd.read_csv('../data/test.csv', encoding="ISO-8859-1")
    df_pro_desc = pd.read_csv('../data/product_descriptions.csv')
    df_attr = pd.read_csv('../data/attributes.csv')
    df_brand = df_attr.ix[df_attr.name == "MFG Brand Name", ["product_uid", "value"]].rename(columns={"value": "brand"})

    df_trainSyn = pd.read_csv('../data/aliCleaned/train_SynonymDropBox.csv',dtype={'Synonym':np.dtype(str)})
    df_testSyn = pd.read_csv('../data/aliCleaned/test_SynonymDropBox.csv',dtype={'Synonym':np.dtype(str)})

    df_train = pd.merge(df_train,df_trainSyn,on='id', how='left')
    df_test = pd.merge(df_test,df_testSyn,on='id', how='left')

    df_attr = df_attr.dropna()
    material = dict()
    df_attr['about_material'] = df_attr['name'].str.lower().str.contains('material')
    for row in df_attr[df_attr['about_material']].iterrows():
        r = row[1]
        product = r['product_uid']
        value = r['value']
        material.setdefault(product, '')
        material[product] = material[product] + ' ' + str(value)
    df_material = pd.DataFrame.from_dict(material, orient='index')
    df_material = df_material.reset_index()
    df_material.columns = ['product_uid', 'material']

    color = dict()
    df_attr['about_color'] = df_attr['name'].str.lower().str.contains('color')
    for row in df_attr[df_attr['about_color']].iterrows():
        r = row[1]
        product = r['product_uid']
        value = r['value']
        color.setdefault(product, '')
        color[product] = color[product] + ' ' + str(value)
    df_color = pd.DataFrame.from_dict(color, orient='index')
    df_color = df_color.reset_index()
    df_color.columns = ['product_uid', 'color']
	

    num_train = df_train.shape[0]
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_material, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_color, how='left', on='product_uid')
    df_all.Synonym.fillna('ForgetIt',inplace=True)

    df_all.fillna('', inplace=True)
    df_all.search_term.replace(spell_check_dict,inplace=True)

else:
    df_train = pd.read_csv('../data/train.csv', encoding="ISO-8859-1")
    num_train = df_train.shape[0]

if Snow:
    filename = '../data/aliCleaned/dfAll_' + dateTag + '_snowBall' + '.csv'
else:
    filename = '../data/aliCleaned/dfAll_' + dateTag + '.csv'

if doStem:
    df_all['search_term'] = df_all['search_term'].map(lambda x:str_stem(x))
    df_all['product_title'] = df_all['product_title'].map(lambda x:str_stem(x))
    df_all['product_description'] = df_all['product_description'].map(lambda x:str_stem(x))
    df_all['brand'] = df_all['brand'].map(lambda x:str_stem(x))
    df_all['Synonym'] = df_all['Synonym'].map(lambda x:str_stem(x))
    df_all['material'] = df_all['material'].map(lambda x: str_stem(x))
    df_all['color'] = df_all['color'].map(lambda x: str_stem(x))
    df_all.to_csv(filename,index=False)

else:
    df_all = pd.read_csv(filename, encoding="ISO-8859-1")
    print("reading file:",filename)


df_all['len_of_brand'] = df_all['brand'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_title'] = df_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)


df_all['query_in_title'] = df_all[['search_term','product_title']].apply(lambda x:str_whole_word(x['search_term'],x['product_title'],0),axis=1)
df_all['query_in_description'] = df_all[['search_term','product_description']].apply(lambda x:str_whole_word(x['search_term'],x['product_description'],0),axis=1)

df_all['word_in_title'] = df_all[['search_term','product_title']].apply(lambda x:str_common_word(x['search_term'],x['product_title']),axis=1)
df_all['word_in_description'] = df_all[['search_term','product_description']].apply(lambda x:str_common_word(x['search_term'],x['product_description']),axis=1)

## jaccard:
df_all['jaccard'] = df_all[['search_term','product_title']].apply(lambda x:jaccard(x['search_term'],x['product_title']),axis=1)

df_all['ratio_title'] = df_all['word_in_title']/df_all['len_of_query']
df_all['ratio_description'] = df_all['word_in_description']/df_all['len_of_query']
df_all['attr'] = df_all['search_term']+"\t"+df_all['brand']
df_all['word_in_brand'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['ratio_brand'] = df_all['word_in_brand']/df_all['len_of_brand']
df_brand = pd.unique(df_all.brand.ravel())

####color and material
df_all['word_in_color'] = df_all[['search_term','color']].apply(lambda x:str_common_word(x['search_term'],x['color']),axis=1)
df_all['word_in_material'] = df_all[['search_term','material']].apply(lambda x:str_common_word(x['search_term'],x['material']),axis=1)

#synonym :
df_all['len_of_Syn'] = df_all['Synonym'].map(lambda x:len(x.split())).astype(np.int64)
  

df_all['syn_in_title'] = df_all[['Synonym','product_title']].apply(lambda x:str_common_word(x['Synonym'],x['product_title']),axis=1)
df_all['syn_in_description'] = df_all[['Synonym','product_description']].apply(lambda x:str_common_word(x['Synonym'],x['product_description']),axis=1)

df_all['ratio_syn'] = df_all['syn_in_description']/df_all['len_of_Syn']
df_all.ix[df_all.Synonym == 'ForgetIt',['syn_in_title']] = -1

random.seed(2016)
d={}
i = 1
for s in df_brand:
    d[s]=i
    i+=1
df_all['brand_feature'] = df_all['brand'].map(lambda x:d[x])
df_all['search_term_feature'] = df_all['search_term'].map(lambda x:len(x))

### Reading the selected columns from Arman features:
tr_newFeat = pd.read_csv('../data/newFeatTrain.csv')
ts_newFeat = pd.read_csv('../data/newFeatTest.csv')
df_allNew = pd.concat((tr_newFeat, ts_newFeat), axis=0, ignore_index=True)

### Reading Cosine similarity 
tr_tfidf = pd.read_csv('../data/Train_tfidf_org.csv')
tr_tfidf.drop(['relevance'],axis=1,inplace=True)
ts_tfidf = pd.read_csv('../data/Test_tfidf_org.csv')
df_allTfidf = pd.concat((tr_tfidf, ts_tfidf), axis=0, ignore_index=True)

df_all = pd.merge(df_all,df_allNew,on='id')
df_all = pd.merge(df_all,df_allTfidf,on='id')


df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']
y_train = df_train['relevance'].values
X_train =df_train[:]
X_test = df_test[:]
print("--- Features Set: %s minutes ---" % round(((time.time() - start_time)/60),2))


XGBdemo = xgb.XGBRegressor(objective="reg:linear",seed= 1300,nthread= 3)
tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
tsvd = TruncatedSVD(n_components=100, random_state = 2016)
clf = pipeline.Pipeline([
        ('union', FeatureUnion(
                    transformer_list = [
                        ('cst',  cust_regression_vals()),  
                        ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='search_term')), \
                                                    ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                        ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')),\
                                                    ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                        ('txt3', pipeline.Pipeline([('s3', cust_txt_col(key='product_description')),\
                                                    ('tfidf3', tfidf), ('tsvd3', tsvd)])),
                        ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf),\
                                                    ('tsvd4', tsvd)])),
                        ('txt5', pipeline.Pipeline([('s5', cust_txt_col(key='Synonym')), \
                                                    ('tfidf5', tfidf), ('tsvd5', tsvd)]))
                        ],
                    transformer_weights = {
                        'cst': 1.0,
                        'txt1': 0.5,
                        'txt2': 0.25,
                        'txt3': 0.0,
                        'txt4': 0.5,
                        'txt5': 1.0
                        },
                n_jobs = -1
                )), 
        ('xgb', XGBdemo)])

params = {'xgb__n_estimators': [550], 'xgb__max_depth': [8], 'xgb__subsample': [0.9], 'xgb__min_child_weight': [8], 'xgb__learning_rate': [0.03], \
	'xgb__colsample_bytree': [0.8]}

model = grid_search.GridSearchCV(estimator = clf, param_grid = params, n_jobs = 1, cv = 2, verbose = 20,\
                                 scoring=RMSE)
model.fit(X_train, y_train)

print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)

y_pred = model.predict(X_test)

for i in range(len(y_pred)):
	if y_pred[i]>3:
		y_pred[i] = 3
	if y_pred[i]<1:
		y_pred[i] = 1

SubFileName = './submissions/XGboostGS2' + ResuleDateTag + '.csv'
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(SubFileName,index=False)
print("--- Training & Testing: %s minutes ---" % round(((time.time() - start_time)/60),2))

#End of Code
