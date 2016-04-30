# -*- coding: utf-8 -*-
"""
__file__

    project_params.py
    
__description__

     global project parameter variables
     
    
__author__

    Arman Akbarian
"""

encoding = 'ISO-8859-1'

output_root_dir = '/Users/arman/kaggleout/homedepot/'

input_root_dir = '/Users/arman/kaggledata/homedepot/'

########################################################

train_raw_file = input_root_dir + 'train.csv'

test_raw_file = input_root_dir + 'test.csv'

attribute_raw_file = input_root_dir + 'attributes.csv'

description_raw_file = input_root_dir + 'product_descriptions.csv'

synonyms_train_raw_file = input_root_dir + 'train_synonyms.csv'

synonyms_test_raw_file = input_root_dir  + 'test_synonyms.csv'


#######################################################

train_proccessed_file = output_root_dir + 'train_processed.pkl'

test_processed_file = output_root_dir + 'test_processed.pkl'

attribute_processed_file = output_root_dir + 'attributes_processed.pkl'

product_descriptions_processed_file = output_root_dir + 'product_descriptions_processed.pkl'

#########################################################

train_grams_file =  output_root_dir + 'train_grams.pkl'

test_grams_file = output_root_dir + 'test_grams.pkl'

product_descriptions_grams_file =  output_root_dir + 'product_descriptions_grams.pkl'

#########################################################

train_features_file = output_root_dir + 'train_features.pkl'

test_features_file = output_root_dir + 'test_features.pkl'

##########################################################



