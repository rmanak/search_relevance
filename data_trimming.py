# -*- coding: utf-8 -*-
"""

__file__

    data_cleaning_basics.py
    
__description__

     Data Cleaning:
     - removing stopwords
     - removing numbers
     - removing measuring units
     - removing model numbers
     - removing punctuations

    
__author__

    Hamid Omid
    
"""

from __future__ import print_function
import pandas as pd
from nltk.corpus import stopwords
import re
import sys
import project_params as pp




stops = stopwords.words('english')



def prog():
    print(".",end='')
    sys.stdout.flush()
    
def longprog():
    print("....",end='')
    sys.stdout.flush()
    
    

stops =  stopwords.words('english')+["ft" , "lb" , "lbs" , "cm" , "gal" , "inch" \
, "kg", "sq\\.ft", "gallon" , "meter" , "feet" , "foot" , "watt" , "cu\\.ft" , "watts"\
 , "volt" , "volts" , "acre" , "kw" , "ounce" , "ounces" , "ml" , "liter" , "ml" \
 , "oz" , "sq" , "hours" , "hour" , "cu" , "mm" , "ps" , "cc" , "x" , "amp" , "years" \
 , "year" , "in" , "In" , "It" , "The" , "This" , "That" , "and" , "And" , "if" ,\
 "If" ,"those" , "these" , "Those" , "These" , "I" , "Its", "We", "He", "She", \
 "You" ,"Itself" , "Gal" , "Inch" , "INCH" , "All" , "But" , "Ft", "FT" ,\
 " Lbs" , "Cm" , "Kg" , "Amp" , "Cu" , "Oz" , "Volt" , "Watts" , "Ml" , "Acre" ,\
 "Kw" , " Ounce" , "Ounces" , "Liter" , "Sq" , "Years" , "Feet" , "Meter" , "CU"\
 , "Ps" , "Cc" , "CC" , "thats" , "can" , "could" , "Do" , "DO" , "For" , "ll"\
 , "did" , "as" , "As", "maybe" , "perhaps" , "may " , "WARNING" , "warning" \
 , "Warning" , "To" , "to" , "where" , "Where" , "Which" , "which" , "whose" \
 , "Whose" , "What" , "what" , "fl" , "Fl" , "SC" , "BTU"]
 



def remove_stopwords(s):
    y = ''
    for word in s.split():
        if word not in stops:
            y = y + word + ' '
    return y.rstrip()   
    
def remove_numbers(s):
    regex = "\\(|\\)|\\."
    key = s
    key = re.sub( regex , ' ', key )
    regex = "[^ ]*\\d+[^ ]*"
    key = re.sub( regex , '', key )
    return key.strip()
         
    
def trim(sentence)    :
    
    regex = "\\.|\\-|\\,|\\'|\\`|\\&|\\#|\\$|\\;|\\/|\\:|\\*|\\+|[[:punct:]]+"
    key = sentence
    key = re.sub( regex , ' ', key )
    regex = '\\"'
    key = re.sub( regex , ' ', key )
    key = remove_stopwords( key )
    key = remove_numbers( key )
    regex = '\\b\\w{1}\\b'
    key = re.sub( regex , ' ', key )
    regex = '[ ]{2,}'
    key = re.sub( regex , ' ', key )
    

    return key.strip()



if __name__ == '__main__':


    ######### reading csv files #############
    print("Loading data.",end='')
    
    
    dfTrain = pd.read_csv(pp.train_raw_file,encoding=pp.encoding); prog()
    
    dfTest = pd.read_csv(pp.test_raw_file,encoding=pp.encoding); prog()
    
    dfAttribute = pd.read_csv(pp.attribute_raw_file,encoding=pp.encoding); prog()
    
    dfProdDescription = pd.read_csv(pp.description_raw_file,encoding=pp.encoding); prog()
        
    print("Done.")
    
    ######## trimming data ###############
    
    print("Trimming data.",end='')



    dfProdDescription['product_description_trimmed'] = dfProdDescription['product_description'].map(trim); longprog()

    dfTrain['product_title_trimmed'] = dfTrain['product_title'].map(trim); longprog()
    dfTrain['search_term_trimmed'] = dfTrain['search_term'].map(trim); longprog()

    dfTest['product_title_trimmed'] = dfTest['product_title'].map(trim); prog()
    dfTest['search_term_trimmed'] = dfTest['search_term'].map(trim)

    print("Done.")


    ###### Saving data ########                                           
    
    print("Saving data.",end='')

    dfTrain.to_csv(pp.output_root_dir +  'Train_trimmed'+ '.csv',index=False); prog()                                  
      
    dfTest.to_csv(pp.output_root_dir +  'Test_trimmed'+ '.csv',index=False); prog()    

    dfProdDescription.to_csv(pp.output_root_dir +  'Product_descriptions_trimmed'+ '.csv',index=False)                                  
                              
    print("Done.")
    
