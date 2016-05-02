'''
Detecting the best features among ~600 new features.

INPUT FILES: 
Train(i).csv (The new features of train set; made by Arman)
Test(j).csv  (the new features of test set; made by Arman)

OUTPUTS:
newFeatTrain.csv (a file that only has the most relevant features)
newFeatTest.csv (a file that only has the most relevant features)

__Authors__:
Ali Narimani, Hamid Omid

__Veresion__:
1.0
'''


import numpy as np
import pandas as pd

### Reading input data:
df_ts1 = pd.read_csv('../../homedepotdata/Test1.csv')
df_ts2 = pd.read_csv('../../homedepotdata/Test2.csv')
df_ts3 = pd.read_csv('../../homedepotdata/Test3.csv')
df_ts4 = pd.read_csv('../../homedepotdata/Test4.csv')

df_tr1 = pd.read_csv('../../homedepotdata/Train1.csv')
df_tr2 = pd.read_csv('../../homedepotdata/Train2.csv')

### concat train and test:
frames = [df_ts1,df_ts2,df_ts3,df_ts4]
TEST = pd.concat(frames, axis=0, ignore_index=True)

frames = [df_tr1,df_tr2]
TRAIN = pd.concat(frames, axis=0, ignore_index=True)

### Drop columns with less than `espislon` variation:
names = list(TEST.columns)
columns2drop = []
stdTr = {}
stdTs = {}
epsilon = 10**(-3) # this is a subjective choice, but we have no time
for column in names:
    sd = np.std(TEST[column])
    stdTs[column] = sd
    sd = np.std(TRAIN[column])
    stdTr[column] = sd
    if sd < epsilon:
        columns2drop.append(column)
TRAIN.drop(columns2drop,axis=1,inplace=True)
TEST.drop(columns2drop,axis=1,inplace=True)

### Drop columns that are correlated more than (1-eta):
names = TEST.columns
corrDrop = []
eta = 0.2    # this is a subjective choice, but we have no time
for c1 in range(len(names)):
    col1 = names[c1]
    for c2 in range(c1+1,len(names)):
        col2 = names[c2]
        buff = abs(np.corrcoef(TRAIN[col1],TRAIN[col2])[0,1])
        if buff > (1-eta) :
            corrDrop.append(col2)
TRAIN.drop(corrDrop,axis=1,inplace=True)
TEST.drop(corrDrop,axis=1,inplace=True)

### Detect columns with a higher than `delta` correlation with relevance:
names = TEST.columns
corrRelev = {}
goodCol = []
delta = 0.05  # this is a subjective choice, but we have no time
for c1 in range(len(names)):
    col = names[c1]
    buff = abs(np.corrcoef(TRAIN[col],TRAIN.relevance)[0,1])
    corrRelev[col] =  buff
    
    if buff > delta:
        goodCol.append(col)

### writing data files
trainNew = TRAIN[goodCol]
testNew = TEST[goodCol]

trainNew.to_csv('newFeatTrain.csv',index=False)
testNew.to_csv('newFeatTest.csv',index=False)

# End of Code
