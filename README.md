Product Search Relevance
========================

This is the source code for the HomeDepot's data science challenge on Kaggle
 [[+]](https://www.kaggle.com/c/home-depot-product-search-relevance).
 Our solution gives the RMSE score of 0.456 and scored at top 3.5% 
 spot in the leadboard. 

**Team:**

* Arman Akbarian
* Ali Narimani
* Hamid Omid

Overview of the ML pipeline:
---------------------------

![alt tag](https://github.com/rmanak/search_relevance/blob/master/img/ML_homedepot.png)

As you can see the feature engineering part involves 4 parts:

- basic features: at ``xgb.py`` before merging other features
- extended features: at ``feat_eng_extended.py`` 
- extended tf-idf: at ``feat_eng_tfidf.py`` (followed by ``model_xgb.py``) 
- basic tf-idf: done at the pipeline at ``xgb.py`` 

Text preprocessing is done by ``preprocessing.py``.

**Misc Files Description**:

- ``spell_corrector.py``: builds ``spell_corr.py`` a python dictionary for correcting spellings in search query.
- ``data_trimming.py``: a pre-cleaning process needed by ``spell_corrector.py`` and ``Synonyms.py``
- ``data_selection.py``: feature selection after extended features are built.
- ``nlp_utils.py``: a NLP utility funciton and wrapper classes for quick prototyping
- ``ngrams.py``: builts n-grams
- ``Synonyms.py``: finds the synonyms of the words in search query
- ``project_params.py``: sets global variables for the project, I/O directories etc...



Preparation:
-----------

In ``project_params.py`` edit the following to the correct path

     - ``input_root_dir``: the path to the original .csv files in your system

     - ``output_root_dir``: a path with few GB disk space avaiable for I/O

Requirements:
-------------

#### Get all Requirements
To do all of the following (1,2,3) you can simply use:

    ``make install``

#### Packages
To install the required packages execute:

    ``pip install -r requirements.txt``

The project needs the following python packages:
   -numpy
   -pandas
   -scikit-learn
   -nltk

You may also need to install nltk data:

  - in python:

    ``>>> nltk.download()``

  - or via command line:

    ``python -m nltk.downloader all``

  - or makefile will take take care of everythin:

    ``make install``

(Particularly ``nlp_utils`` uses WordNet data.)

### Data
The data, can be downloaded from the competition's homepage
 [[+]](https://www.kaggle.com/c/home-depot-product-search-relevance).


Initial Test:
-------------

run ``make testutils`` to check if things work!

*Note: This documentation and repo is under construction, I will hopefully clean up 
the repo and modify all of the scripts such that the pipeline works perfectly on a 
single machine*
