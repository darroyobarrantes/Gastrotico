# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 19:48:31 2022

@author: Fabián Fernández Chaves
"""

#Creates features
from nltk import word_tokenize

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    for w in word_features:
        features[w] = (w in words)
# Creating features for each comment
featuresets_new = [find_features(comment) for comment in documents_new]
featuresets_new = pd.DataFrame.from_dict(featuresets_new)
featuresets_new = featuresets_new.astype(float)



#Eliminate foreign characters:
from langdetect import detect
def det(x):
    try:
        lang = detect(x)
    except:
        lang = 'Other'
    return lang

df_new['language'] = df_new['Comment_Description'].apply(det)

#Drop lines
df_new = df_new.loc[(df_new['language'] != 'he') & (df_new['language'] != 'ko')]

#Drop temp column
df_new = df_new.drop('language', axis=1)
