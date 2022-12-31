#!/usr/bin/env python
# coding: utf-8

# ## 1.Library Import

# In[1]:


import os
os.environ['http_proxy'] = 'http://proxy-chain.intel.com:911'
os.environ['https_proxy'] = 'http://proxy-chain.intel.com:912'

#!pip install openpyxl
#!pip3 install xgboost
#!pip install -U imbalanced-learn
#!pip install gensim
#!pip install nltk
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
#!pip install sklearn
#!pip install langdetect


# Import required libraries
import numpy as np
import openpyxl
import re
from numpy import mean
from numpy import std
import pandas as pd
import pyodbc
from time import time
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
from collections import Counter
from matplotlib import pyplot
from langdetect import detect

# Import required libraries for machine learning classifiers and NLP
from sklearn import preprocessing
import gensim
from imblearn.under_sampling import RandomUnderSampler
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from imblearn.over_sampling import RandomOverSampler 
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
#from xgboost import XGBClassifier


#try:
  #from catboost import CatBoostClassifier
#except:
  #!pip install catboost
  #from catboost import CatBoostClassifier

from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict
from sklearn.feature_selection import SelectFromModel

# Import required libraries for performance metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix

#Pickle model for later use
import pickle


# ### 3.1 Feature Engineering

# In[23]:


#Pickling for later use
with open("word_features.txt", "wb") as fp:
    pickle.dump(word_features, fp)


# In[43]:


#Pickling for later use
col_train = X_train.columns.tolist()
with open("col_train.txt", "wb") as fp:
    pickle.dump(col_train, fp)


# ### 3.4 Classification Models

# In[48]:


# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# label_encoder = label_encoder.fit(y)
# label_encoded_y = label_encoder.transform(y)

# #exporting the departure encoder
# output = open('label_encoder.pkl', 'wb')
# pickle.dump(label_encoder, output)
# output.close()


# In[45]:


# Hacer un voting ensemble, que es un modelo compuesto.
# Set random seed
np.random.seed(42)
SEED_RANDOM = 42
n_estimators = 50

def get_voting():
    # define the base models
    models = list()
    #models.append(('log_model', LogisticRegression(max_iter=10000)))
    #models.append(('svc_model', LinearSVC(dual=False)))
    #models.append(('svc1_model', SVC(kernel="linear", C=0.025, probability=True)))
    #models.append(('svc2_model', SVC(gamma=2, C=1, probability=True)))
    models.append(('dtr_model', DecisionTreeClassifier(max_depth=None, min_samples_split=2)))
    models.append(('rfc_model', RandomForestClassifier(min_samples_split=2)))
    #models.append(('erf_model', ExtraTreesClassifier(n_estimators=n_estimators, max_depth=None, min_samples_split=2)))
    models.append(('mnb_model', MultinomialNB()))
    #models.append(('gnb_model', GaussianNB()))
    #models.append(('bnb_model', BernoulliNB()))
    #models.append(('adaboost_model', AdaBoostClassifier(n_estimators=n_estimators)))
    #models.append(('cbc_model', CatBoostClassifier())) #Too slow
    #models.append(('gbc_model', GradientBoostingClassifier(n_estimators=n_estimators)))
    #models.append(('xgb_model', XGBClassifier(objective="multi:softprob", random_state=SEED_RANDOM, n_estimators = n_estimators, max_depth=2, use_label_encoder=False)))  
    # define the voting ensemble
    ensemble = VotingClassifier(estimators=models, voting='hard')
    return ensemble


# In[46]:


# Hacer modelos individuales

# get a list of models to evaluate
def get_models():

    models = dict()   
    #models['log_model'] = LogisticRegression(max_iter=10000)
    #models['svc_model'] = LinearSVC(dual=False)
    #models['svc1_model'] = SVC(kernel="linear", C=0.025, probability=True)
    #models['svc2_model'] = SVC(gamma=2, C=1, probability=True)
    models['dtr_model'] = DecisionTreeClassifier(max_depth=None, min_samples_split=2)
    models['rfc_model'] = RandomForestClassifier(min_samples_split=2)
    #models['erf_model'] = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=None, min_samples_split=2)
    #models['mnb_model'] = MultinomialNB()
    models['gnb_model'] = GaussianNB()
    #models['bnb_model'] = BernoulliNB()
    #models['adaboost_model'] = AdaBoostClassifier(n_estimators=n_estimators)
    #models['cbc_model'] = CatBoostClassifier()
    #models['gbc_model'] = GradientBoostingClassifier(n_estimators=n_estimators)
    #models['xgb_model'] = XGBClassifier(objective="multi:softprob", random_state=SEED_RANDOM, n_estimators = n_estimators, max_depth=2, use_label_encoder=False)
    models['voting_ensemble'] = get_voting()
    return models


# In[47]:


# Adjuntar todos los modelos
models = get_models()


# ### 3.5 Evaluación de Modelos

# In[48]:


# Define evaluation metrics
scoring='f1_weighted' #f1_weighted es la medida de evaluación 

# Evaluar cada modelo
def evaluate_model(model, X, label_encoded_y):#train_y
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1) #Repeat with n_splits = 10 and n_repeats_10 if memory allows it
    scores = cross_val_score(model, X, label_encoded_y, scoring=scoring, cv=cv, n_jobs=-1, error_score='raise')
    return scores


# In[49]:


# Crea tabla de evaluacion
#del list
results, names = list(), list()
columns = list(models.keys())
data = []
score_model = pd.DataFrame()


# In[50]:


#Aplica evaluacion a cada modelo
for name, model in models.items():
    scores = evaluate_model(model, X,label_encoded_y)#train_y
    results.append(scores)
    names.append(name)
    zipped = zip(columns, scores)
    a_dictionary = dict(zipped)
    data.append(a_dictionary)
    #print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))


# In[51]:


#Genera tabla final con resultados
score_model = score_model.append(data, True)
score_model.loc['mean'] = score_model.mean()
score_model = score_model.iloc[-1:]
score_model = score_model.astype(float)
score_model['Mejor resultado'] = score_model.idxmax(axis=1)
score_model = score_model.rename(index={'mean': str(scoring)})


# In[52]:


score_model


# In[53]:


### Selecciona el mejor resultado
bestmodel= models[score_model['Mejor resultado'].iloc[0]]
#bestmodel = models['xgb_model']


# In[54]:


#Entrena el modelo con el mejor resultado
bestmodel.fit(X, label_encoded_y)#train_y


# In[55]:


#salva modelo a disco
filename = 'bestmodel.sav'
pickle.dump(bestmodel, open(filename, 'wb'))


# In[56]:


#Predice resultados para X_test
prediction = bestmodel.predict(X_test)

#Reverse Label Encoding
#prediction = label_encoder.inverse_transform(prediction)
#X_test['prediction'] = prediction.astype(str)

#See distribution
#X_test["prediction"].value_counts(normalize=True) * 100


# In[57]:


#Nos dice la proporcion de positivos y negativos en %
X_test['prediction'].value_counts(normalize=True)* 100


# In[59]:


#Inspect Feature importance
importance = bestmodel.feature_importances_

#summarize feature importance
feat_importances = pd.Series(bestmodel.feature_importances_, index=X.columns)
feat_importances.nlargest(25).sort_values(axis=0, ascending=True).plot(kind='barh')
plt.title("Top 25 variables con más peso")
plt.show()


# In[60]:


#Identificar clases para matriz de confusion
classes = y_test.unique().tolist()


# In[62]:


# Plot non-normalized confusion matrix to check how the model performs over seach label
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test.astype(str), X_test["prediction"].astype(str))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=classes)
disp.plot(xticks_rotation='vertical')
plt.show()


#plot_confusion_matrix(bestmodel, X_test, y_test, )


# In[63]:


X_test.to_pickle("./X_test.pkl")
del X_train, y_train, y_test


# # Predicción para el resto de las reseñas

# In[ ]:


# Primero hay que limpiar y seleccionar las reseñas en español que no estén en las 987? que se utilizan para entrenar el modelo
# A esos datos los llamé df_new


# In[ ]:


# Cargar modelo y datos de ese modelo guardados arriba
import pickle
from sklearn.preprocessing import LabelEncoder

filename = 'bestmodel.sav'
loaded_model = pickle.load(open(filename, 'rb'))

X_test = pd.read_pickle("./X_test.pkl")

label_file = open('label_encoder.pkl', 'rb')
label_encoder = pickle.load(label_file) 
label_file.close()

with open("word_features.txt", "rb") as fp:
    word_features = pickle.load(fp)


# In[ ]:


# Crea tabla columnas con palabras del comentario
documents_new = df_new['Comentario'].tolist()
featuresets_new = [find_features(comment) for comment in documents_new]
featuresets_new = pd.DataFrame.from_dict(featuresets_new)
featuresets_new = featuresets_new.astype(float)


# In[ ]:


#featuresets_new hay que unirlo con la columna de Calificacion (0-5) y con la de tipos de comida, a la cual hay que aplicarle get_dummies
#A ese nuevo df le llamaré df_new_X_test


# In[ ]:


##Se alinea el número de columnas del modelo original con el número de columnas de las nuevas reseñas
list = X_test.columns.difference(df_new_X_test.columns).tolist()
if len(list) > 0:
    df_new_X_test[list] = 0
else:
    pass
list2 = df_new_X_test.columns.difference(X_test.columns).tolist()
if len(list2) > 0:
    df_new_X_test = df_new_X_test.drop(list2, axis=1)
else:
    pass


# In[ ]:


#Si causa poblemas, revisar que prediction o prediction2 no esten en df_join o eliminarlos
df_new_X_test = df_new_X_test.drop(['prediction', 'prediction2'], axis=1)


# In[ ]:


#Realizar Predicción
prediction = loaded_model.predict(df_new_X_test)

#Reverse Label Encoding
#prediction = label_encoder.inverse_transform(prediction)
df_new_X_test['prediction'] = prediction.astype(str)
#Ver la distribución
df_new_X_test['prediction'].value_counts(normalize=True)


# In[ ]:


#Concatenar (pegar uno abajo de otro los dos archivos: ariginal con 987 reseñas con la Calificación_Comentario original, 
# y las nuevas reseñas con el la Calificación_Comentario predicho)
sentiment = pd.concat([df1, df_new_X_test])


# In[ ]:


#Saves file
results.to_csv('results.csv', index=False)

