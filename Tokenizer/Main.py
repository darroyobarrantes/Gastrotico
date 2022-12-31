# -*- coding: utf-8 -*-

#
############Instalar Spacy#################
#pip install -U pip setuptools wheel
#pip install -U spacy
#python -m spacy download es_core_news_sm
#
"""
Created on Tue Oct  4 16:32:08 2022

@author: Fabián Fernández Chaves
"""


from nltk import SnowballStemmer
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import numpy as np
import pandas as pd
import limpieza as lp
import Unir as un
import Procesos as pro

from progress.bar import Bar
import time

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from imblearn.over_sampling import RandomOverSampler 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
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
from xgboost import XGBClassifier

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

import matplotlib.pyplot as plt
import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
from collections import Counter
from matplotlib import pyplot


#Pickle model for later use
import pickle


spanishstemmer=SnowballStemmer("spanish")

##Guardar las reseñas unidas para no realizar el proceso cada vez
resena= un.unirResenas()
#resena.to_csv('Datos_Procesados/Reseñas_bloque.csv')  


#Abrir las reseñas previamente guardadas
#resena=pd.read_csv("Datos_Procesados/Reseñas_bloque.csv", index_col=0, encoding='utf8')

#cargar comentarios de entrenamiento
comentarios_traning=pd.read_csv("comentarios_train.csv", index_col=0, encoding='utf8')

#eliminar comentarios en ingles de las resenas
resena= lp.eliminarEng(resena)
#resena.to_csv('Datos_Procesados/Reseñas_sinIngles.csv')  

#eliminar comentarios de training de los comentarios totales y tomar las filas 
#necesarias para los procesos
resena_preProce= pro.elm_com_traning()
    
#resena_preProce.to_csv('Datos_Procesados/Reseñas_preproce.csv')  



#-----------------------------------------------------------------------------
#agregar columna de calificación comentario a traning
cr=pd.read_csv("Muestra_original.csv", index_col=0, encoding='utf8')
comentarios_traning_preProce=pro.add_calComen(cr,comentarios_traning)

#comentarios_traning_preProce.to_csv('Datos_Procesados/Comenta_Trai_PreProce.csv')  


#procesar datos de entrenamiento
comentarios_traning_preProce= lp.eliminarPunctuation(comentarios_traning_preProce)
comentarios_traning_preProce= lp.formato(comentarios_traning_preProce)
#comentarios_traning_preProce.to_csv('Datos_Procesados/Comenta_Trai_Proce.csv') 
comentarios_traning_preProce=pd.read_csv("Datos_Procesados/Comenta_Trai_Proce.csv", index_col=0, encoding='utf8')
lista_tokens_comenta= lp.normalize(comentarios_traning_preProce)
#restablecer el índice
lista_tokens_comenta.reset_index(drop=True, inplace=True)


#procesar datos completos
#resena_preProce=pd.read_csv("Datos_Procesados/Reseñas_preproce.csv", index_col=0, encoding='utf8') 
resena_preProce= lp.eliminarPunctuation(resena_preProce)
resena_preProce = lp.cambiarCalificacion(resena_preProce)
resena_preProce= lp.formato(resena_preProce)
resena_preProce= lp.quitarEmoji(resena_preProce)
#resena_preProce.to_csv('Datos_Procesados/resena_Proce.csv')
resena_preProce=pd.read_csv("Datos_Procesados/resena_Proce.csv", index_col=0, encoding='utf8') 
lista_tokens_comenta_resena= lp.normalize(resena_preProce)
#restablecer el índice
lista_tokens_comenta_resena.reset_index(drop=True, inplace=True)


lista_total_tokens=[]
va=[]
for indice_fila, fila in lista_tokens_comenta.iterrows():
    va=fila['Comentario']
    lista_total_tokens+=va
for tok in lista_total_tokens:
    if len(tok)>=2:
        lista_total_tokens.remove(tok)

BOW = FreqDist(lista_total_tokens)
len(BOW)


# listing the 6000 most frequent words (around half of the total words, that is XXXXXX)
word_features = list(BOW.keys())[:2900]


#---eliminar los tokens menores a tres letras
new_lista_tokens_comenta=list(lista_tokens_comenta["Comentario"])
for ltc in new_lista_tokens_comenta:
    
    for l in ltc:
        if len(l) <=2 or l not in word_features:
            ltc.remove(l)
            


##Agregar un metodo que elimine los que tienen menos de 5 palabras
new_lista_tokens_comenta=lp.eliminarPeque(new_lista_tokens_comenta)
#devolver la normalizacion al formato de lista
lista_tokens_comenta=lp.transLista(lista_tokens_comenta)
lista_tokens_comenta_resena=lp.transLista(lista_tokens_comenta_resena)

#------Eliminar de featuresets las filas que no contengan la cantidad adecuada de tokens
lista_tokens_comenta=pro.elim_features_peque(lista_tokens_comenta, new_lista_tokens_comenta)
#restablecer el índice
lista_tokens_comenta.reset_index(drop=True, inplace=True)

#
def find_features(document):
    words= word_tokenize(document)
    features={}
    for w in word_features:
        features[w]=(w in words)
    return features

#-----------------------------------------------------------------------------
documents = lista_tokens_comenta['Comentario'].tolist()
featuresets=[find_features(comment) for comment in documents]
featuresets= pd.DataFrame.from_dict(featuresets).astype(float)

foodTypes =  lista_tokens_comenta['Tipo_Comida'].str.get_dummies(sep=',')

featuresets=featuresets.join(foodTypes)
featuresets=featuresets.join(lista_tokens_comenta['Calificacion_Comentario'])

#featuresets.to_csv("featuresets.csv")

X = featuresets
#featuresets.to_csv("featuresets.csv")

y = lista_tokens_comenta['Calificacion']

di = {"positive": 1, "negative": 0}
y = y.replace(di)


# split data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

#Pickling for later use
with open("word_features.txt", "wb") as fp:
    pickle.dump(word_features, fp)

#Pickling for later use
col_train = X_train.columns.tolist()
with open("col_train.txt", "wb") as fp:
    pickle.dump(col_train, fp)


# Hacer un voting ensemble, que es un modelo compuesto.
# Set random seed
np.random.seed(42)
SEED_RANDOM = 42
n_estimators = 50

def get_voting():
    # define the base models
    models = list()
    #models.append(('log_model', LogisticRegression(max_iter=10000)))
    models.append(('svc_model', LinearSVC(dual=False)))
    models.append(('svc1_model', SVC(kernel="linear", C=0.025, probability=True)))
    models.append(('svc2_model', SVC(gamma=2, C=1, probability=True)))
    models.append(('dtr_model', DecisionTreeClassifier(max_depth=None, min_samples_split=2)))
    models.append(('rfc_model', RandomForestClassifier(min_samples_split=2)))
    #models.append(('erf_model', ExtraTreesClassifier(n_estimators=n_estimators, max_depth=None, min_samples_split=2)))
    models.append(('mnb_model', MultinomialNB()))
    models.append(('gnb_model', GaussianNB()))
    models.append(('bnb_model', BernoulliNB()))
    #models.append(('adaboost_model', AdaBoostClassifier(n_estimators=n_estimators)))
    #models.append(('cbc_model', CatBoostClassifier())) #Too slow
    models.append(('gbc_model', GradientBoostingClassifier(n_estimators=n_estimators)))
    models.append(('xgb_model', XGBClassifier(objective="binary:logistic", random_state=SEED_RANDOM, n_estimators = n_estimators, max_depth=2, use_label_encoder=False)))  
    # define the voting ensemble
    ensemble = VotingClassifier(estimators=models, voting='hard')
    return ensemble




# Hacer modelos individuales

# get a list of models to evaluate
def get_models():

    models = dict()   
    #models['log_model'] = LogisticRegression(max_iter=10000)
    models['svc_model'] = LinearSVC(dual=False)
    models['svc1_model'] = SVC(kernel="linear", C=0.025, probability=True)
    models['svc2_model'] = SVC(gamma=2, C=1, probability=True)
    models['dtr_model'] = DecisionTreeClassifier(max_depth=None, min_samples_split=2)
    models['rfc_model'] = RandomForestClassifier(min_samples_split=2)
    #models['erf_model'] = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=None, min_samples_split=2)
    models['mnb_model'] = MultinomialNB()
    models['gnb_model'] = GaussianNB()
    models['bnb_model'] = BernoulliNB()
    #models['adaboost_model'] = AdaBoostClassifier(n_estimators=n_estimators)
    #models['cbc_model'] = CatBoostClassifier()
    models['gbc_model'] = GradientBoostingClassifier(n_estimators=n_estimators)
    models['xgb_model'] = XGBClassifier(objective="multi:softprob", random_state=SEED_RANDOM, n_estimators = n_estimators, max_depth=2, use_label_encoder=False)
    models['voting_ensemble'] = get_voting()
    return models



# Adjuntar todos los modelos
models = get_models()


# ### 3.5 Evaluación de Modelos


# Define evaluation metrics
scoring='f1_weighted' #f1_weighted es la medida de evaluación 

# Evaluar cada modelo
def evaluate_model(model, X_train, y_train):#train_y
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1) #Repeat with n_splits = 10 and n_repeats_10 if memory allows it
    scores = cross_val_score(model, X_train, y_train, scoring=scoring, cv=cv)
    return scores


# Crea tabla de evaluacion
#del list
results, names = list(), list()
columns = list(models.keys())
data = []
score_model = pd.DataFrame()


#Aplica evaluacion a cada modelo
for name, model in models.items():
    scores = evaluate_model(model, X_train,y_train)#train_y
    results.append(scores)
    names.append(name)
    zipped = zip(columns, scores)
    a_dictionary = dict(zipped)
    data.append(a_dictionary)
    #print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))



#Genera tabla final con resultados
score_model = score_model.append(data, True)
score_model.loc['mean'] = score_model.mean()
score_model = score_model.iloc[-1:]
score_model = score_model.astype(float)
score_model['Mejor resultado'] = score_model.idxmax(axis=1)
score_model = score_model.rename(index={'mean': str(scoring)})


score_model.to_csv('score_model.csv', index=False)



### Selecciona el mejor resultado
#bestmodel= models[score_model['Mejor resultado'].iloc[0]]
bestmodel = models['rfc_model']#rfc_model


#Entrena el modelo con el mejor resultado
bestmodel.fit(X_train, y_train)#train_y


#salva modelo a disco
filename = 'bestmodel.sav'
pickle.dump(bestmodel, open(filename, 'wb'))


# X_test.drop("prediction", axis=1, inplace=True)
#Predice resultados para X_test
prediction = bestmodel.predict(X_test)


X_test["prediction"]=prediction

#Nos dice la proporcion de positivos y negativos en %
X_test['prediction'].value_counts(normalize=True)* 100


#Inspect Feature importance
importance = bestmodel.feature_importances_

#summarize feature importance
feat_importances = pd.Series(bestmodel.feature_importances_, index=X_train.columns)
feat_importances.nlargest(25).sort_values(axis=0, ascending=True).plot(kind='barh')
plt.title("Top 25 variables con más peso")
plt.show()


#Identificar clases para matriz de confusion
classes = y_test.unique().tolist()


# Plot non-normalized confusion matrix to check how the model performs over seach label
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test.astype(str), X_test["prediction"].astype(str))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=classes)
disp.plot(xticks_rotation='vertical')
plt.show()


X_test.to_pickle("./X_test.pkl")
#del X_train, y_train, y_test



# Cargar modelo y datos de ese modelo guardados arriba
import pickle
from sklearn.preprocessing import LabelEncoder

filename = 'bestmodel.sav'
loaded_model = pickle.load(open(filename, 'rb'))

X_test = pd.read_pickle("./X_test.pkl")

#label_file = open('label_encoder.pkl', 'rb')
#label_encoder = pickle.load(label_file) 
#label_file.close()

with open("word_features.txt", "rb") as fp:
    word_features = pickle.load(fp)


# Crea tabla columnas con palabras del comentario
documents_new = lista_tokens_comenta_resena['Comentario'].tolist()
featuresets_new = [find_features(comment) for comment in documents_new]
featuresets_new = pd.DataFrame.from_dict(featuresets_new)
featuresets_new = featuresets_new.astype(float)


foodTypes_new =  lista_tokens_comenta_resena['Tipo_Comida'].str.get_dummies(sep=',')

featuresets_new=featuresets_new.join(foodTypes)
featuresets_new=featuresets_new.join(lista_tokens_comenta_resena['Calificacion_Comentario'])
#replace los campos de tipo de comida que no contienen
featuresets_new = featuresets_new.replace('nan', 0)
featuresets_new = featuresets_new.fillna(0)



df_new_X_test=featuresets_new

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


#Si causa poblemas, revisar que prediction o prediction2 no esten en df_join o eliminarlos
df_new_X_test = df_new_X_test.drop(['prediction'], axis=1)


#Realizar Predicción
prediction = loaded_model.predict(df_new_X_test)

#Reverse Label Encoding
#prediction = label_encoder.inverse_transform(prediction)
df_new_X_test['prediction'] = prediction#.astype(str)
#Ver la distribución
df_new_X_test['prediction'].value_counts(normalize=True)


#Concatenar (pegar uno abajo de otro los dos archivos: ariginal con 987 reseñas con la Calificación_Comentario original, 
# y las nuevas reseñas con el la Calificación_Comentario predicho)

# comentarios de training
df1=pd.DataFrame(columns=["Comentario"] )
df1["Comentario"]=comentarios_traning_preProce['Comentario']
df1=df1.join(foodTypes)
df1=df1.join(comentarios_traning_preProce['Calificacion'])
df1=df1.join(comentarios_traning_preProce['Calificacion_Comentario'])
#replace los campos de tipo de comida que no contienen
df1 = df1.replace('nan', 0)
df1 = df1.fillna(0)
di = {"positive": 1, "negative": 0}
df1['Calificacion'] = df1['Calificacion'].replace(di)

df2=pd.DataFrame(columns=["Comentario"] )
df2["Comentario"]=resena_preProce['Comentario']
df2=df2.join(foodTypes_new)
df2=df2.join(df_new_X_test['prediction'])
df2=df2.join(resena_preProce['Calificacion_Comentario'])
#aqui te va la calificacion
#replace los campos de tipo de comida que no contienen
df2 = df2.replace('nan', 0)
df2 = df2.fillna(0)
df2 = df2.rename(columns={"prediction": "Calificacion"})


sentiment = pd.concat([df2,df1])
sentiment = sentiment.replace('nan', 0)
sentiment = sentiment.fillna(0)

sentiment.to_csv('results.csv', index=False)

#sentiment=pd.read_csv("results.csv", index_col=0, encoding='utf8')





