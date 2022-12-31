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
import pandas as pd
from progress.bar import Bar
import time

def elm_com_traning(resena, comentarios_traning):
    resena_preProce= pd.DataFrame(columns=["Tipo_Comida","Calificación","Comentario","Calificacion_Comentario"] )
    bar1 = Bar("proceso", max=len(resena))
    for fila_index, fila in resena.iterrows():
        existe=True
        for filaD_index, filaD in comentarios_traning.iterrows():
            if fila['Comentario']==filaD['Comentario']:
                existe=False
                break
        if existe:
            resena_preProce.loc[len(resena_preProce.index)] = [fila['Tipo_Comida'],'nan',fila['Comentario'],fila['Calificacion_Comentario']]
        time.sleep(0.00000002)
        bar1.next()
    bar1.finish() 
    return resena_preProce

def add_calComen(cr,comentarios_traning):
    comentarios_traning_preProce= pd.DataFrame(columns=["Tipo_Comida","Calificacion","Comentario","Calificacion_Comentario"] )
    bar1 = Bar("proceso", max=len(cr))
    for fila_index, fila in cr.iterrows():
        existe=False
        for filaD_index, filaD in comentarios_traning.iterrows():
            if fila['Comentario']==filaD['Comentario']:
                existe=True
                break
        if existe:
            comentarios_traning_preProce.loc[len(comentarios_traning_preProce.index)] = [filaD['Tipo_Comida'],filaD['Calificacion_Comentario'],filaD['Comentario corregido'],fila['Calificacion_Comentario']]
        time.sleep(0.00000002)
        bar1.next()
    bar1.finish() 
    return comentarios_traning_preProce

def elim_features_peque(lista_tokens_comenta, new_lista_tokens_comenta):
    i=0
    for fila in lista_tokens_comenta["Comentario"]:
        text1=fila
        existe=True
        for  filaD in new_lista_tokens_comenta:
            text2=" ".join(filaD)
            if text1==text2:
                existe=False
                
        if existe:
            lista_tokens_comenta=lista_tokens_comenta.drop(i)
        i+=1
    return lista_tokens_comenta