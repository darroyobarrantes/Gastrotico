# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 15:01:42 2022

@author: fafec
"""
import os
import re
import csv
import time
import string
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from progress.bar import Bar



def deEmojify(text):#Eliminar caracteres esppeciales 
    regrex_pattern = re.compile(pattern="["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                "]+", flags=re.UNICODE)

    punct = string.punctuation
    for c in punct:
        text = text.replace(c, " ")

    return regrex_pattern.sub(r'', text)


def iniciarContador(index):#Inicia el array del contador de terminos*apariciones
    return np.zeros((index,), dtype=int)


def comparation(comentario, term):#Comparacion de terminis y comentarios
    termFrag, comenFrag = fragmentar(term, comentario)

    cantTerm = len(termFrag)
    cantCom = len(comenFrag)-cantTerm

    for i in range(cantCom):
        texto = ""
        com = comenFrag[i:i+cantTerm]
        for j in com:
            texto += j+" "

        if fuzz.ratio(texto.lower(), term.lower()) >= 85:
            return True

def fragmentar(termino, comentario):#Fragmenta 
    termFrag = termino.split()
    comenFrag = comentario.split()
    return termFrag, comenFrag


def cargarComentarios():
    with open("Reseñas.csv", 'r', encoding="utf8") as file_name:
        file_read = csv.reader(file_name)
        listaComen = list(file_read)
        listaComen.pop(0)
    return listaComen

def unirResenas(term):
        
    contenido = os.listdir("Reseñas/Central")
    
    print(contenido)
    df= pd.DataFrame(columns=["Zona","Comentario"])
    dif=pd.DataFrame(columns=term)
    arrayCeros=np.zeros((len(term),), dtype=int)
    
    for l in contenido:
        
        resena= pd.read_csv("Reseñas/Central/"+l, encoding='utf8')
        bar1 = Bar(l+':', max=len(resena))
        
        for indice_fila, fila in resena.iterrows():
            
            
            zona=l[7:-4]
            comen=fila["Comentario"]
            term=[]
            entrada=[zona,comen]
            df.loc[len(df.index)] = entrada
            dif.loc[len(dif.index)] = arrayCeros
            time.sleep(0.000002)
            bar1.next()
        bar1.finish()
    dtf=pd.concat([df, dif], axis=1)
    return dtf