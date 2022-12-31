# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 15:44:36 2022

@author: Fabián Fernández Chaves
"""
from fuzzywuzzy import fuzz
import pandas as pd
import csv
import numpy as np
import re
import string



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




lp = pd.read_csv("Reseñas.csv", encoding='utf8')
termBusqueda = pd.read_csv("ListaC.csv", encoding='utf8')


numMenciones = iniciarContador(len(termBusqueda))
listaComen = cargarComentarios()
n = 0
for coment in listaComen:

    n += 1
    comentario = coment[6]
    comentario = deEmojify(comentario)
    terminos = []
    i = 0
    print(n, "/", len(listaComen))
    for indice, fila in termBusqueda.iterrows():
        term = fila[0]
        if comparation(comentario, term):
            terminos.append(term)
            numMenciones[i] += 1
        i += 1
    if len(terminos) != 0:
        coment.append(terminos)
    else:
        coment.append("")










