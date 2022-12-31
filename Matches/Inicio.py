# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 23:03:38 2022

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
import pandas as pd
import Metodos as mt
from progress.bar import Bar
import time



termBusqueda = pd.read_csv("ListaC.csv", encoding='utf8')
tm=termBusqueda.to_numpy().tolist()
tem=[]
for t in tm:
    tem.append(t[0])
    





contenido = os.listdir("Reseñas")



df= pd.DataFrame(columns=["Zona", "termino", "menciones"])
dif=pd.DataFrame(columns=tem)


for l in contenido:
    resena= pd.read_csv("Reseñas/"+l, encoding='utf8')
    zona=l[7:-4]
    numMenciones = mt.iniciarContador(len(termBusqueda))
    n = 0
    
    
    bar1 = Bar(zona+':', max=len(termBusqueda))
    for indice, fila in termBusqueda.iterrows():
        term = fila[0].lower()
        i = 0
        for indice_fila, fila in resena.iterrows():
    
    
            n += 1
            comentario = fila["Comentario"]
            comentario = mt.deEmojify(comentario)
            

            if mt.comparation(comentario, term):
                numMenciones[indice] += 1
        df.loc[len(df.index)]=[zona, term, numMenciones[indice]]    
        
        i += 1
        time.sleep(0.000002)
        bar1.next()
    bar1.finish()   
    
data_completa= pd.DataFrame(columns=['Region','ID','Nombre','Calificación','Tipo_Comida','Direccion','Coordenadas','Comentario','Fecha_Comentario','Calificacion_Comentario','Terminos_comida'])

for l in contenido:
    resena= pd.read_csv("Reseñas/"+l, encoding='utf8')
    zona=l[7:-4]
    
    
    bar1 = Bar(zona+':', max=len(resena))
    for indice_fila, filaR in resena.iterrows():
        list_terms=''

        for indice, fila in termBusqueda.iterrows():
            term = fila[0].lower()

    
            comentario = filaR["Comentario"]
            comentario = mt.deEmojify(comentario)
            

            if mt.comparation(comentario, term):
                list_terms+=','+term
        list_terms=list_terms.strip(',')
        data_completa.loc[len(data_completa.index)]=[zona,filaR["ID"],filaR["Nombre"],filaR["Calificación"],filaR["Tipo_Comida"],filaR["Direccion"],filaR["Coordenadas"],filaR["Comentario"],filaR["Fecha_Comentario"],filaR["Calificacion_Comentario"],list_terms]  
        
        
        time.sleep(0.00000002)
        bar1.next()
    bar1.finish()      
    
data_completa.to_csv("Data_completa.csv")


df.to_csv("match_terminos_localidad.csv")