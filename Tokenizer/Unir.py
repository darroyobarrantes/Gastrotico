# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 10:43:24 2022

@author: fafec
"""

import os
import pandas as pd
from progress.bar import Bar
import time


def unirResenas():
        
    contenido = os.listdir("C:/Users/fafec/Gastroticos/Tokenizer/Reseñas")
    
    print(contenido)
    df= pd.DataFrame(columns=["ID","Nombre","Calificación","Tipo_Comida","Direccion","Coordenadas","Comentario","Fecha_Comentario","Calificacion_Comentario"] )
    
    for l in contenido:
        
        resena= pd.read_csv("reseñas/"+l, encoding='utf8')
        bar1 = Bar(l+':', max=len(resena))
        
        for indice_fila, fila in resena.iterrows():
            
            
            df.loc[len(df.index)] = fila
            time.sleep(0.000002)
            bar1.next()
        bar1.finish()
   
    return df

def muestraComentario(datos):
    comPos=datos[datos['Calificacion_Comentario'] == "positive"]
    comNeg=datos[datos['Calificacion_Comentario'] == "negative"]
    
    compos=comPos.sample(frac=1).reset_index(drop=True)
    comneg=comNeg.sample(frac=1).reset_index(drop=True)
    
    compos=compos.head(500)
    comneg=comneg.head(500)
    
    compos.to_csv('Muestra_positiva.csv')  
    comneg.to_csv('Muestra_negativa.csv')  
    
    
    

