# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 18:07:49 2022

@author: Fabián Fernández Chaves
"""

import pandas as pd
 
df2 = pd.read_csv("Hola.csv", header=None, 
names=["ID","Nombre","Calificación","Tipo_Comida","Direccion","Coordenadas","Comentario","Fecha_Comentario","Calificacion_Comentario"])
 
print(df2)