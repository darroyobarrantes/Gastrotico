# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 00:24:43 2022

@author: fafec
"""
import string
import pandas as pd
from fuzzywuzzy import fuzz

termBusqueda = pd.read_csv("ListaC.csv", encoding='utf8')
tm=termBusqueda.to_numpy().tolist()
tem=[]
for t in tm:
    tem.append(t[0])
j=0

for t in tem:
    k=0
    i=0
    for l in tem:
        if fuzz.ratio(t.lower(), l.lower()) >= 95:
            
            if i==0:
                i+=1    
            else:
                print(t, j," ... ",k, l)
        k+=1
    j+=1