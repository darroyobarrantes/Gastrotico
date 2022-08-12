# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:46:45 2022

@author: Fabián Fernández Chaves
"""

import csv

def escribir(documento, nombre):
    n="./documentos/"+nombre
    lista_datos=documento
    campos=["Año","Provincia","Consulta","Frecuencia","Categoria"]
    with open(n, 'w', newline = '',encoding='utf-8') as myFile:
        writer = csv.DictWriter(myFile,fieldnames=campos)
        writer.writeheader()
        for datos in lista_datos: 
            ano=datos[0]
            provincia=datos[1]
            consulta=datos[2]
            frecuencia=datos[3].replace("\n","")
            frecuencia=frecuencia.replace("\r","")
            frecuencia=frecuencia.replace("\r\n","")
            frecuencia=frecuencia.replace(" ","")
            categoria=datos[4]
            writer.writerow({"Año":ano,"Provincia":provincia,"Consulta":consulta,"Frecuencia":frecuencia,"Categoria":categoria})
         
    print("Writing complete")
    