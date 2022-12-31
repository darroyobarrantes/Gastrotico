# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 20:14:27 2022

@author: Fabián Fernández Chaves
"""
import csv
import pandas as pd
primer_Lista=[]
segunda_Lista=[]
lista_Rest=[]

with open("Unir Archivos/Reseña el carmen 4.csv", 'r', encoding="utf8" ) as file_name:
    file_read = csv.reader(file_name)

    primer_Lista = list(file_read)
    primer_Lista.pop(0)

indice=0
for l in primer_Lista:
    lista_Rest.append(l)
    indice+=1
    
    
    
with open("Unir Archivos/Reseña Jacó.csv", 'r', encoding="utf8") as file_name:
    file_read = csv.reader(file_name)
    segunda_Lista = list(file_read)


indice=0
for l in segunda_Lista:
    lista_Rest.append(l)
    indice+=1
  
    



df=pd.read_csv("Unir Archivos/Reseñas.csv")

campos=["ID","Nombre","Calificación","Tipo_Comida","Direccion","Coordenadas","Comentario","Fecha_Comentario","Calificacion_Comentario"]
with open('Unir Archivos/Reseñas.csv', 'w', newline = '',encoding='utf-8') as myFile:
    writer = csv.DictWriter(myFile,fieldnames=campos)
    writer.writeheader()
    for fila in lista_Rest: 
        writer.writerow({"ID":fila[0],"Nombre":fila[1],"Calificación":fila[2],"Tipo_Comida":fila[3],"Direccion":fila[4],"Coordenadas":fila[5],"Comentario":fila[6],"Fecha_Comentario":fila[7],"Calificacion_Comentario":fila[8]})
     
print("Writing complete") 