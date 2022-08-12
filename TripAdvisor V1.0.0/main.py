# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 12:52:19 2022

@author: Fabián Fernández Chaves
"""

#Importar librerias necesarias

from selenium import webdriver
from selenium.webdriver.chrome.service import Service

import metodos.obtenerDatosRest as Odr
import metodos.obtenerListaLinks as Ol
import metodos.seleccion as sel


import time
import csv

#opciones
options = webdriver.ChromeOptions()
options.add_argument('--start-maximized')

#Cambiar el nombre del usuario segun la pc
options.add_argument('user-data-dir=C://Users//fafec//AppData//Local//Google//Chrome//User Data')

#Agregar la ruta en la que está el driver de chrome
s=Service('chromedriver.exe')

#Agregar las opciones y el servicio al webdriver 
driver = webdriver.Chrome(chrome_options = options,service=s)
#Dar 5 segundos para elegir el perfil desde el que se va a ejecutar el script
time.sleep(2)

#inicializar variables necesarias
links=[] #Links de las paginas que se guardarán
rango_i=0 #Rango de inicio para la busqueda de links
rango_f=30 #rango final para la busqueda de links

caracteristicas=[]#Almacen temporal de las caracteristicas del hotel
lista_Rest=[]#Diccionario donde se almacena toda la info 
zona=sel.seleccionar_zona(1)
print(zona)
##### Designar la ruta de busqueda #####
driver.get(zona)
##### Cargar correctamente los elementos de la página #####
Ol.cargar_datos(driver)
##### Recorrer los resultados de busqueda #####
Ol.abanzar_pagina(rango_i, rango_f, driver, links)


###############################################################################
###############################################################################
# Sección para extraer los datos de los links recuperados
###############################################################################
###############################################################################

#### Acceder a cada uno de los links obtenidos ####
Odr.abrir_pag(links, lista_Rest, driver)
 
###############################################################################
###############################################################################
# Escribir los datos en un archivo csv
###############################################################################
###############################################################################

campos=["ID","Nombre","Calificación","Tipo_Comida","Direccion","Coordenadas","Comentario","Fecha_Comentario","Calificacion_Comentario"]
with open('documentos/Reseñas.csv', 'w', newline = '',encoding='utf-8') as myFile:
    writer = csv.DictWriter(myFile,fieldnames=campos)
    writer.writeheader()
    for rest in lista_Rest: 
        comentarios=rest[3]
        
        
        for com in comentarios:
            ide=rest[6]
            nom=rest[0]
            cal=rest[1]
            tipC=rest[2]
            dire=rest[4]
            come=com[0].replace("\n","")
            come=come.replace("\r","")
            come=come.replace("\r\n","")
            calCom=com[1]
            fecha_com=com[2]
            coorde=rest[5]
            writer.writerow({"ID":ide,"Nombre":nom,"Calificación":cal,"Direccion":dire,"Tipo_Comida":tipC,"Comentario":come,"Calificacion_Comentario":calCom,"Coordenadas":coorde,"Fecha_Comentario":fecha_com})
     
print("Writing complete") 