# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:45:08 2022

@author: Fabián Fernández Chaves
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service

import extraerConsultas as Ec
import cargarDatos as Cd
import extraerTemas as Et
import escribirDocumentos as Ed

import time


lista_provincias=["SJ","P", "A","H"]
lista_anos=["0","1","2"]
link="https://trends.google.com/trends/explore?cat=71&date=2020-01-01%202020-12-31&geo=CR-"
documento_consultas=[]
documento_temas=[]

#opciones
options = webdriver.ChromeOptions()
options.add_argument('--start-maximized')
options.add_argument('user-data-dir=C://Users//fafec//AppData//Local//Google//Chrome//User Data')

#Agregar la ruta en la que está el driver de chrome
s=Service('chromedriver.exe')
options.add_argument('--start-maximized')
#Agregar la ruta en la que está el driver de chrome
s=Service('chromedriver.exe')

#Agregar las opciones y el servicio al webdriver 
driver = webdriver.Chrome(chrome_options = options,service=s)
#Dar 5 segundos para elegir el perfil desde el que se va a ejecutar el script
time.sleep(2)




for p in lista_provincias:
    dire=link+p
    for a in lista_anos:
        indice=a+p
        print(indice)
        l = list(dire)
        l[56] = a
        l[69] = a
        direc = "".join(l)
        driver.get(direc)
        time.sleep(3)
        Cd.cargar_datos(driver)
        time.sleep(3)
        ano="202"+a
        Ec.extraer_consultas(ano,p,documento_consultas, driver)
        Et.extraer_temas(ano,p,documento_temas, driver)
        

Ed.escribir(documento_consultas,"Consultas.csv")
Ed.escribir(documento_temas,"Temas.csv")


