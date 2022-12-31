# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 16:51:47 2022

@author: Fabián Fernández Chaves
"""
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import csv

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

def cargar_datos():
    time.sleep(2)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    dropdowns=driver.find_elements(By.CLASS_NAME,'bullets-view-selector.ng-pristine.ng-valid.ng-not-empty.ng-untouched')
    dropdown_options=driver.find_elements(By.XPATH,'//md-option[@value="bullets"]/div[1]')    
    
   
    dropdown1 = dropdowns[0]
    try:
        time.sleep(3)
        dropdown1.click()
    except:
        try:
            time.sleep(3)
            dropdown1.click()
        except:
            try:
                time.sleep(3)
                dropdown1.click()
            except:
                time.sleep(3)
                dropdown1.click()
    
    time.sleep(2)
    dropdown_option1=dropdown_options[0]
    try:
        dropdown_option1.click()
    except:
        try:
            time.sleep(3)
            dropdown_option1.click()
        except:
            time.sleep(3)
            dropdown_option1.click()
    
    
    dropdown2 = dropdowns[1]
    try:
        time.sleep(3)
        dropdown2.click()
    except:
        try:
            time.sleep(3)
            dropdown2.click()
        except:
            try:
                time.sleep(3)
                dropdown2.click()
            except:
                time.sleep(3)
                dropdown2.click()
    
    time.sleep(2)
    dropdown_option2=dropdown_options[1]
    try:
        dropdown_option2.click()
    except:
        try:
            time.sleep(3)
            dropdown_option2.click()
        except:
            time.sleep(3)
            dropdown_option2.click()
def extraer_consultas(a,p,documento_consultas):
    
    for j in range(5):
        datos_nombres=driver.find_elements(By.XPATH,'//div[@id="RELATED_QUERIES"]/following-sibling::trends-widget/ng-include/widget/div/div/ng-include//span')
        datos_valor=driver.find_elements(By.XPATH,'//div[@id="RELATED_QUERIES"]/following-sibling::trends-widget/ng-include/widget/div/div/ng-include//div[@class="progress-value"]')
        nombres=[elem.get_attribute('textContent') for elem in datos_nombres]
        valores=[elem.get_attribute('textContent') for elem in datos_valor]   
        for i in range(5):
            documento_consultas.append([a,p,nombres[i],valores[i],"Alimentos y bebidas"])
        
        driver.find_element(By.XPATH,'//md-content/div/div/div[3]/trends-widget/ng-include/widget/div/div/ng-include/div/div[6]/pagination/div/button[2]/md-icon').click()
        time.sleep(2)

    
    
def extraer_temas(a,p,documento_temas):
    for j in range(5):
        datos_nombres=driver.find_elements(By.XPATH,'//div[@id="RELATED_TOPICS"]/following-sibling::trends-widget/ng-include/widget/div/div/ng-include//span')
        datos_valor=driver.find_elements(By.XPATH,'//div[@id="RELATED_TOPICS"]/following-sibling::trends-widget/ng-include/widget/div/div/ng-include//div[@class="progress-value"]')
        nombres=[elem.get_attribute('textContent') for elem in datos_nombres]
        valores=[elem.get_attribute('textContent') for elem in datos_valor]   
        for i in range(5):
            documento_temas.append([a,p,nombres[i],valores[i],"Alimentos y bebidas"])
        
        driver.find_element(By.XPATH,'//md-content/div/div/div[2]/trends-widget/ng-include/widget/div/div/ng-include/div/div[6]/pagination/div/button[2]/md-icon').click()
        time.sleep(2)

def escribir(documento, nombre):
    lista_datos=documento
    campos=["Año","Provincia","Consulta","Frecuencia","Categoria"]
    with open(nombre, 'w', newline = '',encoding='utf-8') as myFile:
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
        cargar_datos()
        time.sleep(3)
        ano="202"+a
        extraer_temas(ano,p,documento_temas)
        extraer_consultas(ano,p,documento_consultas)

escribir(documento_consultas,"Consultas.csv")
escribir(documento_temas,"Temas.csv")