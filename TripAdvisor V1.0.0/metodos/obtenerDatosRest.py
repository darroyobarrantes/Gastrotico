# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:08:36 2022

@author: Fabián Fernández Chaves
"""

from selenium.webdriver.common.by import By
import metodos.obtenerDatosComentario as Odc
import time


##################################################################
#Recorre la lista de links que se obtubieron y los va abriendo uno
#a uno. Tambien Selecciona la opcion de todos los idiomas
##################################################################
def abrir_pag(lista, lista_Rest, driver):
    linea=0
    comentarios=[]#Almacen temporal de los comentarios
    for l in lista:
        comentarios=[]
        try:
            driver.get(l)
            
        except:
            print("No pudimos cargar la página")
            time.sleep(10)
            driver.get(l)
        
           
        #Almacen temporal de las caracteristicas del hotel
        token=l[49:]
        token=token.split("-")
        token=token[0]+"-"+token[1]
        
        caracteristicas=obtener_caracteristicas(driver)   
        linea+=1
        print("_________________________________________________________")
        print(linea)
        print(len(lista))
        print(l)
        print("---------------------------------------------------------")
        time.sleep(4)
        # Busca la opción de elegir todos los idiomas y da click
        try:
            driver.find_element(By.ID,"filters_detail_language_filterLang_es").click()            
            comentarios=Odc.recorrer_coment(comentarios, driver)
            coordenadas=obtener_coordenadas(driver)   
            lista_r=[caracteristicas[0],caracteristicas[1],caracteristicas[2],comentarios,caracteristicas[3],coordenadas,token]
            lista_Rest.append(lista_r)   
        except:
            try:
                print("no encontramos la opcion de todos los idiomas. Reintentando")
                time.sleep(2)
                driver.find_element(By.ID,"filters_detail_language_filterLang_es").click() 
                comentarios=Odc.recorrer_coment(comentarios, driver)
                coordenadas=obtener_coordenadas(driver)   
                lista_r=[caracteristicas[0],caracteristicas[1],caracteristicas[2],comentarios,caracteristicas[3],coordenadas,token]
                lista_Rest.append(lista_r)   
            except:
                try:
                    print("no encontramos la opcion de todos los idiomas. Reintentando")
                    time.sleep(4)
                    driver.find_element(By.ID,"filters_detail_language_filterLang_es").click() 
                    comentarios=Odc.recorrer_coment(comentarios, driver)
                    coordenadas=obtener_coordenadas(driver)   
                    lista_r=[caracteristicas[0],caracteristicas[1],caracteristicas[2],comentarios,caracteristicas[3],coordenadas,token]
                    lista_Rest.append(lista_r)   
                except:
                    print("No hay comentarios")
                
                
         
        print(len(comentarios))
          
        # Llamar el metodo de recorrer los comentarios

        
#--------------------------------------------------------------------------        

##################################################################
#Extrae los caracteristicas de la pagia actual
##################################################################
def obtener_caracteristicas(driver):
    nombre=""
    calificacion=""
    tipo_comida=""
    direcc=""
    try:
        direcc=driver.find_element(By.XPATH,'//*/div/div[3]/span[1]/span/a').text
        nombre=driver.find_element(By.XPATH,'//*[@class="HjBfq"]').text
        calificacion=driver.find_element(By.XPATH,'//*[@class="ZDEqb"]').text
        calificacion=calificacion[0:3]
        try:
            tipo_comida=driver.find_element(By.XPATH,'//div[contains(text(),"TIPOS DE COMIDA")]/following-sibling::div').text
        except:
            tipo_comida="N/E"
        
    except:
        try:
            print("Hubo un error obteniendo los datos. Reintentado")
            time.sleep(4)
            direcc=driver.find_element(By.XPATH,'//*/div/div[3]/span[1]/span/a').text
            nombre=driver.find_element(By.XPATH,'//div/div[1]/h1').text
            calificacion=driver.find_element(By.XPATH,'//*[@class="fdsdx"]').text
            calificacion=calificacion[0:3]
            try:
                tipo_comida=driver.find_element(By.XPATH,'//div[contains(text(),"TIPOS DE COMIDA")]/following-sibling::div').text
            except:
                tipo_comida="N/E"
            
        except:
            print("Hubo un error obteniendo los datos. Reintentado")
            time.sleep(4)
            direcc=driver.find_element(By.XPATH,'//*/div/div[3]/span[1]/span/a').text
            nombre=driver.find_element(By.XPATH,'//div/div[1]/h1').text
            try:
                calificacion=driver.find_element(By.XPATH,'//*[@class="fdsdx"]').text
                calificacion=calificacion[0:3]
            except:
                calificacion="N/E"
            try:
                tipo_comida=driver.find_element(By.XPATH,'//div[contains(text(),"TIPOS DE COMIDA")]/following-sibling::div').text
            except:
                tipo_comida="N/E"
            
    caracteristicas=[nombre,calificacion,tipo_comida, direcc]
    return caracteristicas
#--------------------------------------------------------------------------        


##################################################################
#Obtener coordenadas
##################################################################
def obtener_coordenadas(driver):
    coordenadas=""
    url=driver.find_element(By.XPATH,'//*/div[1]/div/div[3]/div/div/div[1]/span[2]/a')
    driver.get(url.get_attribute("href"))
    time.sleep(10)
    coordenadas=driver.current_url
    coordenadas=coordenadas[33:]
    coordenadas=coordenadas.split("/")
    coordenadas=coordenadas[0]
    return coordenadas
 
#--------------------------------------------------------------------------        
