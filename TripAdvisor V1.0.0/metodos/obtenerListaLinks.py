# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 12:56:06 2022

@author: Fabián Fernández Chaves
"""

from selenium.webdriver.common.by import By
import time


##################################################################
#la pagina suele cargar incorrectaente los resultados, así que 
#links de cada uno de los resultados para almacenarlos en una lista
##################################################################
def cargar_datos(driver):
    try:
        time.sleep(5)
        driver.find_element(By.ID,"geobroaden_opt_out").click()
    except:
        print("NA")    
    try:
        
        time.sleep(5)
        driver.find_element(By.XPATH,"// a[contains(text(),\'Siguiente')]").click()
        time.sleep(5)
        driver.find_element(By.XPATH,"// a[contains(text(),\'Anterior')]").click()
        time.sleep(5)
    except:
        print("No hay más paginas de resultados")
#--------------------------------------------------------------------------        


##################################################################
#Recorre los elementos que componen la busqueda y extrae los
#links de cada uno de los resultados para almacenarlos en una lista
##################################################################
def obtener_links(links, rango_i, rango_f, driver):
    for i in range(rango_i, rango_f):
        
        select=str(i+1)
        
        restaurant=driver.find_elements(By.XPATH,'//*[@data-test="'+select+'_list_item"]/span/div[1]/div[2]/div[1]/div/span/a')
        
        link = [elem.get_attribute('href') for elem in restaurant]
        if len(link)!=0:
            links+=link
#--------------------------------------------------------------------------

def abanzar_pagina(rango_i, rango_f, driver, links):
    siguiente=driver.find_elements(By.XPATH,"//*[@class='nav next disabled']")
    while True:
        time.sleep(3)
        obtener_links(links,rango_i, rango_f, driver)
        time.sleep(1)
        if len(siguiente)!=0:
            break
        else:
            try:
                driver.find_element(By.XPATH,"// a[contains(text(),\'Siguiente')]").click()
            except:
                try:
                    time.sleep(2)
                    driver.find_element(By.XPATH,"// a[contains(text(),\'Siguiente')]").click()
                except:
                    break
            rango_i+=30
            rango_f+=30
            time.sleep(2)
            siguiente=driver.find_elements(By.XPATH,"//*[@class='nav next disabled']")
    time.sleep(3)
#--------------------------------------------------------------------------        
