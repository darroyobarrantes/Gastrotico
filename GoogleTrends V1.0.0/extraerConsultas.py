# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:46:23 2022

@author: Fabián Fernández Chaves
"""
from selenium.webdriver.common.by import By
import time


def extraer_consultas(a,p,documento_consultas, driver):
    
    for j in range(5):
        datos_nombres=driver.find_elements(By.XPATH,'//div[@id="RELATED_QUERIES"]/following-sibling::trends-widget/ng-include/widget/div/div/ng-include//span')
        datos_valor=driver.find_elements(By.XPATH,'//div[@id="RELATED_QUERIES"]/following-sibling::trends-widget/ng-include/widget/div/div/ng-include//div[@class="progress-value"]')
        nombres=[elem.get_attribute('textContent') for elem in datos_nombres]
        valores=[elem.get_attribute('textContent') for elem in datos_valor]   
        for i in range(5):
            documento_consultas.append([a,p,nombres[i],valores[i],"Alimentos y bebidas"])
        
        driver.find_element(By.XPATH,'//md-content/div/div/div[3]/trends-widget/ng-include/widget/div/div/ng-include/div/div[6]/pagination/div/button[2]/md-icon').click()
        time.sleep(2)
