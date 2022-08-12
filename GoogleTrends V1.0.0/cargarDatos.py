# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:45:41 2022

@author: Fabián Fernández Chaves
"""

from selenium.webdriver.common.by import By
import time

def cargar_datos(driver):
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