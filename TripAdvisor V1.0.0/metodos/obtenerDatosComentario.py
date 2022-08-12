# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:07:41 2022

@author: Fabián Fernández Chaves
"""
from selenium.webdriver.common.by import By
import time


##################################################################
#Funcion que avanza a la siguiente pagina de comentarios si es que
#está disponible, si no hay más resultados se detiene
##################################################################
def recorrer_coment(comentarios, driver):
    pagina=1
    print("Estamos recorriendo los comentarios")
    # Guarda los elementos que contengan el css especifico->Boton Siguiente desactivado
    try:
        while True:
            print(pagina)
            # Cuando avanzamos 10 paginas refrescamos para evitar errores
            if pagina%10==0:
                driver.refresh()
            time.sleep(6)
            # Si ya no hay más paginas de resultados finalizamos el ciclo
            
             
             #############
            comentarios+=obtener_comentario(driver)
             ################
             
             # Si existe un boton de siguiente se presiona, sino terminamos ciclo
            try:
                driver.find_element(By.XPATH,"// a[contains(text(),\'Siguiente')]").click()
            except:
                try:
                    driver.find_element(By.XPATH,"// a[contains(text(),\'Siguiente')]").click()
                except:
                    try:
                        driver.find_element(By.XPATH,"// a[contains(text(),\'Siguiente')]").click()
                    except:
                        break
            time.sleep(4)
             # Guarda los elementos que contengan el css especifico->Boton Siguiente desactivado
            pagina+=1
            
            print('--------------------------------')
                
    except:
        print("no funciona")

    
    time.sleep(4)
    return comentarios
 
#--------------------------------------------------------------------------        


##################################################################
#Extrae los comentarios de la pagia actual y tambien verifica si son
#y hay que clicar en ver más
##################################################################
def obtener_comentario(driver):
    print("Estamos obteniendo los comentarios")
    #verificar que no hayan "mostrar más"
     
    try:
        time.sleep(5)
        driver.find_element(By.CLASS_NAME, 'taLnk.ulBlueLinks').click()
        
    except:
        try:
            time.sleep(2)
            driver.find_element(By.CLASS_NAME, 'taLnk.ulBlueLinks').click()
            
        except:
            try:
                time.sleep(1)
                driver.find_element(By.CLASS_NAME, 'taLnk.ulBlueLinks').click()
                
            except:
                try:
                    time.sleep(1)
                    driver.find_element(By.CLASS_NAME, 'taLnk.ulBlueLinks').click()
                    
                except:
                    print("No encontró comentarios con Más")
                
    time.sleep(4)
    com=driver.find_elements(By.XPATH,'//*[@class="ui_column is-9"]/div[@class="prw_rup prw_reviews_text_summary_hsx"]/div[@class="entry"]/p')
    cal=driver.find_elements(By.XPATH,'//*[@class="ui_column is-9"]/span[1]')
    fec=driver.find_elements(By.XPATH,'//*[@class="ui_column is-9"]/span[2]')
    fecha=[elem.get_attribute('title') for elem in fec]
    calificacion=[elem.get_attribute('class') for elem in cal]
    comentario=[elem.get_attribute('textContent') for elem in com]
    
    fecha=obtener_fecha(fecha)
    
    try:
       i=0
       
       for c in calificacion:
           calificacion[i]=c[24]
           i+=1
       size=len(comentario)
    except:
           print("califica")
       
    
    comentarios=[]
    for j in range(size):
        com=[]
        com.append(comentario[j])
        com.append(calificacion[j])
        com.append(fecha[j])
        comentarios.append(com)
    
    return comentarios
#-------------------------------------------------------------------------        

##################################################################
#Cambia el formato de la fecha
##################################################################
def obtener_fecha(fecha):
    meses=["enero","febrero","marzo","abril","mayo","junio","julio","agosto","setiembre","octubre","noviembre","diciembre"]
    fec=[]
    for f in fecha:
        i=1
        for m in meses:
            valor=f.find(m)
            if valor>0:
                numero=str(i)
                f.replace(m,numero)
            else:
                i=i+1
        f=f.replace(" ","")
        f=f.replace("de","-")
        fec.append(f)
        
    fecha=fec
    return fecha
#-------------------------------------------------------------------------        
