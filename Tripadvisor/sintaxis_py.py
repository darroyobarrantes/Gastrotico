# -*- coding: utf-8 -*-
"""
Created on Mon May  9 20:02:44 2022

@author: Fabián Fernández Chaves
"""
#Importar librerias necesarias

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import csv

#opciones
options = webdriver.ChromeOptions()
options.add_argument('--start-maximized')
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

#######################bloque de funciones#######################

##################################################################
#Recorre los elementos que componen la busqueda y extrae los
#links de cada uno de los resultados para almacenarlos en una lista
##################################################################
def obtener_links(links, rango_i, rango_f):
    for i in range(rango_i, rango_f):
        
        select=str(i+1)
        
        restaurant=driver.find_elements(By.XPATH,'//*[@data-test="'+select+'_list_item"]/span/div[1]/div[2]/div[1]/div/span/a')
        
        link = [elem.get_attribute('href') for elem in restaurant]
        if len(link)!=0:
            links+=link
#--------------------------------------------------------------------------

##################################################################
#la pagina suele cargar incorrectaente los resultados, así que 
#links de cada uno de los resultados para almacenarlos en una lista
##################################################################
def cargar_datos():
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
#Funcion que avanza a la siguiente pagina de resultados si es que
#está disponible, si no hay más resultados se detiene
##################################################################
def abanzar_pagina(rango_i, rango_f):
    siguiente=driver.find_elements(By.XPATH,"//*[@class='nav next disabled']")
    while True:
        time.sleep(5)
        obtener_links(links,rango_i, rango_f)
        time.sleep(5)
        if len(siguiente)!=0:
            break
        else:
            try:
                driver.find_element(By.XPATH,"// a[contains(text(),\'Siguiente')]").click()
            except:
                try:
                    time.sleep(5)
                    driver.find_element(By.XPATH,"// a[contains(text(),\'Siguiente')]").click()
                except:
                    break
            rango_i+=30
            rango_f+=30
            time.sleep(5)
            siguiente=driver.find_elements(By.XPATH,"//*[@class='nav next disabled']")
    time.sleep(7)
#--------------------------------------------------------------------------        

##################################################################
#Recorre la lista de links que se obtubieron y los va abriendo uno
#a uno. Tambien Selecciona la opcion de todos los idiomas
##################################################################
def abrir_pag(lista, lista_rest):
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
        
        caracteristicas=obtener_caracteristicas()   
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
            comentarios=recorrer_coment(comentarios)
            coordenadas=obtener_coordenadas()   
            lista_r=[caracteristicas[0],caracteristicas[1],caracteristicas[2],comentarios,caracteristicas[3],coordenadas,token]
            lista_Rest.append(lista_r)   
        except:
            try:
                print("no encontramos la opcion de todos los idiomas. Reintentando")
                time.sleep(2)
                driver.find_element(By.ID,"filters_detail_language_filterLang_es").click() 
                comentarios=recorrer_coment(comentarios)
                coordenadas=obtener_coordenadas()   
                lista_r=[caracteristicas[0],caracteristicas[1],caracteristicas[2],comentarios,caracteristicas[3],coordenadas,token]
                lista_Rest.append(lista_r)   
            except:
                try:
                    print("no encontramos la opcion de todos los idiomas. Reintentando")
                    time.sleep(4)
                    driver.find_element(By.ID,"filters_detail_language_filterLang_es").click() 
                    comentarios=recorrer_coment(comentarios)
                    coordenadas=obtener_coordenadas()   
                    lista_r=[caracteristicas[0],caracteristicas[1],caracteristicas[2],comentarios,caracteristicas[3],coordenadas,token]
                    lista_Rest.append(lista_r)   
                except:
                    print("No hay comentarios")
                
                
         
        print(len(comentarios))
          
        # Llamar el metodo de recorrer los comentarios

        
#--------------------------------------------------------------------------        

##################################################################
#Obtener coordenadas
##################################################################
def obtener_coordenadas():
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

##################################################################
#Funcion que avanza a la siguiente pagina de comentarios si es que
#está disponible, si no hay más resultados se detiene
##################################################################
def recorrer_coment(comentarios):
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
            comentarios+=obtener_comentario()
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
            time.sleep(7)
             # Guarda los elementos que contengan el css especifico->Boton Siguiente desactivado
            pagina+=1
            
            print('--------------------------------')
                
    except:
        print("no funciona")

    
    time.sleep(7)
    return comentarios
 
#--------------------------------------------------------------------------        

##################################################################
#Extrae los comentarios de la pagia actual y tambien verifica si son
#y hay que clicar en ver más
##################################################################
def obtener_comentario():
    print("Estamos obteniendo los comentarios")
    #verificar que no hayan "mostrar más"
     
    try:
        time.sleep(7)
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
                
    time.sleep(7)
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

##################################################################
#Extrae los caracteristicas de la pagia actual
##################################################################
def obtener_caracteristicas():
    nombre=""
    calificacion=""
    tipo_comida=""
    direcc=""
    try:
        direcc=driver.find_element(By.XPATH,'//*/div/div[3]/span[1]/span/a').text
        nombre=driver.find_element(By.XPATH,'//*[@class="fHibz"]').text
        calificacion=driver.find_element(By.XPATH,'//*[@class="fdsdx"]').text
        calificacion=calificacion[0:3]
        tipo_comida=driver.find_element(By.XPATH,'//div[contains(text(),\"TIPOS DE COMIDA")]/following-sibling::div').text
        
        
    except:
        try:
            print("Hubo un error obteniendo los datos. Reintentado")
            time.sleep(4)
            direcc=driver.find_element(By.XPATH,'//*/div/div[3]/span[1]/span/a').text
            nombre=driver.find_element(By.XPATH,'//div/div[1]/h1').text
            calificacion=driver.find_element(By.XPATH,'//*[@class="fdsdx"]').text
            calificacion=calificacion[0:3]
            tipo_comida=driver.find_element(By.XPATH,'//div[contains(text(),\"TIPOS DE COMIDA")]/following-sibling::div').text
            
        except:
            tipo_comida="N/E"
    caracteristicas=[nombre,calificacion,tipo_comida, direcc]
    return caracteristicas
#--------------------------------------------------------------------------        

####################### fin del Bloque de funciones#######################


##### Designar la ruta de busqueda #####
driver.get("https://www.tripadvisor.com.mx/Restaurants-g1070992-Santiago_de_Puriscal_Province_of_San_Jose.html")
##### Cargar correctamente los elementos de la página #####
cargar_datos()
##### Recorrer los resultados de busqueda #####
abanzar_pagina(rango_i, rango_f)


###############################################################################
###############################################################################
# Sección para extraer los datos de los links recuperados
###############################################################################
###############################################################################

#### Acceder a cada uno de los links obtenidos ####
abrir_pag(links, lista_Rest)
 

###############################################################################
###############################################################################
# Escribir los datos en un archivo csv
###############################################################################
###############################################################################

campos=["ID","Nombre","Calificación","Tipo_Comida","Direccion","Coordenadas","Comentario","Fecha_Comentario","Calificacion_Comentario"]
with open('Hola.csv', 'w', newline = '',encoding='utf-8') as myFile:
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
             
