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

#opciones
options = webdriver.ChromeOptions()
options.add_argument('--start-maximized')
options.add_argument('user-data-dir=C://Users//fafec//AppData//Local//Google//Chrome//User Data')

#Agregar la ruta en la que está el driver de chrome
s=Service('chromedriver.exe')

#Agregar las opciones y el servicio al webdriver 
driver = webdriver.Chrome(chrome_options = options,service=s)
#Dar 5 segundos para elegir el perfil desde el que se va a ejecutar el script
time.sleep(5)

#inicializar variables necesarias
links=["https://www.tripadvisor.com.mx/Restaurant_Review-g1070992-d4228402-Reviews-Pollos_Bar-Santiago_de_Puriscal_Province_of_San_Jose.html","https://www.tripadvisor.com.mx/Restaurant_Review-g1070992-d13830849-Reviews-Mirachos_Bar_Restaurante-Santiago_de_Puriscal_Province_of_San_Jose.html","https://www.tripadvisor.com.mx/Restaurant_Review-g1070992-d19850861-Reviews-El_Tulin-Santiago_de_Puriscal_Province_of_San_Jose.html"] #Links de las paginas que se guardarán
lista_rest=[]
rango_i=0 #Rango de inicio para la busqueda de links
rango_f=30 #rango final para la busqueda de links

comentarios=[]#Almacen temporal de los comentarios
caracteristicas=[]#Almacen temporal de las caracteristicas del hotel
lista_Rest={}#Diccionario donde se almacena toda la info 

#######################bloque de funciones#######################


##################################################################
#Recorre la lista de links que se obtubieron y los va abriendo uno
#a uno. Tambien Selecciona la opcion de todos los idiomas
##################################################################
def abrir_pag(lista, lista_rest):
    linea=0
    comentarios=[]#Almacen temporal de los comentarios
    for l in lista:
        try:
            driver.get(l)
            
        except:
            print("No pudimos cargar la página")
            time.sleep(10)
            driver.get(l)
        #Almacen temporal de las caracteristicas del hotel
        caracteristicas=obtener_caracteristicas()   
        linea+=1
        print("_________________________________________________________")
        print(linea)
        print(l)
        print("---------------------------------------------------------")
        time.sleep(5)
        # Busca la opción de elegir todos los idiomas y da click
        try:
            driver.find_element(By.ID,"filters_detail_language_filterLang_ALL").click()            
            comentarios=recorrer_coment(comentarios)
        except:
            try:
                print("no encontramos la opcion de todos los idiomas. Reintentando")
                time.sleep(10)
                driver.find_element(By.ID,"filters_detail_language_filterLang_ALL").click() 
                comentarios=recorrer_coment(comentarios)
            except:
                print("No hay comentarios")
            
        print(len(comentarios))
        print(comentarios)    
        # Llamar el metodo de recorrer los comentarios
        lista_rest[caracteristicas[0]]={'Caracteristicas':caracteristicas[2:3], 'Comentarios':comentarios}
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
            time.sleep(5)
            # Si ya no hay más paginas de resultados finalizamos el ciclo
            
            time.sleep(5)
             
             #############
            comentarios+=obtener_comentario()
             ################
             
             # Si existe un boton de siguiente se presiona, sino terminamos ciclo
            try:
                driver.find_element(By.XPATH,"// a[contains(text(),\'Siguiente')]").click()
            except:
                break
            time.sleep(5)
             # Guarda los elementos que contengan el css especifico->Boton Siguiente desactivado
            pagina+=1
            
            print('--------------------------------')
                
    except:
        print("no funciona")

    
    time.sleep(5)
    return comentarios
 
#--------------------------------------------------------------------------        

##################################################################
#Extrae los comentarios de la pagia actual y tambien verifica si son
#y hay que clicar en ver más
##################################################################
def obtener_comentario():
     print("Estamos obteniendo los comentarios")
     #verificar que no hayan "mostrar más"
     print("Hay 1")
     
     try:
         time.sleep(5)
         driver.find_element(By.CLASS_NAME, 'taLnk.ulBlueLinks').click()
         print("Hay 2")
     except:
         print("no entró")
         
     print("Hay 3")
     time.sleep(5)
     com=driver.find_elements_by_xpath('//*[@class="ui_column is-9"]/div[@class="prw_rup prw_reviews_text_summary_hsx"]/div[@class="entry"]/p')
     print("Hay 4")
     comentarios=[elem.get_attribute('textContent') for elem in com]
     print(comentarios)
     return comentarios
#--------------------------------------------------------------------------        

##################################################################
#Extrae los caracteristicas de la pagia actual
##################################################################
def obtener_caracteristicas():
    nombre=""
    calificacion=""
    tipo_comida=""
    try:
        nombre=driver.find_element_by_xpath('//div/div[1]/h1').text
        calificacion=driver.find_element_by_xpath('//*[@class="fdsdx"]').text
        calificacion=calificacion[0:3]
        tipo_comida=driver.find_element_by_xpath('//div[contains(text(),\"TIPOS DE COMIDA")]/following-sibling::div').text
    except:
        try:
            print("Hubo un error obteniendo los datos. Reintentado")
            time.sleep(10)
            nombre=driver.find_element_by_xpath('//div/div[1]/h1').text
            calificacion=driver.find_element_by_xpath('//*[@class="fdsdx"]').text
            calificacion=calificacion[0:3]
            tipo_comida=driver.find_element_by_xpath('//div[contains(text(),\"TIPOS DE COMIDA")]/following-sibling::div').text
        except:
            time.sleep(10)
            tipo_comida="N/E"
    caracteristicas=[nombre,calificacion,tipo_comida]
    print(caracteristicas)
    return caracteristicas
#--------------------------------------------------------------------------        

####################### fin del Bloque de funciones#######################


##### Designar la ruta de busqueda #####
driver.get('https://www.tripadvisor.com.mx/Restaurants-g644051-Parrita_Province_of_Puntarenas.html')


###############################################################################
###############################################################################
# Sección para extraer los datos de los links recuperados
###############################################################################
###############################################################################

#### Acceder a cada uno de los links obtenidos ####
abrir_pag(links, lista_Rest)
print("------------------------------------------------------")
print("------------------------------------------------------")
print("------------------------------------------------------")
print("------------------------------------------------------")
print("------------------------------------------------------")
print("------------------------------------------------------")
print("------------------------------------------------------")
print("------------------------------------------------------")
print(lista_Rest)







time.sleep(10)

driver.close()   
    
             
