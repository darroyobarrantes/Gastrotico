# -*- coding: utf-8 -*-

"""
Created on Tue Oct  4 16:32:08 2022

@author: Fabián Fernández Chaves
"""
from progress.bar import Bar
import time
import string
import spacy as sp
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import SnowballStemmer
from langdetect import detect
import re

nltk.download("stopwords")
nltk.download('punkt')
sw=stopwords.words('spanish')
spanishstemmer=SnowballStemmer("spanish")


def eliminarEng(resena):
    
    bar1 = Bar('Eng:', max=len(resena))
    
    for indice_fila, fila in resena.iterrows():
        time.sleep(0.0002)
        bar1.next()
        text=fila['Comentario']
        if detect(text)!="es":
            resena=resena.drop( resena[resena['Comentario']==fila['Comentario']].index)
    bar1.finish()
    return resena
    


def addStopWords():
    f = pd.read_csv("spanish-stop-words.txt", encoding="utf-8")
    for indice_fila, fila in f.iterrows():
      sw.extend([fila["Palabras"]])


def eliminarPunctuation(resena):
    bar1 = Bar('Elim Puntc:', max=len(resena))
    punctuation = string.punctuation
    punctuation+='''1234567890!¿¡@#$%^&*(){}[]|._-`/?:;"'\,~12345678876543'''
    for indice_fila, fila in resena.iterrows():
        
        time.sleep(0.000002)
        bar1.next()
        
        comen=fila['Comentario']
        for char in comen:
            if char in punctuation:
                comen = comen.replace(char, "")
        
        resena['Comentario']=resena['Comentario'].replace(fila['Comentario'],comen) 
    bar1.finish()
    return resena
    

def cambiaEmojis(resena):
    
    emoticons = {
        ":)": "😃",
        ":D": "😃",
        "X)": "😃",
        "x)": "😃",
        "XD": "😃",
        "xD": "😃",
        "Xd": "😃",
        "xd": "😃",
        ":p": "😛",
        ":P": "😛",
        "<3": "❤️",
        "</3": "💔",
        ":(": "🙁",
        ":/": "🙁",
        ":'(": "🙁",
    }

    bar1 = Bar('Conv Emojis:', max=len(resena))
    for indice_fila, fila in resena.iterrows():
        time.sleep(0.000002)
        bar1.next()
        comen=fila['Comentario']
        for emote, rpl in emoticons.items():
            comen = comen.replace(emote, rpl)
            
        resena['Comentario']=resena['Comentario'].replace(fila['Comentario'],comen)   
    bar1.finish()
    return resena


def formato(resena):
    bar1 = Bar('Formato:', max=len(resena))
    for indice_fila, fila in resena.iterrows():
        
        time.sleep(0.000002)
        bar1.next()
        
        comen=fila['Comentario']
        comen= comen.strip()
        comen=comen.lower()
        comen=" ".join(comen.split())
        resena['Comentario']=resena['Comentario'].replace(fila['Comentario'],comen) 
    bar1.finish()
    return resena

def normalize(resena):
    addStopWords()
    lista_comen_token=pd.DataFrame(columns=["Tipo_Comida","Calificacion","Comentario","Calificacion_Comentario"])
    nlp = sp.load("es_core_news_sm")
    
    bar1 = Bar('Normalizacion:', max=len(resena))
    
    for indice_fila, fila in resena.iterrows():
        time.sleep(0.000002)
        bar1.next()
        
        text=fila['Comentario']
        calF=fila['Calificacion_Comentario']
        words = nltk.tokenize.word_tokenize(text)
        removing_custom_words = [w for w in words if not w in sw]
        text=" ".join(removing_custom_words)
        
        
        doc = nlp(text)
        wordss = [t.orth_ for t in doc if not t.is_punct | t.is_stop]
        lexical_tokens = [t.lower() for t in wordss if len(t) > 1 and     
        t.isalpha()]
        stems = [spanishstemmer.stem(token) for token in lexical_tokens]
        lista_comen_token.loc[indice_fila] = [fila["Tipo_Comida"],fila["Calificacion"],stems,fila["Calificacion_Comentario"]]
            
    bar1.finish()
    return lista_comen_token

def cambiarCalificacion(resena):
    lista_comen_token=pd.DataFrame(columns=["Tipo_Comida","Calificacion","Comentario","Calificacion_Comentario"])
    
    bar1 = Bar('calificación:', max=len(resena))
    
    for indice_fila, fila in resena.iterrows():
        time.sleep(0.0002)
        bar1.next()
        
        calF=fila['Calificacion_Comentario']
        
        if calF >= 3:
            calF="positive"
        else:
            calF="negative"
        lista_comen_token.loc[indice_fila] = [fila["Tipo_Comida"],calF,fila["Comentario"],fila["Calificacion_Comentario"]]
            
    bar1.finish()
    return lista_comen_token

def agCategoriaComidaCal(resena):
    
    bar1 = Bar('addInf:', max=len(resena))
    for indice_fila, fila in resena.iterrows():
        
        time.sleep(0.0002)
        bar1.next()
        
        comen=fila['Comentario']
        cal=fila['Calificacion_Comentario']
        
        if cal==1:    
            comen = comen+" "+"calificación uno"       
        if cal==2:    
            comen = comen+" "+"calificación dos"   
        if cal==3:    
            comen = comen+" "+"calificación tres"     
        if cal==4:    
            comen = comen+" "+"calificación cuatro"
        if cal==5:    
            comen = comen+" "+"calificación cinco"     
        resena['Comentario']=resena['Comentario'].replace(fila['Comentario'],comen) 
    bar1.finish()
    return resena

def eliminarPeque(resena):
    for fila in resena:
        if len(fila)<5:
            resena.remove(fila)
    return resena

def transLista(resena):
    for indice_fila, fila in resena.iterrows():
        comentarios=fila["Comentario"]
        for l in comentarios:
            text= " ".join(comentarios)
            
        resena.loc[indice_fila] = [fila["Tipo_Comida"],fila["Calificacion"],text,fila['Calificacion_Comentario']]
        
    return resena

def deEmojify(string):
    emoji_pattern  = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    
    return emoji_pattern.sub(r'', string)

def quitarEmoji(resena):
    bar1 = Bar('emoji:', max=len(resena))
    
    for indice_fila, fila in resena.iterrows():
        
        time.sleep(0.000002)
        bar1.next()
        
        comen=fila['Comentario']
        comen = deEmojify(comen)
        
        resena['Comentario']=resena['Comentario'].replace(fila['Comentario'],comen) 
    bar1.finish()
    return resena


    
