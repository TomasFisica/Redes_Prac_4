# -*- coding: utf-8 -*-
"""
Created on Fri May 22 13:04:51 2020

@author: tomas
"""

import numpy as np
import keras
from keras.layers import Dense,Input,Concatenate
from keras.models import Sequential,Model
import numpy as np
import copy
from matplotlib import pyplot as plt
import random
# =============================================================================
# Funciones a utilizar
# =============================================================================
def Gra(dic,lis,color):
    """Grafica cada una de las loss"""
    plt.plot(range(1200),dic[lis[0]]["loss"],color[0],label="{} ejemplos".format(lis[0]))
    lis=lis[1:]
    color=color[1:]
    if len(lis)>0:
        Gra(dic,lis,color)
def Grafica(dic,lis):
    """Funcion para presentar todas las loss"""        
    color=["b","k","r","g","c","m","y"]
    Gra(dic,lis,color)
    plt.ylabel("MSE")
    plt.ylim([0,1])
    plt.xlabel("Epocas")
    plt.legend()

def Generar(num):
    """Generar el diccionario"""
    Mi=Mapeo_logistico()
    Mi.Modelar()
    Mi.Generar_datos(int(num))
    Mi.Entrenar()
    Historia=Mi.Historia
    return Historia.history
   


def Plot_loss__val_loss(num):
    """Graficar loss de entrenamiento y de validacion """
    plt.plot(range(1200),dic[num]["val_loss"],"r",label="{} Validacion".format(num))
    plt.plot(range(1200),dic[num]["loss"],"k",label="{} Ejemplo".format(num))
    plt.ylabel("MSE")
    plt.ylim([0,1])
    plt.xlabel("Epocas")
    plt.legend()
 

class Mapeo_logistico():
    
    def __init__(self):
        self.Modelo=0
        self.Historia=0
        self.x_train=[]
        self.y_train=[]
        self.x_test=[]
        self.y_test=[]
        self.Factor_conversion=0 #ayuda a convertir los facoteres de 
    def Modelar(self):   
        Entrada=Input(shape=(1,))
        Capa_Oculta=Dense(units=5,activation="sigmoid",use_bias=True)(Entrada)
        Concatenacion=Concatenate()([Entrada,Capa_Oculta])
        Saldia=Dense(units=1,activation="linear",use_bias=True)(Concatenacion)
        self.Modelo=Model(inputs=Entrada,outputs=Saldia)
        mse=keras.optimizers.Adam(lr=1e-3)
        self.Modelo.compile(optimizer=mse,loss="mse",metrics=[keras.metrics.MeanSquaredError()])
    def Seleccion(self,rand,train,num_ejem):
        val=random.choice(rand)
        rand.remove(val)
        train.append(val)
        if len(train)<num_ejem:
            self.Seleccion(rand,train,num_ejem)
        
    def Generar_datos(self,num_ejemplo):
        """Generar los datos del mapeo logistico"""       
        datos=list(range(110))
        datos=list(map(lambda arg:arg/100,datos))
        self.Seleccion(datos,self.x_train,num_ejemplo)
        self.y_train=list(map(lambda arg:4*arg*(1-arg),self.x_train))  
        self.Seleccion(datos,self.x_test,10)
        self.y_test=list(map(lambda arg:4*arg*(1-arg),self.x_test))
    def Entrenar(self):
        self.Historia=self.Modelo.fit(x=self.x_train,y=self.y_train,epochs=1200,batch_size=10,validation_data=[self.x_test,self.y_test])
    def Graficar(self):
        His=Historia.history
    def Predecir(self,list_test):
        return self.Modelo.predict(list_test)
def MapeoLogistico():    
    lis=[]
    print("Inicializar la lista con el número de ejemplos. Maximo 7 elementos")    
    while True:
        valor=int(input("Introduzca el numero de ejemplo. Para finalizar introduzca 0 o un negativo: "))
        if valor>0 and len(lis)<7:
            lis.append(valor)
            print(lis)
        else:
            break
    dic={lis[i]:Generar(lis[i]) for i in range(len(lis))}
    Grafica(dic,lis)
    return dic
if __name__=="__main__":
    dic=MapeoLogistico()
    
   









