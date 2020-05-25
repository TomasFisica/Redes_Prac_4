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

# =============================================================================
# Funciones a utilizar
# =============================================================================
def Generar(arg):
    """Generar los datos de entrenamiento"""
    if len(arg)==6:
        ytra=arg[1:]
        arg.remove(arg[5])
        return ytra
    else:
        Valor=arg[len(arg)-1]
        Valor_a_pasar=4*Valor*(1-Valor)
        arg.append(Valor_a_pasar)
        return Generar(arg)

def Datos(xtrain,ytrain,ini,fin):
    

class Mapeo_logistico():
    
    def __init__(self):
        self.Modelo=0
        self.Historia=0
        self.x_train=0
        self.y_train=0
        self.Factor_conversion=0 #ayuda a convertir los facoteres de 
    def Modelar(self):   
        Entrada=Input(shape=(1,))
        Capa_Oculta=Dense(units=5,activation="sigmoid",use_bias=True)(Entrada)
        Concatenacion=Concatenate()([Entrada,Capa_Oculta])
        Saldia=Dense(units=1,activation="linear",use_bias=True)(Concatenacion)
        self.Modelo=Model(inputs=Entrada,outputs=Saldia)
        mse=keras.optimizers.Adam(lr=1e-3)
        self.Modelo.compile(optimizer=mse,loss="mse",metrics=[keras.metrics.MeanSquaredError()])
    def Generar_datos(self,ini,fin):
        """Generar los datos del mapeo logistico"""    
        self.x_train=list(range(ini,fin))
        # self.x_train=keras.utils.normalize(self.x_train)  #Normalizo para evitar trabajar ocn grande datos     
        xtrai=list(map(lambda arg:-arg/50,self.x_train))
        ytrai=list(map(lambda arg:4*arg*(1-arg),xtrai))
        self.x_train=list(map(lambda arg:arg/50,self.x_train))
        self.y_train=list(map(lambda arg:4*arg*(1-arg),self.x_train))+ytrai    
        self.x_train=self.x_train+xtrai
    def Entrenar(self):
        self.Historia=self.Modelo.fit(x=self.x_train,y=self.y_train,epochs=1000,batch_size=1)
    def Graficar(self):
        His=Historia.history
    def Predecir(self,list_test):
        return self.Modelo.predict(list_test)
Mi=Mapeo_logistico()
Mi.Modelar()
Mi.Generar_datos(0,50)
Mi.Entrenar()
num=-5
porcentaje=[(Mi.Predecir([num])-4*num*(1-num))/(4*num*(1-num))*100 for num in lista if num!=0]
porcentaje=list(map(float,porcentaje))
Modelo.predict([0])





x_train=list(range(50))
x_train2=list(range(50))
# self.x_train=keras.utils.normalize(self.x_train)  #Normalizo para evitar trabajar ocn grande datos     
x_train2=list(map(lambda arg:-arg/50,x_train2))  
x_train=list(map(lambda arg:-arg/50,x_train2))  
y_train=list(map(lambda arg:4*arg*(1-arg),x_train))    
y_train2=list(map(lambda arg:4*arg*(1-arg),x_train2))    


Mi.Modelo.save("C:/Users/tomas/Escritorio")



initializer = keras.initializers.RandomNormal(mean=0., stddev=1.)
layer = keras.layers.Dense(3, kernel_initializer=initializer)
