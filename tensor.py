# -*- coding: utf-8 -*-
"""
Created on Thu May 21 10:35:58 2020

@author: tomas
"""

import numpy as np
import keras
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import copy
from matplotlib import pyplot as plt
# =============================================================================
# Funciones a utilizar
# =============================================================================

# =============================================================================
# Convertir los -1 en 1
# =============================================================================
# Convertir -1 en 0
"""
Function=lambda arg : 0 if arg==-1 else arg
Function2=lambda arg2: list(map(Function,arg2))
# Volver a convertir en array
Function3=lambda arg3: np.array(list(map(list, map(Function2,arg3))))
de=Function3(y_train)
de=np.array(de)
"""

class Xor():
    
    def __init__(self,num_neuronas,num_entrada):
        self.Num_Neuronas=num_neuronas
        self.Num_Entrada=num_entrada
        self.X_Train=0
        self.Y_Train=0
        self.Modelo=0
        self.Historia=0
        self.copiado1=lambda arg: [i.append(1) for i in arg]
        self.copiado2=lambda arg: [i.append(0) for i in arg]    
        self.xor=lambda arg: 1 if np.count_nonzero(arg)%2==0 else 0
        self.xor_2=lambda arg1: list(map(self.xor,arg1))

    def Generar_x_train(self,lista):
        lis1=copy.deepcopy(lista)
        lis2=copy.deepcopy(lista)
        self.copiado2(lis2)
        self.copiado1(lis1)
        if len(lis1+lis2)==2**self.Num_Entrada:
            return lis1+lis2
            pass
        return self.Generar_x_train(lis1+lis2)
    def Generar_datos(self):
        """Genero los datos"""
        self.X_Train=self.Generar_x_train([[1],[0]])
        self.X_Train=np.array(self.X_Train)
        self.Y_Train=np.array(self.xor_2(self.X_Train)).reshape([2**self.Num_Entrada,1])
        

    
    def Modelar(self):
        self.Modelo = Sequential()
        self.Modelo.add(Dense(units=self.Num_Neuronas,activation='tanh',input_dim=self.Num_Entrada,use_bias=True))
        self.Modelo.add(Dense(units=1,activation='sigmoid'))
        sgd=keras.optimizers.SGD(lr=0.01)
        self.Modelo.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['binary_accuracy'])
        
    def Entrenar(self,num_epocas):
        try:
            self.Historia=self.Modelo.fit(self.X_Train,self.Y_Train,epochs=num_epocas,batch_size=1,shuffle=True)
        except:
            print("Debe ingresar un entero")
    def Graficar(self):
        Dicc_history=self.Historia.history
        Epohs=len(Dicc_history["loss"])
                
        plt.subplot(211)
        p1,=plt.plot(range(Epohs),Dicc_history["loss"],"r")
        plt.legend(["Loss"])
        plt.title("Numero de Neuronas {}. Numero de entradas {}".format(self.Num_Neuronas,self.Num_Entrada))
        plt.subplot(212)
        p2,=plt.plot(range(Epohs),Dicc_history["binary_accuracy"],"r")
        plt.legend(["Acc"])
def Programa(neu,inp,epo):    
    
    Mi_Xor=Xor(neu, inp)
    Mi_Xor.Generar_datos()
    Mi_Xor.Modelar()
    Mi_Xor.Entrenar(epo)
    Mi_Xor.Graficar()
    
if __name__=="__main__":
    print("Para salir ingrese 0 neuronas y 0 entradas")
    while True:
        Neurona=int(input("Neurona: "))
        Entrada=int(input("Entrada: "))
        Epocas=int(input("Epocas: "))
        if Neurona==Entrada and Neurona==0:
            break
        Programa(Neurona,Entrada,Epocas)
        
        
 