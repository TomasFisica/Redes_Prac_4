# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 19:13:27 2020

@author: tomas
"""
import numpy as np
from matplotlib import pyplot as plt
# =============================================================================
# Funciones a utilizar
# =============================================================================

class Hopfield():
    """Modelo de RNA tipo Hopfield"""
    
    def __init__(self,num_N,Alfa,ruido):
        """Inicializacion de los valores"""
        self.num_N=num_N #Numero unidades de la red
        self.num_P=int(Alfa*num_N)   #Numero de patrones
        self.Alfa=Alfa              #Relacion entre P y N "P/N"
        self.Patrones=self.Generar_Patrones(ruido)
        self.Matriz_Conexion=self.Generar_Matriz_Conexion()      
            
    def Generar_Patrones(self,ruido="defecto"):
        """Método para generar los patrones. Con ruido o sin ruido"""
        patrones=[]
        generacion1=lambda arg: 1 if arg>0 else -1
        generacion2=lambda arg: list(map(generacion1, arg))
        if ruido=="Sin_Ruido":
            patrones=np.random.normal(size=(self.num_P,self.num_N))
            patrones=np.array(list(map (generacion2,patrones)))
            return patrones
        else:
            pass
            
    
    def Generar_Matriz_Conexion(self):
        """Método para generar la matriz de conexiones"""
        producto_externo=lambda arg: arg.reshape([self.num_N,1])@arg.reshape([1,self.num_N])
        matriz_sumada=list(map(producto_externo, self.Patrones))
        matriz_conexion=np.zeros([self.num_N,self.num_N])
        for i in range(self.num_P):
            matriz_conexion+=matriz_sumada[i]
        #la matriz de abajo al multiplicarse por una matriz, devuelve una matriz con la diagonal = 0
        diagonalizar=np.ones_like(matriz_conexion)-np.diag([1 for i in range(self.num_N)]) 
        matriz_conexion=matriz_conexion*diagonalizar
        return matriz_conexion/self.num_N
    
    def Calcular_overlap(self,):
        """Funcion para calcular cada overlap"""
        return
    
    def Scalon(self, matriz):
        """Método para aplicar funcion escalon a matriz"""
        matriz=list(map(lambda arg: 1 if arg>0 else (-1 if arg<0 else 0),matriz) )
        return np.array(matriz)

# =============================================================================
#     Método funcional
#     def Dinamica_red(self, entrada):
#         """Método para evolucionar la red"""
#         output=self.Matriz_Conexion@entrada.reshape([self.num_N,1])
#         output=self.Scalon(output)
#         if (output!=entrada).all():
#             return self.Dinamica_red(output)
#         else: 
#             return output        
# =============================================================================
    def Dinamica_red(self, entrada):
        """Método para evolucionar la red"""
        while True:
            output=self.Matriz_Conexion@entrada.reshape([self.num_N,1])
            output=self.Scalon(output)
            if (output==entrada).all():
                break
            entrada=output
        return output        
    def Generar_distribucion_overlap(self):
        """Funcion para generar la distribucion de los overlaps"""
        Lista_S_Fijos=[]
        Lista_Overlap=[]
        for i in range(len(self.Patrones)):
            Lista_S_Fijos=self.Dinamica_red(self.Patrones[i])
            Lista_Overlap+=[float(Lista_S_Fijos.reshape([1,self.num_N])@self.Patrones[i].reshape([self.num_N,1]))/self.num_N]
        return Lista_Overlap

    def Distribuir(self,Bins=2):
        """Método que genera la distribucion de overlap y mostrar"""
        plt.hist(self.Generar_distribucion_overlap())

Mi=Hopfield(4000, 0.12, "Sin_Ruido")
Mi.Distribuir()