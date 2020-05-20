import numpy as np
from matplotlib import pyplot as plt
"""
    El programa inicia pidiendo al usuario que inserte un numero que sera el numero de neuronas en la primera capa
    Debe tener especial cuidado a la hora de correr el programa debido a la inicializacion de los pesos (la magnitud 
    inicial de los mismo debe disminuir a medida que aumenta el numero de neuornas) del mismo modo tambien debe
    cuidarse el facor que multiplica los gradientes para la correccion de los pesos
"""
class Neu_tgh():
    "Neurona cuya funcion de activacion es la tangente hiperbolica"
    def __init__(self,num_entrada):
        self.num_entr=num_entrada #Numero de entradas más el bais
        self.Pesos=np.random.randn(self.num_entr+1,1)
        self.Grad_Local=0
        self.Entrada_arr=0
        self.Salida=0
    def Fordware(self,Entradas):
        self.Entrada_arr=np.array(Entradas)
        self.Entrada_arr=np.append(self.Entrada_arr,1)    #Agrego bais
        self.Entrada_arr=self.Entrada_arr.reshape([1,self.num_entr+1])
        self.Salida=self.Entrada_arr@self.Pesos
        return float(np.tanh(self.Salida))
    def Glocal(self,grad_ext):
        self.Grad_Local=1-float(np.square(np.tanh(self.Salida)))
        self.Grad_Local*=grad_ext
    def Backpropagation(self, grad_ext):
        self.Glocal(grad_ext)
        Back=self.Grad_Local*self.Pesos.T[:,:self.num_entr]
        self.Pesos-=(self.Entrada_arr.T*self.Grad_Local)*1e-2
        return Back
def Salidas_Capa(Capa, xtrain):
    """Funcion que realzia el flujo hacia adelante en la red neuronal"""
    Salida=[0]*len(Capa)
    Tipo_Capa=Capa[0].num_entr
    if Tipo_Capa==1:
        for i in range(len(Capa)):
            Salida[i]=Capa[i].Fordware(xtrain[i])
    else:
        for i in range(len(Capa)):
            Salida[i]=Capa[i].Fordware(xtrain)
    return Salida
def Gradientes_Capa(Capa,Grad_ant):
    """Funcion que realiza el flujo hacia atras en la red neuronal"""
    Tipo_Capa=Capa[0].num_entr
    Salida=[0]*len(Capa)
    for i in range(len(Capa)):
        Salida[i]=Capa[i].Backpropagation(Grad_ant[0][i])
    if Tipo_Capa!=1:
        Salida=np.sum(Salida,axis=0)
    return Salida
def Test(PrimeraCapa,SegundaCapa,x_train,y_train):
    Score1=0;Score2=0;Score3=0;loss=0;acierto=0
    for i in range(200):
        Score1=Salidas_Capa(PrimeraCapa,x_train[i])
        Score2=Salidas_Capa(SegundaCapa,Score1)
        Score3=Ultima_Neurona.Fordware(Score2)
        """Determino la loss"""
        loss+=np.square(y_train[i]-Score3)  #Loss de MSE
        if (Score3>0 and y_train[i]>0)or (Score3<0 and y_train[i]<0):
            acierto+=1
    return float(loss)/200,acierto
def Accuarry():
    pass
"""Variables a utilizar"""
"""Primero inicializo el numero de entradas"""
Num=int(input("Ingrese el numeor de entradas que desea para su XOR"))#Numero de neuronas por capa
"""Genero las neuronas"""
Primera_Capa=[0]*Num
Segunda_Capa=[0]*Num
Ultima_Neurona=Neu_tgh(Num)
for i in range(Num):    #Creo las neuronas de las dos capas
    Primera_Capa[i]=Neu_tgh(1)  #Una entrada
    Segunda_Capa[i]=Neu_tgh(Num)    #Al ser la segunda capa totalmente conectada con la primera
"""Genero los Datos {En dos lineas ;) }"""
x_train=np.random.choice([1,-1],[2000,Num])
y_train=np.prod(x_train,axis=1).reshape(2000,1)

x_test=np.random.choice([1,-1],[200,Num])
y_test=np.prod(x_test,axis=1).reshape(200,1)
#Divido los datos en 10 bloques de 2000
num_bloq=10
num_train=2000


los=0
Score_1=np.array([])
Score_2=np.array([])
Score_3=np.array([])
Grad_Loss=np.array([])
Grad_Primera_Capa=np.array([])
Grad_Segunda_Capa=np.array([])
Loss=[]
Loss_test=[]
Acc=[]
Acc_test=[]
losstest=0
accu=0
accu_test=0
acierto_Tra=0
Lus=0

"""Ahora realizo el entrenamiento"""
for j in range(30):
    """Desordeno los ejemplos"""
    max_aux=np.insert(x_train,[x_train.shape[1]],y_train,1)
    np.random.shuffle(max_aux)                  #Desorganizo
    x_train=max_aux[:,:max_aux.shape[1]-1]      #Re armo la matriz de la info de fotos
    y_train=max_aux[:,max_aux.shape[1]-1:]   #RE armo el vector de al clasificacion correcta 
    i=0
    for k in range(10):
        for i in range(200):
            
            Score_1=Salidas_Capa(Primera_Capa,x_train[i])
            Score_2=Salidas_Capa(Segunda_Capa,Score_1)
            Score_3=Ultima_Neurona.Fordware(Score_2)
            """Determino la loss"""
            los+=np.square(y_train[i]-Score_3)  #Loss de MSE
            """Voy a por los gradientes"""
            Grad_Loss=Ultima_Neurona.Backpropagation((y_train[i]-Score_3)*(-2))
            Grad_Segunda_Capa=Gradientes_Capa(Segunda_Capa,Grad_Loss)
            Grad_Primera_Capa=Gradientes_Capa(Primera_Capa,Grad_Segunda_Capa)
            """Determino la acc"""
            if (Score_3>0 and y_train[i]>0)or (Score_3<0 and y_train[i]<0):
                acierto_Tra+=1
            
            i+=1
    Losstest,accu=Test(Primera_Capa,Segunda_Capa,x_test,y_test)
    Loss_test.append(Losstest)
    Acc_test.append(accu)
    Acc.append(acierto_Tra/10)
    acierto_Tra=0
    Loss.append(float(los)/2000)
    los=0
#Debido a la natiraleza de neustra acc, ver cuantos aciertos tiene en 1000 pruebas, puede calcularse sin necesidad de funcion
#y mantener un codigo legible
"""Grafico la loss y la acc"""
f,(ax1)=plt.subplots(1)
ax1.plot(range(30),Loss,color="K",label="Loss train");ax1.plot(range(30),Loss_test,color="r",label="Loss teste")
plt.legend()
"""Grafico la acc"""
f,(ax2)=plt.subplots(1)
ax2.plot(range(30),Acc,color="K",label="Acc train");ax2.plot(range(30),Acc_test,color="r",label="Acc teste")
plt.legend()
    
    
"""Graficar la loss de train y la de test
plt.plot(range(150),Loss,cloro="K",label="Loss train");plt.plot(range(150),Loss_test,color="r",label="Loss test")
plt.legend()
"""
