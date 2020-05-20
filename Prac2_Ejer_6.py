import numpy as np
from matplotlib import pyplot as plt

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
        self.Pesos-=(self.Entrada_arr.T*self.Grad_Local)*1e-1
        return Back
def Loss_test(N11,N12,N21,N22,N33,xtrain,ytrain):
    """Determinar loss y accc para la primera arquitectura"""
    Score1=0;Score2=0;Score3=0;loss=0;acierto=0
    for i in range(100):
        Score1=[N11.Fordware(xtrain[i][0]),N12.Fordware(xtrain[i][1])]
        Score2=[N21.Fordware(Score1),N22.Fordware(Score1)]
        Score3=N33.Fordware(Score2)
        loss+=np.square(ytrain[i]-Score3)  #Loss de MSE
        if (Score3>0 and ytrain[i]>0)or (Score3<0 and ytrain[i]<0):
            acierto+=1
    return float(loss)/100,acierto
def Loss_test2(N211,N212,N222,N233,xtrain,ytrain):
    """Determinar loss y acc para la segunda arquitectura"""
    Score1=0;Score2=0;Score3=0;loss=0;acierto=0
    for i in range(100):
        Score1=[N2_11.Fordware(xtrain[i][0]),N212.Fordware(xtrain[i][1])]
        Score2=[N222.Fordware(Score1)]
        Score1.append(Score2[0])
        Score3=N233.Fordware(Score1)
        """Determino la loss del entrenamiento"""
        loss=np.square(ytrain[i]-Score3)  #Loss de MSE
        if (Score3>0 and ytrain[i]>0)or (Score3<0 and ytrain[i]<0):
            acierto+=1
    return float(loss)/100,acierto  
    
"""Variables a utilizar"""
x_train=np.random.choice([1,-1],[3000,2])
y_train=np.prod(x_train,axis=1).reshape(3000,1)
x_test=np.random.choice([1,-1],[100,2])
y_test=np.prod(x_test,axis=1).reshape(100,1)
loss=0
loss_test=0
accu_test=0
acc=0
Score_1=0
Score_2=0
Score_3=0
Grad_33=0
Grad_21=0
Grad_22=0
Grad_aux=0
Loss=[]
Losstest=[]
Accu_test=[]
Accu=[]
i=0

"""Ahora defino las neuronas"""
N_11=Neu_tgh(1)
N_12=Neu_tgh(1)
N_21=Neu_tgh(2)
N_22=Neu_tgh(2)
N_33=Neu_tgh(2)
"""Ahora realizo el entrenamiento"""
for j in range(30):
    for k in range(100):
        
        Score_1=[N_11.Fordware(x_train[i][0]),N_12.Fordware(x_train[i][1])]
        Score_2=[N_21.Fordware(Score_1),N_22.Fordware(Score_1)]
        Score_3=N_33.Fordware(Score_2)
        """Determino la loss del entrenamiento"""
        loss=np.square(y_train[i]-Score_3)  #Loss de MSE
        """Determino el gradiente"""
        Grad_33=N_33.Backpropagation((y_train[i]-Score_3)*(-2))
        Grad_21=N_21.Backpropagation(Grad_33[0][0])
        Grad_22=N_22.Backpropagation(Grad_33[0][1])
        N_11.Backpropagation(Grad_21[0][0]+Grad_22[0][0])
        N_12.Backpropagation(Grad_21[0][1]+Grad_22[0][1])
        """Determino el accu"""
        if (Score_3>0 and y_train[i]>0)or (Score_3<0 and y_train[i]<0):
            acc+=1
        
        i+=1
    loss_test,accu_test= Loss_test(N_11,N_12,N_21,N_22,N_33,x_test,y_test)
    Losstest.append(loss_test)
    Accu_test.append(accu_test)
    Loss.append(float(loss)/100)
    loss=0
    Accu.append(acc)
    acc=0
    
"""Grafico los resultados"""
"""Grafico la loss"""
f,(ax1)=plt.subplots(1)
ax1.plot(range(30),Loss,color="K",label="Loss train");ax1.plot(range(30),Losstest,color="r",label="Loss teste")
plt.legend()
"""Grafico la acc"""
f,(ax2)=plt.subplots(1)
ax2.plot(range(30),Accu,color="K",label="Acc train");ax2.plot(range(30),Accu_test,color="r",label="Acc teste")
plt.legend()

"""Ahora realizo el ejercicio con la segunda arquitectura"""

N2_11=Neu_tgh(1)
N2_12=Neu_tgh(1)
N2_22=Neu_tgh(2)
N2_33=Neu_tgh(3)

Loss2=[]
Losstest2=[]
Accu_test2=[]
Accu2=[]
i=0


"""Ahora realizo el entrenamiento"""
for j in range(30):
    for k in range(100):
        
        Score_1=[N2_11.Fordware(x_train[i][0]),N2_12.Fordware(x_train[i][1])]
        Score_2=[N2_22.Fordware(Score_1)]
        Score_1.append(Score_2[0])
        Score_3=N2_33.Fordware(Score_1)
        """Determino la loss del entrenamiento"""
        loss=np.square(y_train[i]-Score_3)  #Loss de MSE
        """Determino el gradiente"""
        Grad2_33=N2_33.Backpropagation((y_train[i]-Score_3)*(-2))
        Grad2_22=N2_22.Backpropagation(Grad2_33[0][2])
        N2_11.Backpropagation(Grad2_22[0][0]+Grad2_33[0][0])
        N2_12.Backpropagation(Grad2_22[0][1]+Grad2_33[0][1])
        """Determino el accu"""
        if (Score_3>0 and y_train[i]>0)or (Score_3<0 and y_train[i]<0):
            acc+=1
        
        i+=1
    loss_test,accu_test= Loss_test2(N2_11,N2_12,N2_22,N2_33,x_test,y_test)
    Losstest2.append(loss_test)
    Accu_test2.append(accu_test)
    Loss2.append(float(loss)/100)
    loss=0
    Accu2.append(acc)
    acc=0
    
"""Grafico los resultados"""
"""Grafico la loss"""
f,(ax3)=plt.subplots(1)
ax3.plot(range(30),Loss2,color="K",label="Loss train 2");ax3.plot(range(30),Losstest2,color="r",label="Loss teste 2")
plt.legend()
"""Grafico la acc"""
f,(ax4)=plt.subplots(1)
ax4.plot(range(30),Accu2,color="K",label="Acc train 2");ax4.plot(range(30),Accu_test2,color="r",label="Acc teste 2")
plt.legend()
