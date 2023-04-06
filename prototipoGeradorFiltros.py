# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:20:35 2023

@author: bruno
"""
import math
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from functools import reduce
from operator import mul

def PrototypeFilter(signal, factorX, factorY):
    vectorFiltered=signal.copy()
    
    for i in range(len(signal)):
        vectorFiltered[i]=factorX[0]*signal[i]
        
        for j in range(1, len(factorX)):
            if i>=j:
                vectorFiltered[i]+=factorX[j]*signal[i-j]
                
    """
    filterCopy=vectorFiltered.copy()
    
    for i in range(len(vectorFiltered)):
        
        for j in range(1, len(factorY)):
            if i>=j:            
                vectorFiltered[i]+=factorY[j]*filterCopy[i-j]        
                
    """            
               
    for j in range(1, len(factorY)):
        filterCopy=vectorFiltered.copy()    
        for i in range(0, len(vectorFiltered)):
            if i>=j:
                vectorFiltered[i]+=factorY[j]*filterCopy[i-j]        
                
    
    
        
    
    max_val = max(abs(vectorFiltered))

    vectorFiltered = vectorFiltered/max_val   
    

    

    return vectorFiltered

def multiply(A, B, m, n):
  
    prod = [0]* (m + n - 1) 
    
    for i in range(m):
          
        for j in range(n):
            prod[i + j] += A[i] * B[j];
    return prod;

def FilterProjectButter(fp, fs, atp, ats, minN):


    
    oms=fs/fp



    n=math.log10((math.pow(10, ats/10)-1)/(math.pow(10, atp/10)-1))/2*math.log10(oms)

    

    if n<0:
        n=n*-1
        highpass=True
    else:
        highpass=False
    
    
    
    

    ats1=10*math.log(math.pow(oms, 2*n)*(math.pow(10, atp/10) -1)+1)

    omc=oms/(math.pow((math.pow(10, ats1/10)-1), 1/(2*n)))

    

    aux=2*3.14*fp
    
    
    
    
    
    n=int(math.ceil(n))
    
    if minN>n:
        n=minN

    polos=[]

    vet=[]

    for i in range(0, n):
        if highpass:
            polos.append([[math.cos(((math.pi+2*math.pi*i)/(2*(-n))+(math.pi/2)))], [math.sin(((math.pi+2*math.pi*i)/(2*(-n))+(math.pi/2)))]])
        else:
           polos.append([[math.cos(((math.pi+2*math.pi*i)/(2*n)+(math.pi/2)))], [math.sin(((math.pi+2*math.pi*i)/(2*n)+(math.pi/2)))]])
           
         
    

    polos=np.array(polos)
    
   
   
    polos= polos.astype(float)

    polos2=[]

    for i in range(0, n):
        polos2.append([[[1], [1], [0]], [[0], [-aux*omc*float(polos[i][0])], [-aux*omc*float(polos[i][1])]]])
        vet.append(0)

    polos2=np.array(polos2)

       

    polos3=[]
    



    for i in range(0, int(math.pow(2, n))):
        polos3.append([[polos2[0][vet[0]][0]], [polos2[0][vet[0]][1]], [polos2[0][vet[0]][2]]])
    
        for j in range(1, n):
            polos3[i][0]=int(polos3[i][0]+polos2[j][vet[j]][0])
            a=float((polos3[i][1]*polos2[j][vet[j]][1])+(polos3[i][2]*polos2[j][vet[j]][2]))
            b=float((polos3[i][1]*polos2[j][vet[j]][2])+(polos3[i][2]*polos2[j][vet[j]][1]))
            polos3[i][1]=a
            polos3[i][2]=b
    
        
        j=n-1
        if vet[j]==0:
            vet[j]=1
        else:
            while vet[j]==1:
                vet[j]=0
                if j>0:
                    if vet[j-1]==0:
                        vet[j-1]=1
                    else:
                        j=j-1
                else:
                    vet[j]=0
                
                
        
     
            
   

    dividendos=[(math.pow((aux*omc), n))]
    
    

    divisores=[]



    for i in range(0, n+1):
        divisores.append([i, 0.0])
        for j in range(0, len(polos3)):
            if i==polos3[j][0]:
                divisores[i][1]+=float(polos3[j][1])
                
        if (divisores[i][1]<0.001) & (divisores[i][1]>0.0):
            divisores[i][1]=0
        elif (divisores[i][1]>-0.001) & (divisores[i][1]<0.0):
            divisores[i][1]=0
    
    
    

    auxZ1=[-1, 1]
    auxZ2=[1, 1]
    auxZ3=multiply(auxZ1, [2*fp], len(auxZ1), 1)

    protoZ1=[1]
    protoZ2=[1]

    list1=[[1]]
    list2=[[1]]

    for i in range(0, n):
        protoZ1=multiply(protoZ1, auxZ1, len(protoZ1), len(auxZ1))
        protoZ2= multiply(protoZ2, auxZ2, len(protoZ2), len(auxZ2))
        list1.append(protoZ1)
        list2.append(protoZ2)


    divList=[]
    
    
     
    

    for i in range(0, n):
        dividendos=multiply(dividendos, auxZ2, len(dividendos), len(auxZ2))
        #divList.append(multiply([divisores[i][1]], list1[i], 1, len(list1[i])))
    
   
    
    for i in range(0, len(divisores)):
        grauN=divisores[i][0]
        auxDiv=[divisores[i][1]]
        for j in range(0, n):
            if j<grauN:
                auxDiv=multiply(auxDiv, auxZ3, len(auxDiv), len(auxZ3))
            else:
                auxDiv=multiply(auxDiv, auxZ2, len(auxDiv), len(auxZ2))
        divList.append(auxDiv)
        
    
    
    
   
    
    divisores=[]+divList[0]

    for i in range(1, len(divList)):
        divisores = [elemA + elemB for elemA, elemB in zip(divisores, divList[i])]

    
    
    
    
    if highpass:
        Y1=dividendos[len(dividendos)-1]
    else:
        Y1=divisores[len(divisores)-1]
        
    if Y1<0:
        Y2=-Y1
    else:
        Y2=Y1
    
    
    for i in range(len(divisores)):
        divisores[i]=divisores[i]/Y1
    
    
    for i in range(len(dividendos)):
        dividendos[i]=dividendos[i]/Y2
    
    
    
    fatorX=[]
    fatorY=[]
    
    if highpass:
        for i in range(len(dividendos)-1, -1, -1):
            fatorY.append(dividendos[i])
        for i in range(len(divisores)-1, -1, -1):
            fatorX.append(divisores[i])
    else:
        for i in range(len(dividendos)-1, -1, -1):
            fatorX.append(dividendos[i])
        for i in range(len(divisores)-1, -1, -1):
            fatorY.append(-divisores[i])
    
    
    vector=[1.0]*200


    vector=np.array(vector) 

    vector=PrototypeFilter(vector, fatorX, fatorY)

    if vector[100]<0:
        for i in range(len(fatorX)):
            fatorX[i]/=(-1)
    
    
    
        

    

    return fatorX, fatorY




signals, fields = wfdb.rdsamp('104', channels = [0], sampto=5000)





fp=15
fs=60
atp=1
ats=50

fatorXlow, fatorYlow=FilterProjectButter(fp, fs, atp, ats, 10)
vectorLPF= PrototypeFilter(signals, fatorXlow, fatorYlow)


fp=5
fs=1
atp=1
ats=50



fatorXhigh, fatorYhigh=FilterProjectButter(fp, fs, atp, ats, 12)


vectorHPF= PrototypeFilter(vectorLPF, fatorXhigh, fatorYhigh)
#vectorHPF= PrototypeFilter(signals, fatorXhigh, fatorYhigh)



plt.plot(signals)
#plt.plot(vectorLPF)
plt.plot(vectorHPF)



#%%
import math
import numpy as np
import wfdb
import matplotlib.pyplot as plt

def PrototypeFilter(signal, factorX, factorY):
    vectorFiltered=signal.copy()
    
    for i in range(len(signal)):
        vectorFiltered[i]=factorX[0]*signal[i]
        
        for j in range(1, len(factorX)):
            if i>=j:
                vectorFiltered[i]+=factorX[j]*signal[i-j]
                
    """
    filterCopy=vectorFiltered.copy()
    
    for i in range(len(vectorFiltered)):
        
        for j in range(1, len(factorY)):
            if i>=j:            
                vectorFiltered[i]+=factorY[j]*filterCopy[i-j]        
                
    """            
               
    for j in range(1, len(factorY)):
        filterCopy=vectorFiltered.copy()    
        for i in range(0, len(vectorFiltered)):
            if i>=j:
                vectorFiltered[i]+=factorY[j]*filterCopy[i-j]        
                
    
    
        
    
    max_val = max(abs(vectorFiltered))

    vectorFiltered = vectorFiltered/max_val   
    

    

    return vectorFiltered

def multiply(A, B, m, n):
  
    prod = [0]* (m + n - 1) 
    
    for i in range(m):
          
        for j in range(n):
            prod[i + j] += A[i] * B[j];
    return prod;

def FilterProjectCheb1(fp, fs, atp, ats, rp, minN):


    
    oms=fs/fp


    n=math.log10((math.pow(10, ats/10)-1)/(math.pow(10, atp/10)-1))/2*math.log10(oms)
    #n=np.arccosh(math.sqrt(((10 ** (0.1*fs)) - 1.0) / ((10 ** (0.1 * fp)) - 1.0))) /np.arccosh(oms)

    

    if n<0:
        n=n*-1
        highpass=True
    else:
        highpass=False
    
    
    
    

    ats1=10*math.log(math.pow(oms, 2*n)*(math.pow(10, atp/10) -1)+1)

    omc=oms/(math.pow((math.pow(10, ats1/10)-1), 1/(2*n)))

    

    aux=2*3.14*fp
    
    
    
   

    i=1
    while i<n:
        i=i+1
    
    n=i
    
    if minN>n:
        n=minN
        
    #n=int(math.ceil(n))  
    
    if minN>n:
        n=minN

    polos=[]

    vet=[]
    
    v=(1/n)*np.arcsinh(1/math.sqrt((10**(rp/10))-1))

    for i in range(0, n):
        if highpass:
            polos.append([[-math.sin((3.14159265*((2*i)-1))/(2*(-n)))*math.sinh(v)], [-math.cos((3.14159265*((2*i)-1))/(2*(-n)))*math.cosh(v)]])
        else:
           polos.append([[-math.sin((3.14159265*((2*i)-1))/(2*n))*math.sinh(v)], [-math.cos((3.14159265*((2*i)-1))/(2*n))*math.cosh(v)]])
           
         
    

    polos=np.array(polos)
    
   
    polos= polos.astype(float)

    polos2=[]

    for i in range(0, n):
        polos2.append([[[1], [1], [0]], [[0], [-aux*omc*float(polos[i][0])], [-aux*omc*float(polos[i][1])]]])
        vet.append(0)

    polos2=np.array(polos2)

       

    polos3=[]
    



    for i in range(0, int(math.pow(2, n))):
        polos3.append([[polos2[0][vet[0]][0]], [polos2[0][vet[0]][1]], [polos2[0][vet[0]][2]]])
    
        for j in range(1, n):
            polos3[i][0]=int(polos3[i][0]+polos2[j][vet[j]][0])
            a=float((polos3[i][1]*polos2[j][vet[j]][1])+(polos3[i][2]*polos2[j][vet[j]][2]))
            b=float((polos3[i][1]*polos2[j][vet[j]][2])+(polos3[i][2]*polos2[j][vet[j]][1]))
            polos3[i][1]=a
            polos3[i][2]=b
    
        
        j=n-1
        if vet[j]==0:
            vet[j]=1
        else:
            while vet[j]==1:
                vet[j]=0
                if j>0:
                    if vet[j-1]==0:
                        vet[j-1]=1
                    else:
                        j=j-1
                else:
                    vet[j]=0
                
                
        
     
            
   

    dividendos=[(math.pow((aux*omc), n))]
    
    

    divisores=[]



    for i in range(0, n+1):
        divisores.append([i, 0.0])
        for j in range(0, len(polos3)):
            if i==polos3[j][0]:
                divisores[i][1]+=float(polos3[j][1])
                
        if (divisores[i][1]<0.001) & (divisores[i][1]>0.0):
            divisores[i][1]=0
        elif (divisores[i][1]>-0.001) & (divisores[i][1]<0.0):
            divisores[i][1]=0
    
    
    

    auxZ1=[-1, 1]
    auxZ2=[1, 1]
    auxZ3=multiply(auxZ1, [2*fp], len(auxZ1), 1)

    protoZ1=[1]
    protoZ2=[1]

    list1=[[1]]
    list2=[[1]]

    for i in range(0, n):
        protoZ1=multiply(protoZ1, auxZ1, len(protoZ1), len(auxZ1))
        protoZ2= multiply(protoZ2, auxZ2, len(protoZ2), len(auxZ2))
        list1.append(protoZ1)
        list2.append(protoZ2)


    divList=[]
    
    
     
    

    for i in range(0, n):
        dividendos=multiply(dividendos, auxZ2, len(dividendos), len(auxZ2))
        #divList.append(multiply([divisores[i][1]], list1[i], 1, len(list1[i])))
    
   
    
    for i in range(0, len(divisores)):
        grauN=divisores[i][0]
        auxDiv=[divisores[i][1]]
        for j in range(0, n):
            if j<grauN:
                auxDiv=multiply(auxDiv, auxZ3, len(auxDiv), len(auxZ3))
            else:
                auxDiv=multiply(auxDiv, auxZ2, len(auxDiv), len(auxZ2))
        divList.append(auxDiv)
        
    
    
    
   
    
    divisores=[]+divList[0]

    for i in range(1, len(divList)):
        divisores = [elemA + elemB for elemA, elemB in zip(divisores, divList[i])]
    
    
    
    if highpass:
        Y1=dividendos[len(dividendos)-1]
    else:
        Y1=divisores[len(divisores)-1]
        
    if Y1<0:
        Y2=-Y1
    else:
        Y2=Y1
    
    
    for i in range(len(divisores)):
        divisores[i]=divisores[i]/Y1
    
    
    for i in range(len(dividendos)):
        dividendos[i]=dividendos[i]/Y2
    
    

    fatorX=[]
    fatorY=[1.0]
    
    
    if highpass:
        for i in range(len(dividendos)-1, -1, -1):
            fatorY.append(dividendos[i])
        for i in range(len(divisores)-1, -1, -1):
            fatorX.append(divisores[i])
    else:
        for i in range(len(dividendos)-1, -1, -1):
            fatorX.append(dividendos[i])
        for i in range(len(divisores)-2, -1, -1):
            fatorY.append(-divisores[i])
        

    vector=[1.0]*500


    vector=np.array(vector) 

    vector=PrototypeFilter(vector, fatorX, fatorY)

    if vector[100]<0:
        for i in range(len(fatorX)):
            fatorX[i]/=(-1)

    return fatorX, fatorY




signals, fields = wfdb.rdsamp('105', channels = [0], sampto=5000)

fp=15
fs=60
atp=1
ats=50

fatorXlow, fatorYlow=FilterProjectCheb1(fp, fs, atp, ats, 0.1, 4)
vectorLPF= PrototypeFilter(signals, fatorXlow, fatorYlow)

fp=5
fs=1
atp=1
ats=50



fatorXhigh, fatorYhigh=FilterProjectCheb1(fp, fs, atp, ats, 0.1, 3)


#vectorHPF= PrototypeFilter(vectorLPF, fatorXhigh, fatorYhigh)

vectorHPF= PrototypeFilter(signals, fatorXhigh, fatorYhigh)

plt.plot(signals)
plt.plot(vectorLPF)
plt.plot(vectorHPF)



#%%

import math
import numpy as np
import wfdb
import matplotlib.pyplot as plt

def PrototypeFilter(signal, factorX, factorY):
    vectorFiltered=signal.copy()
    
    for i in range(len(signal)):
        vectorFiltered[i]=factorX[0]*signal[i]
        
        for j in range(1, len(factorX)):
            if i>=j:
                vectorFiltered[i]+=factorX[j]*signal[i-j]
                
    """
    filterCopy=vectorFiltered.copy()
    
    for i in range(len(vectorFiltered)):
        
        for j in range(1, len(factorY)):
            if i>=j:            
                vectorFiltered[i]+=factorY[j]*filterCopy[i-j]        
                
    """            
               
    for j in range(1, len(factorY)):
        filterCopy=vectorFiltered.copy()    
        for i in range(0, len(vectorFiltered)):
            if i>=j:
                vectorFiltered[i]+=factorY[j]*filterCopy[i-j]        
                
    
    
        
    
    max_val = max(abs(vectorFiltered))

    vectorFiltered = vectorFiltered/max_val   
    

    

    return vectorFiltered

def multiply(A, B, m, n):
  
    prod = [0]* (m + n - 1) 
    
    for i in range(m):
          
        for j in range(n):
            prod[i + j] += A[i] * B[j];
    return prod;

def FilterProjectCheb2(fp, fs, atp, ats, rp, minN):


    
    oms=fs/fp


    n=math.log10((math.pow(10, ats/10)-1)/(math.pow(10, atp/10)-1))/2*math.log10(oms)
    #n=np.arccosh(math.sqrt(((10 ** (0.1*fs)) - 1.0) / ((10 ** (0.1 * fp)) - 1.0))) /np.arccosh(oms)
    

    if n<0:
        n=n*-1
        highpass=True
    else:
        highpass=False
    
    
    
    

    ats1=10*math.log(math.pow(oms, 2*n)*(math.pow(10, atp/10) -1)+1)

    omc=oms/(math.pow((math.pow(10, ats1/10)-1), 1/(2*n)))

    

    aux=2*3.14*fp
    
    
    i=1
    while i<n:
        i=i+1
    
    n=i
    
    if minN>n:
        n=minN
        
    #n=int(math.ceil(n))    

    polos=[]

    vet=[]
    
    v=(1/n)*np.arcsinh(1/math.sqrt((10**(rp/10))-1))

    for i in range(0, n):
        if highpass:
            polos.append([[1/(-math.sin((3.14159265*((2*i)-1))/(2*(-n)))*math.sinh(v))], [1/(-math.cos((3.14159265*((2*i)-1))/(2*(-n)))*math.cosh(v))]])
        else:
           polos.append([[1/(-math.sin((3.14159265*((2*i)-1))/(2*n))*math.sinh(v))], [1/(-math.cos((3.14159265*((2*i)-1))/(2*n))*math.cosh(v))]])
           
         

    polos=np.array(polos)
    
   
    polos= polos.astype(float)

    polos2=[]

    for i in range(0, n):
        polos2.append([[[1], [1], [0]], [[0], [-aux*omc*float(polos[i][0])], [-aux*omc*float(polos[i][1])]]])
        vet.append(0)

    polos2=np.array(polos2)

       

    polos3=[]
    



    for i in range(0, int(math.pow(2, n))):
        polos3.append([[polos2[0][vet[0]][0]], [polos2[0][vet[0]][1]], [polos2[0][vet[0]][2]]])
    
        for j in range(1, n):
            polos3[i][0]=int(polos3[i][0]+polos2[j][vet[j]][0])
            a=float((polos3[i][1]*polos2[j][vet[j]][1])+(polos3[i][2]*polos2[j][vet[j]][2]))
            b=float((polos3[i][1]*polos2[j][vet[j]][2])+(polos3[i][2]*polos2[j][vet[j]][1]))
            polos3[i][1]=a
            polos3[i][2]=b
    
        
        j=n-1
        if vet[j]==0:
            vet[j]=1
        else:
            while vet[j]==1:
                vet[j]=0
                if j>0:
                    if vet[j-1]==0:
                        vet[j-1]=1
                    else:
                        j=j-1
                else:
                    vet[j]=0
                
                
        
     
            
   

    dividendos=[(math.pow((aux*omc), n))]
    
    

    divisores=[]



    for i in range(0, n+1):
        divisores.append([i, 0.0])
        for j in range(0, len(polos3)):
            if i==polos3[j][0]:
                divisores[i][1]+=float(polos3[j][1])
                
        if (divisores[i][1]<0.001) & (divisores[i][1]>0.0):
            divisores[i][1]=0
        elif (divisores[i][1]>-0.001) & (divisores[i][1]<0.0):
            divisores[i][1]=0
    
    
    

    auxZ1=[-1, 1]
    auxZ2=[1, 1]
    auxZ3=multiply(auxZ1, [2*fp], len(auxZ1), 1)

    protoZ1=[1]
    protoZ2=[1]

    list1=[[1]]
    list2=[[1]]

    for i in range(0, n):
        protoZ1=multiply(protoZ1, auxZ1, len(protoZ1), len(auxZ1))
        protoZ2= multiply(protoZ2, auxZ2, len(protoZ2), len(auxZ2))
        list1.append(protoZ1)
        list2.append(protoZ2)


    divList=[]
    
    
     
    

    for i in range(0, n):
        dividendos=multiply(dividendos, auxZ2, len(dividendos), len(auxZ2))
        #divList.append(multiply([divisores[i][1]], list1[i], 1, len(list1[i])))
    
   
    
    for i in range(0, len(divisores)):
        grauN=divisores[i][0]
        auxDiv=[divisores[i][1]]
        for j in range(0, n):
            if j<grauN:
                auxDiv=multiply(auxDiv, auxZ3, len(auxDiv), len(auxZ3))
            else:
                auxDiv=multiply(auxDiv, auxZ2, len(auxDiv), len(auxZ2))
        divList.append(auxDiv)
        
    
    
    
   
    
    divisores=[]+divList[0]

    for i in range(1, len(divList)):
        divisores = [elemA + elemB for elemA, elemB in zip(divisores, divList[i])]
    
    
    
    if highpass:
        Y1=dividendos[len(dividendos)-1]
    else:
        Y1=divisores[len(divisores)-1]
        
    if Y1<0:
        Y2=-Y1
    else:
        Y2=Y1
    
    
    for i in range(len(divisores)):
        divisores[i]=divisores[i]/Y1
    
    
    for i in range(len(dividendos)):
        dividendos[i]=dividendos[i]/Y2
        
    
    

    fatorX=[]
    fatorY=[1.0]
    
    
    if highpass:
        for i in range(len(dividendos)-1, -1, -1):
            fatorY.append(dividendos[i])
        for i in range(len(divisores)-1, -1, -1):
            fatorX.append(divisores[i])
    else:
        for i in range(len(dividendos)-1, -1, -1):
            fatorX.append(dividendos[i])
        for i in range(len(divisores)-2, -1, -1):
            fatorY.append(-divisores[i])

    vector=[1.0]*500


    vector=np.array(vector) 

    vector=PrototypeFilter(vector, fatorX, fatorY)

    if vector[100]<0:
        for i in range(len(fatorX)):
            fatorX[i]/=(-1)

    return fatorX, fatorY



    
    

 

signals, fields = wfdb.rdsamp('103', channels = [0], sampto=5000)

fp=15
fs=60
atp=1
ats=50

fatorXlow, fatorYlow=FilterProjectCheb2(fp, fs, atp, ats, 0.1, 9)
vectorLPF= PrototypeFilter(signals, fatorXlow, fatorYlow)

fp=5
fs=1
atp=1
ats=50



fatorXhigh, fatorYhigh=FilterProjectCheb2(fp, fs, atp, ats, 0.01, 6)




vectorHPF= PrototypeFilter(signals, fatorXhigh, fatorYhigh)

plt.plot(signals)
plt.plot(vectorLPF)
plt.plot(vectorHPF)



