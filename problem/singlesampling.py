import numpy as np
import matplotlib.pyplot as plt
import joblib

pi=np.pi

#getval input:a,f,phi,starting point,endingpoint,no. of steps
def getval(a,phi,f,sp,ep,n):
    a=a
    f=f
    phi=phi
    t=np.linspace(sp,ep,n)
    return a,phi,f,t

def signal(a,phi,f,t):
    return a*np.sin(2*pi*f*t+phi)

def noise(m,std,n):
    return np.random.normal(m,std,n)

#al parameters have uniform distribution--known priors
'''
P(a)=U[0,10]
P(phi)=U[0,2*pi]
P(f)=U[1,10]
'''

def randomparam(n):
    #arr_a=np.random.uniform(1,10,n).reshape(-1,1)
    arr_a=3
    arr_phi=np.random.uniform(0,2*pi,n).reshape(-1,1)
    #arr_phi=0
    arr_f=np.random.uniform(1,10,n).reshape(-1,1)
    #arr_f=4
    return arr_phi,arr_f

def randomsignalarr(n,t):
    arr_phi,arr_f=randomparam(n)
    arr_sig=signal(3,arr_phi,arr_f,t)
    return np.concatenate((arr_phi,arr_f,arr_sig),axis=1),arr_phi,arr_f

def likelihood(datastrain,signal,std):
    l=len(datastrain)
    lh = -(l/2)*np.log(2*np.pi*std**2)-np.sum((datastrain-signal)**2/(2*std**2))
    return lh

def likelihoodarr(datastrain,signalarr,std):
    lha=[]
    for i in range(signalarr.shape[0]):
        lha.append(likelihood(datastrain,signalarr[i][2:],std))
    lha=np.array(lha).reshape(-1,1)
    return np.concatenate((signalarr,lha),axis=1)

def newsignalarr(d,t,std):
    #a=np.random.uniform(1,10)
    a=3
    phi=np.random.uniform(0,2*pi)
    #phi=pi
    f=np.random.uniform(1,10)
    #f=4
    newsig = signal(a,phi,f,t)
    nsarr=np.concatenate((np.array([phi,f]),newsig,[likelihood(d,newsig,std)]),axis=0)
    return nsarr


#sampling
def lowestlh(lha):
    return np.argmin(lha[:,-1])

def checkchange(new,old,checkarr):
    checkarr=np.delete(checkarr,0)
    if abs(new-old)<=0.1: 
        checkarr=np.append(checkarr,0)
    else:  
        checkarr=np.append(checkarr,1)  
    print(new,old)     
    return np.sum(checkarr)>0, checkarr

def nestedsampling(lha,data,std,t):
    newsamplespace=[]
    checkarr=np.ones(5)
    arg=lowestlh(lha)
    nsarr=newsignalarr(data,t,std)
    while(nsarr[-1]<=lha[arg][-1]):
        nsarr=newsignalarr(data,t,std)   

    dup=np.copy(lha[arg])
    newsamplespace.append(dup)
    lha[arg]=np.copy(nsarr)
    cond=True
    n=0
    
    while(cond and n<1000):
        arg=lowestlh(lha)
        nsarr=newsignalarr(data,t,std)
        
        while(nsarr[-1]<=lha[arg][-1]):
            nsarr=newsignalarr(data,t,std)
          
        cond,checkarr=checkchange(lha[arg][-1],newsamplespace[-1][-1],checkarr)   
        newsamplespace.append(np.copy(lha[arg])) 
       
        n+=1
        print(n,checkarr)
        lha[arg]=np.copy(nsarr)

    return newsamplespace     

std=4
mean=0
a,phi,f,t=getval(3,0,4,0,4,2000)
arr_sig,arr_a,arr_f=randomsignalarr(100,t)
print(arr_sig.shape)
injn=noise(mean,std,len(t))
injsig=signal(a,phi,f,t)
data=injn+injsig

plt.plot(t,injn, label='noise')
plt.plot(t, injsig, label='signal')
#plt.plot(t, data, label='data')
plt.grid()
plt.xlabel('time')
plt.ylabel('data/signal/noise')
plt.legend()
plt.show()

lha=likelihoodarr(data,arr_sig,std)
#print(sorted(lha[:,-1]))
newsamplespace=nestedsampling(lha,data,std,t)
#print(newsamplespace)

joblib.dump((newsamplespace), "p1") 