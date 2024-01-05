import numpy as np
import matplotlib.pyplot as plt
import joblib

pi=np.pi

#getval input:a,f,phi,starting point,endingpoint,stepsize
def getval(a,f,phi,sp,ep,ss):
    a=a
    f=f
    phi=phi
    t=np.arange(sp,ep,ss)
    return a,f,phi,t

def w(f):
    return 2*pi*f

def sig(w,phi,a,t):
    return a*np.sin((w*t+phi)%2*pi)

def noise(m,std,n):
    return np.random.normal(m,std,n)


#alh parameters have uniform distribution--known priors
'''
P(a)=U[0,15]
P(phi)=U[0,2*pi]
P(f)=U[5,75]
'''

def randomparams(n):
    arr_a,arr_f,arr_phi=[],[],[]
    for i in range(n):
        arr_a.append(np.random.randint(0,15))
        arr_phi.append(np.random.uniform(0,2*pi))
        arr_f.append(np.random.randint(5,75))
    return arr_a,arr_phi,arr_f

def randomsignals(arr_f,arr_phi,arr_a,t):
    arr_s=[]
    for i in range(len(arr_f)):
        w1=w(arr_f[i])
        arr_s.append(sig(w1,arr_phi[i],arr_a[i],t))
    np.array(arr_s)    
    return arr_s

def lh(d,s,std,n):
    lha=[]
    nn=np.array((d-s)**2)
    n1=len(nn[0])
    for i in range(n):
        r=np.sum(nn[i])
        lh=(-n1/2)*np.log(2*pi*(std**2))-(r/(2*(std**2)))
        lha.append(lh)
    return np.array(lha)    

def newsamplelikelihood(d,t,std):
       a=np.random.randint(0,15)
       phi=np.random.uniform(0,2*pi)
       f=np.random.randint(0,50)
       wc=w(f)
       newsig=sig(wc,phi,a,t)
       n1=len(t)
       nlh=(-n1/2)*np.log(2*pi*(std**2))-(np.sum((d-newsig)**2)/(2*(std**2)))
       return a,phi,f,nlh


#Sampling

def lowestlh(arr,arr_a,arr_phi,arr_f):
    minlh=min(arr)
    minindex=0
    for i in range(len(arr)):
        if arr[i]==minlh:
            minindex=i
            break
    return arr_a[minindex],arr_phi[minindex],arr_f[minindex],minlh,minindex        

def checkchange(new,old,checkarr):
    print(new,old)
    checkarr=np.delete(checkarr,0)
    if abs(new-old)<=0.1: 
        print("0")
        checkarr=np.append(checkarr,0)
    else:  
        print("1")
        checkarr=np.append(checkarr,1)
    return [np.sum(checkarr)>0, checkarr]
            

def nestedsampling(arr,d,t,std,arr_a,arr_phi,arr_f):
    dlh,da,dphi,df=[],[],[],[]
    checkarr=np.ones(5)
    a,phi,f,llh,minindex=lowestlh(arr,arr_a,arr_phi,arr_f)
    anew,phinew,fnew,nlh=newsamplelikelihood(d,t,std)
    n=0
    cond=True

    while(nlh<=llh):
            anew,phinew,fnew,nlh=newsamplelikelihood(d,t,std)
    da.append(a)
    dphi.append(phi)
    df.append(f)
    dlh.append(llh) 
    checkarr=checkchange(nlh,llh,checkarr)[1]

    while(n<300 and cond):
        while(nlh<=llh):
            anew,phinew,fnew,nlh=newsamplelikelihood(d,t,std)
        change=checkchange(dlh[-1],llh,checkarr)
        if (change[0]):
            da.append(a)
            dphi.append(phi)
            df.append(f)
            dlh.append(llh)
            arr=np.append(arr,nlh)
            arr_a=np.delete(arr_a,minindex)
            arr_phi=np.delete(arr_phi,minindex)
            arr_f=np.delete(arr_f,minindex)
            arr_a=np.append(arr_a,anew)
            arr_phi=np.append(arr_phi,phinew)
            arr_f=np.append(arr_f,fnew)
            arr=arr[arr!=llh]

        print(n,change[1])
        checkarr=change[1]
        cond=change[0]
        a,phi,f,llh,minindex=lowestlh(arr,arr_a,arr_phi,arr_f)
        anew,phinew,fnew,nlh=newsamplelikelihood(d,t,std)
        n+=1
        
    
    dlh=np.array(dlh)
    da=np.array(da)
    dphi=np.array(dphi)
    df=np.array(df)
    return dlh,da,dphi,df   


a,f,phi,t=getval(8,25,pi/3,0,2,0.1)

wc=w(f)

injsig=sig(wc,phi,a,t)

injn=noise(0,2,len(t))

d=injsig+injn

arr_a,arr_phi,arr_f=randomparams(50)

arr_s=randomsignals(arr_f,arr_phi,arr_a,t)
       
nn=lh(d,arr_s,2,len(arr_a))
print(nn)

nss,arr_a,arr_phi,arr_f=nestedsampling(nn,d,t,2,arr_a,arr_phi,arr_f)
print("\n",nss)

joblib.dump((arr_a, arr_f, arr_phi, nss), "var")   