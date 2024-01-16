import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import joblib

#arr_a, arr_f, arr_phi, nss = joblib.load("var")

newsamplespace=joblib.load("p")
newss=np.array(newsamplespace)
#print(newss)
arr_a=newss[:,0]
arr_phi=newss[:,1]
arr_f=newss[:,2]
nss=newss[:,-1]

def plotter(x, y, z, a):
    figure, axis = plt.subplots(3, 1)
    axis[0].scatter(x, a)
    axis[0].set_xlabel("Amplitude")
    axis[1].scatter(y, a)
    axis[1].set_xlabel("Phase")
    axis[2].scatter(z, a)
    axis[2].set_xlabel("Frequency")
    plt.tight_layout()
    plt.show()

plotter(arr_a, arr_phi, arr_f,nss)