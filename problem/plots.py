import matplotlib.pyplot as plt
import joblib

arr_a, arr_f, arr_phi, nss = joblib.load("var")

def plotter(x, y, z, a):
    figure, axis = plt.subplots(3, 1)
    axis[0].scatter(x, a)
    axis[0].set_xlabel("Amplitude")
    axis[1].scatter(y, a)
    axis[1].set_xlabel("Frequency")
    axis[2].scatter(z, a)
    axis[2].set_xlabel("Phase")
    plt.tight_layout()
    plt.show()

plotter(arr_a, arr_f, arr_phi,nss)