import matplotlib.pyplot as plt

from p import nss, arr_a, arr_phi, arr_f

def plotter(x, y, xl):
    plt.scatter(x, y)
    plt.xlabel(xl)
    plt.show()

plotter(arr_a, nss, "Amplitude")
plotter(arr_f, nss, "Frequency")
plotter(arr_phi, nss, "Phase")