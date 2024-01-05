import corner
import matplotlib.pyplot as plt
import numpy as np
import joblib

arr_a, arr_f, arr_phi, nss = joblib.load("var")

def corplot(l,a,b,c):

    # Select two parameters for the corner plot
    param_a = "A"
    param_b = "Phi"
    param_c = "Frequency"
    # Create a corner plot
    data1 = np.vstack([l,a,b]).T
    data2 = np.vstack([l,b,c]).T
    data3 = np.vstack([l,c,a]).T
    figure = corner.corner(data1, labels=[l,param_a, param_b], show_titles=True)
    figure = corner.corner(data2, labels=[l,param_b, param_c], show_titles=True)
    figure = corner.corner(data3, labels=[l,param_c, param_a], show_titles=True)
    # Show the plot
    plt.show()

corplot(nss,arr_a,arr_phi,arr_f)