"""Generate the Blasius solution for the flat plate boundary layer.
"""
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    data = np.loadtxt('blasius.csv', delimiter=',', skiprows=1)
    eta = data[:, 0]
    df = data[:, 1]
    eta_times_df_minus_f = data[:, 2]
    fig = plt.figure()
    plt.plot(eta, df, label=r'$df/d\eta$')
    plt.plot(eta, eta_times_df_minus_f, label=r'$\eta\,df/d\eta-f$')
    plt.xlabel(r'$\eta$')
    plt.legend()
    fig.savefig('Blasius.svg')
