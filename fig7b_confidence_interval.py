import numpy as np
import matplotlib.pyplot as plt

from utils import cast_int

def plot_confidence_interval_size(eps_list: list, confidence_level: float):

    beta = confidence_level
    eps_t_list = [cast_int(eps/2) for eps in eps_list]

    ci_size = [-2*np.log(1-beta)/eps for eps in eps_t_list]

    t_max = 120

    plt.figure()
    x_range = np.arange(len(ci_size))
    plt.plot(x_range, ci_size, 's-')
    plt.xlim([x_range[0],x_range[-1]])
    plt.xticks(x_range, eps_t_list)
    plt.grid(linestyle=':')
    plt.ylabel(rf'Normalized size of {cast_int(beta*100)}% CI $\tau/t_\max$')
    plt.xlabel('$\epsilon_t$')

    plt.draw()

    print(f'Values multiplied by {t_max} minutes:', [120*ci for ci in ci_size])

    plt.show()


if __name__ == "__main__":

    confidence_level = 0.95
    eps_list = [1, 2, 4, 6, 8, 16, 64, np.inf]
    
    plot_confidence_interval_size(eps_list, confidence_level)

