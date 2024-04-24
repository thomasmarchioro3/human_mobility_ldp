import numpy as np
import matplotlib.pyplot as plt

from utils import cast_int

def compute_likelihood(eps: float, n_categories: int):
    m = n_categories
    exp_eps = np.exp(eps)
    p = 1/(1 + (exp_eps-1)/m)  # randomization probability
    likelihood = 1 - p + p/m
    return likelihood

def  plot_likelihood(eps_list: list, n_categories: int):

    eps_c_list = [cast_int(eps/2) for eps in eps_list]  # divide budget by 2
    likelihood_list = [compute_likelihood(eps, n_categories) for eps in eps_c_list]

    plt.figure()
    x_range = np.arange(len(likelihood_list))
    plt.plot(x_range, likelihood_list, 's-')
    plt.xlim([x_range[0],x_range[-1]])
    plt.xticks(x_range, eps_c_list)
    plt.grid(linestyle=':')
    plt.ylabel('Likelihood')
    plt.xlabel('$\epsilon_t$')

    plt.draw()

    plt.show()

if __name__ == "__main__":

    eps_list = [1, 2, 4, 6, 8, 16, 64, np.inf]
    n_categories = 30
    
    plot_likelihood(eps_list, n_categories)