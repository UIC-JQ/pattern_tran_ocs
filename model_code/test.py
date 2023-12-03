import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import csv
def test():
    ep = 1
    # ratio of pattern
    ratio = 1
    slot_per_ep = 200
    dataset = []
    for _ in range(int(ep*ratio)):
        x = np.linspace(1, slot_per_ep, slot_per_ep)
        #mu, sigma = np.random.uniform()*200, np.random.uniform()*80
        mu, sigma = 50, 60
        pdf_value = norm.pdf(x, mu, sigma) 
        pdf_value = np.round(pdf_value*1000)

        dataset.append(pdf_value)
        #print(len(pdf_value))
    #     print(sum(pdf_value))
    plt.plot(x, pdf_value)
    plt.title('Normal Distribution PDF')
    plt.xlabel('Values')
    plt.ylabel('Probability Density')
    plt.show()
test()