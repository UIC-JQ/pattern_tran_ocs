import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import csv

def test():
    dataset = []
    ep = 1
    slot_per_ep = 200
    for _ in range(ep):
        x = np.linspace(1, slot_per_ep, slot_per_ep)
        mu, sigma = np.random.uniform()*20, np.random.uniform()*70
        pdf_value = norm.pdf(x, mu, sigma) 
        pdf_value = np.round(pdf_value*1000)+np.random.choice([1,2,3,4,5])

        dataset.append(pdf_value)
        #print(len(pdf_value))
        print(sum(pdf_value))
        plt.plot(x, pdf_value)
        plt.title('Normal Distribution PDF')
        plt.xlabel('time slots')
        plt.ylabel('task numbers')
        plt.show()

y = 0.25
percentage = f'{y:.0}'
print(percentage)