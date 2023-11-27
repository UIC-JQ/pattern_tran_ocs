import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import csv

mu, sigma = 0.1*100,0.9*20
x = np.linspace(1, 100, 100)
pdf_value =norm.pdf(x, mu, sigma)
print(np.round(pdf_value*1000))