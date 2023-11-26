import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import csv
def test1():
    ep = 550
    # ratio of pattern
    ratio = 0.7
    slot_per_ep = 200
    dataset = []
    for _ in range(int(ep*ratio)):
        x = np.linspace(1, slot_per_ep, slot_per_ep)
        mu, sigma = np.random.uniform()*100, np.random.uniform()*20
        pdf_value = norm.pdf(x, mu, sigma) 
        pdf_value = np.round(pdf_value*1000)
        dataset.append(pdf_value)
        #print(len(pdf_value))
        #print(sum(pdf_value))
    # plt.plot(x, pdf_value)
    # plt.title('Normal Distribution PDF')
    # plt.xlabel('Values')
    # plt.ylabel('Probability Density')
    # plt.show()
    for _ in range(int(ep(1-ratio))):
        temp = []
        for _ in range(slot_per_ep):
            k = np.random.randint(0,13)
            temp.append(k)
        dataset.append(temp)
        #print(len(temp))


    print(len(dataset))

    # Specify the CSV file name for saving
    csv_filename = 'mix_rush_hour_50pec.csv'
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for value in tqdm(dataset):
            csv_writer.writerow(value)


test1()