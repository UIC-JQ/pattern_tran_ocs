import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import csv
from tqdm import tqdm
import ast

def gen_data():
    data_sets = []
    ep = 1
    for _ in range(ep):
        # 200 time slots in one episode
        time_slot_per_ep = 200 
        mu_choices = np.random.uniform(-3, 3, 10)
        std_choices = np.random.uniform(0.1, 2, 10)

        # Randomly select a distribution
        choice = np.random.randint(10)  
        choice2 = np.random.randint(10)

        mu1, std1 = mu_choices[choice], std_choices[choice]
        mu2, std2 = mu_choices[choice2], std_choices[choice2]

        data1 = np.random.normal(mu1, std1, 600)
        #data2 = np.random.normal(mu2, std2, 600)
        #data = np.concatenate((data1, data2))

        hist, _, _ = plt.hist(data1, time_slot_per_ep, density=False, alpha=0.6, color='b')
        # print(sum(hist))
        data_sets.append(hist)

    plt.show()
    # Specify the CSV file name for saving
    csv_filename = 'rush_hour_1200.csv'
    # Write the array line by line to the CSV file.
    # with open(csv_filename, mode='w', newline='') as csv_file:
    #     csv_writer = csv.writer(csv_file)
    #     for value in tqdm(data_sets):
    #         csv_writer.writerow(value)



    return data_sets

def hour_reader():
    # load data from the csv file
    csv_filename = 'rush_hour.csv'  
    data = []

    with open(csv_filename, mode='r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            data.append(row)
    print(len(data[0]))

def gen2():
    ep = 800
    dataset = []
    for _ in range(ep):
        slot_per_ep = 200
        x = np.linspace(1, slot_per_ep, slot_per_ep)
        mu, sigma = 100,20  
        pdf_value = norm.pdf(x, mu, sigma) 
        pdf_value = np.round(pdf_value*1000)
        dataset.append(pdf_value)
        #print(sum(pdf_value))
    # plt.plot(x, pdf_value)
    # plt.title('Normal Distribution PDF')
    # plt.xlabel('Values')
    # plt.ylabel('Probability Density')
    # plt.show()

    # Specify the CSV file name for saving
    csv_filename = 'new_rush_hour_1000.csv'
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for value in tqdm(dataset):
            csv_writer.writerow(value)

def gen3():
    ep = 800
    dataset = []
    for _ in range(ep):
        slot_per_ep = 200
        x = np.linspace(1, slot_per_ep, slot_per_ep)
        mu, sigma = 100,20  
        pdf_value = norm.pdf(x, mu, sigma) 
        pdf_value = np.round(pdf_value*1000)
        dataset.append(pdf_value)
        #print(sum(pdf_value))
    # plt.plot(x, pdf_value)
    # plt.title('Normal Distribution PDF')
    # plt.xlabel('Values')
    # plt.ylabel('Probability Density')
    # plt.show()

    # Specify the CSV file name for saving
    csv_filename = 'new_rush_hour_1000.csv'
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for value in tqdm(dataset):
            csv_writer.writerow(value)

def gen4():
    ep = 1100
    # ratio of pattern
    ratio = 1
    slot_per_ep = 200
    dataset = []
    for _ in range(int(ep*ratio)):
        x = np.linspace(1, slot_per_ep, slot_per_ep)
        mu, sigma = np.random.uniform()*200, np.random.uniform()*80
        pdf_value = norm.pdf(x, mu, sigma) 
        pdf_value = np.round(pdf_value*1000)

        dataset.append(pdf_value)
        #print(len(pdf_value))
    #     print(sum(pdf_value))
    # plt.plot(x, pdf_value)
    # plt.title('Normal Distribution PDF')
    # plt.xlabel('Values')
    # plt.ylabel('Probability Density')
    # plt.show()
    for _ in range(int(ep*(1-ratio))):
        temp = []
        for _ in range(slot_per_ep):
            k = np.random.randint(0,13)
            temp.append(k)
        #print(sum(temp))
        dataset.append(temp)
        #print(len(temp))
    np.random.shuffle(dataset)


    # Specify the CSV file name for saving
    csv_filename = 'mix_rush_hour_{}pec.csv'.format(ratio)
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for value in tqdm(dataset):
            csv_writer.writerow(value)


gen4()