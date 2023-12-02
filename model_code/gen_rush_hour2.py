import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import csv

def test():
    per_slot_time = [[1,2],[3,4]]
    csv_filename = 'test.csv'
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for value in tqdm(per_slot_time):
            csv_writer.writerow(value)

test()