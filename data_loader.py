import csv 
import pandas as pd

def data_load(path):
    """
    Read csv file
    """
    data = pd.read_csv(path)
    return data

content = data_load('Dataset.csv')
