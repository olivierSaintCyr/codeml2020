import pandas as pd
import numpy as np

def chargeData():
    #python3 main.py
    df = pd.read_csv("../dataset/train.csv")
    del df['id']
    return df



# knn : datasetx -> y
def distance(x, x0):
    dist = 0
    for i in range(3):
        dist += (x[i] - x0[i])**2
    return dist

def kernel(x, s):
    return np.exp(-(x**2)/(s**2))

def KNN_find_lables(dataset, s):
    x = dataset.values.tolist()
    # dist = [[distance(i, x[69]), distance(i, x[420])] for i in x]
    distances = [[kernel(distance(i, x[69]), s), kernel(distance(i, x[420]),s)] for i in x]
    # print(dist)
    labels = [np.argmax(i) for i in distances]
    return labels

x = chargeData()
y = KNN_find_lables(x, 810)
df = pd.DataFrame(y)
df.to_csv("../dataset/sub2.csv")

print(df)