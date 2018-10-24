import mmh3
from scipy.sparse import csr_matrix
from sdca_sparse import SDCA
from math import log
from csv import DictReader
import numpy as np
import matplotlib.pyplot as plt
import time

D = 2**20
HASH_SEED = 21
train = 'train.txt'
column_names = ['Label']
for i in range(13):
    column_names.append('I' + str(i))
for i in range(26):
    column_names.append('C' + str(i))

# performs hash trick on data and changes to a csr_matrix
def hash_data(data):
    rows = []
    cols = []
    vals = []
    for idx, row in enumerate(data):
        rows.append(idx)
        cols.append(0)
        vals.append(0)
        for col, val in row.items():
            hash_val = mmh3.hash(col[1:] + str(val), HASH_SEED, signed=False)
            bucket = hash_val % D
            rows.append(idx)
            cols.append(bucket)
            vals.append(1)
    return csr_matrix((vals, (rows, cols)), shape = (len(data), D))

# log loss function for loss of prediction values
def logloss(predictions, y):
    total = 0.0
    for i, yi in enumerate(y):
        p = max(min(predictions[i], 1. - 10e-12), 10e-12)
        total -= log(p) if yi == 1. else log(1. - p)
    return total / len(y)


prev_time = time.time()
y = []
X = []
i = 1
# read in data, using every 250th datapoint to use a smaller dataset
for row in DictReader(open(train), fieldnames = column_names, delimiter='\t'):
    if i % 250 == 0:
        y.append(1. if row['Label'] == '1' else 0.)
        del row['Label']
        X.append(row)
    i += 1

print('training data finished reading')
curr_time = time.time() - prev_time
print('time for read data step = %s seconds' % curr_time)
prev_time = time.time()

# hash trick on data
y = np.array(y)
X = hash_data(X)

print('data finished hashing')
curr_time = time.time() - prev_time
print('time for hash step = %s seconds' % curr_time)

a_0 = np.array([0.0 for i in y])
lamb = 0.00001

epochs_per_data = 1
sdca = SDCA('log')
epochs = []
loss = []

train_start = time.time()
prev_time = train_start
times = []

# run SDCA training for some number of epochs
for i in range(1,11):
    w, a_0 = sdca.train(X, y, a_0, epochs_per_data, lamb = lamb)

    epochs.append(i * epochs_per_data)
    
    print('finished epoch ' + str(i * epochs_per_data))
    curr_time = time.time() - prev_time
    print('time for epoch = %s seconds' % curr_time)
    prev_time = time.time()
    since_start = prev_time - train_start
    print('time since start of training = %s seconds' % since_start)
    times.append(since_start)

    pvals = sdca.getpvals(X)

    loss.append(logloss(pvals, y))
    print('log loss = ' + str(loss[-1]))

print("final log loss = " + str(loss[-1]))

plt.figure(1)
plt.plot(epochs, loss)
plt.xlabel('num epochs')
plt.ylabel('log loss')
plt.title('log loss on criteo vs epochs')

plt.figure(2)
plt.plot(times, loss)
plt.xlabel('time since training start')
plt.ylabel('log loss')
plt.title('log loss on criteo vs time')

plt.show()
