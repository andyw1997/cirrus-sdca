import mmh3
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sdca_sparse import SDCA
from math import log
from csv import reader
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time

D = 2**20
HASH_SEED = 21
train = 'train.txt'

columns = ['Label']
for i in range (1,14):
    columns.append('I'+str(i))
for i in range (1,27):
    columns.append('C'+str(i))

# log loss function for loss of prediction values
def logloss(predictions, y):
    total = 0.0
    for i, yi in enumerate(y):
        p = max(min(predictions[i], 1. - 10e-12), 10e-12)
        total -= log(p) if yi == 1. else log(1. - p)
    return total / len(y)


prev_time = time.time()
y = []
y_test = []

rows = []
cols = []
vals = []
rows_test = []
cols_test = []
vals_test = []

test_count = 0
train_count = 0

# read in data and hash
for idx, row in enumerate(reader(open(train), delimiter='\t')):
    if idx % 100 == 1: # test point
        y_test.append(1. if row[0] == '1' else 0.)
        del row[0]
        rows_test.append(test_count)
        cols_test.append(0)
        vals_test.append(0)
        for col, val in enumerate(row):
            hash_val = mmh3.hash(str(col) + str(val), HASH_SEED, signed=False)
            bucket = hash_val % D
            rows_test.append(test_count)
            cols_test.append(bucket)
            vals_test.append(1)
        test_count += 1
    elif idx % 10 == 0: # normal training point
        y.append(1. if row[0] == '1' else 0.)
        del row[0]
        rows.append(train_count)
        cols.append(0)
        vals.append(0)
        for col, val in enumerate(row):
            hash_val = mmh3.hash(str(col) + str(val), HASH_SEED, signed=False)
            bucket = hash_val % D
            rows.append(train_count)
            cols.append(bucket)
            vals.append(1)
        train_count += 1
    
print('training data finished reading')
curr_time = time.time() - prev_time
print('time for read data step = %s seconds' % curr_time)
prev_time = time.time()

# convert to csr_matrix and garbage collect previous format of data
y_test = np.array(y_test)
X_test = csr_matrix((vals_test, (rows_test, cols_test)), shape = (test_count, D))
vals_test = None
rows_test = None
cols_test = None

y = np.array(y)
X = csr_matrix((vals, (rows, cols)), shape = (train_count, D))
vals = None
rows = None
cols = None

print('data finished hashing')
curr_time = time.time() - prev_time
print('time for hash step = %s seconds' % curr_time)

#### TEMP

lr = LogisticRegression(solver = 'liblinear').fit(X,y)

print('sklearn log loss = ' + str(logloss(lr.predict_proba(X_test)[:,1].T,y_test)))

####

a_0 = np.array([0.0 for i in y])
lamb = 0.00001

epochs_per_data = 1
sdca = SDCA('log')
epochs = []
loss = []
loss_test = []

train_start = time.time()
prev_time = train_start
times = []

# run SDCA training for some number of epochs
for i in range(1,21):
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

    pvals_test = sdca.getpvals(X_test)
    loss_test.append(logloss(pvals_test, y_test))

    print('training log loss = ' + str(loss[-1]))
    print('test log loss = ' + str(loss_test[-1]))

fig = plt.figure(1)
plt.plot(epochs, loss)
plt.xlabel('num epochs')
plt.ylabel('log loss')
plt.title('training log loss on criteo vs epochs')
plt.savefig('training_loss_epochs.png',dpi=300)
plt.close(fig)

fig = plt.figure(2)
plt.plot(times, loss)
plt.xlabel('time since training start')
plt.ylabel('log loss')
plt.title('training log loss on criteo vs time')
plt.savefig('training_loss_time.png',dpi=300)
plt.close(fig)

fig = plt.figure(3)
plt.plot(epochs, loss_test)
plt.xlabel('num epochs')
plt.ylabel('log loss')
plt.title('test log loss on criteo vs epochs')
plt.savefig('test_loss_epochs.png',dpi=300)
plt.close(fig)

fig = plt.figure(4)
plt.plot(times, loss_test)
plt.xlabel('time since training start')
plt.ylabel('log loss')
plt.title('test log loss on criteo vs time')
plt.savefig('test_loss_time.png',dpi=300)
plt.close(fig)

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

