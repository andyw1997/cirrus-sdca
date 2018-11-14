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

train = 'train.txt'

prev_time = time.time()
y = []
y_test = []

rows = []
cols = []
data = []
rows_test = []
cols_test = []
data_test = []

test_count = 0
train_count = 0

# read in data and hash in sparse format
for idx, row in enumerate(reader(open(train), delimiter='\t')):
    if idx % 100 == 1: # test point
        y_test.append(1. if row[0] == '1' else 0.)
        del row[0]
        
        # append bias term
        rows_test.append(test_count)
        cols_test.append(13+26)
        data_test.append(1)

        # read in as sparse format, integerizing
        for i in range(13 + 26):
            if row[i] == '':
                rows_test.append(test_count)
                cols_test.append(i)
                data_test.append(-1)
            elif i < 13:
                val = int(row[i])
                if val != 0:
                    rows_test.append(test_count)
                    cols_test.append(i)
                    data_test.append(val)
            else:
                # categorical, so convert from hex to int
                val = int(row[i], 16)
                rows_test.append(test_count)
                cols_test.append(i)
                data_test.append(val)

        test_count += 1
    elif idx % 10 == 0: # normal training point
        y.append(1. if row[0] == '1' else 0.)
        del row[0]

        # append bias term
        rows.append(train_count)
        cols.append(13+26)
        data.append(1)
        
        # read in as sparse format, integerizing
        for i in range(13 + 26):
            if row[i] == '':
                rows.append(train_count)
                cols.append(i)
                data.append(-1)
            elif i < 13:
                val = int(row[i])
                if val != 0:
                    rows.append(train_count)
                    cols.append(i)
                    data.append(val)
            else:
                # categorical, so convert from hex to int
                val = int(row[i], 16)
                rows.append(train_count)
                cols.append(i)
                data.append(val)

        train_count += 1
    
print('data finished reading')
curr_time = time.time() - prev_time
print('time for read data step = %s seconds' % curr_time)
prev_time = time.time()
print()

train_len = len(data)
test_len = len(data_test)

# find counts of values for each categorical feature in test+train data (+1 for bias feature)
counts = [{} for i in range(13+26+1)]
for i in range(train_len):
    col = cols[i]
    if col >= 13:
        val = data[i]
        if not val in counts[col]:
            counts[col][val] = 0
        counts[col][val] += 1

for i in range(test_len):
    col = cols_test[i]
    if col >= 13:
        val = data_test[i]
        if not val in counts[col]:
            counts[col][val] = 0
        counts[col][val] += 1

print('data finished counting')
curr_time = time.time() - prev_time
print('time for counting = %s seconds' % curr_time)
print()

buckets = [
    0.49, 0.99, 1.74, 2.865, 4.5525, 7.08375, 10.880625, 16.5759375,
    25.11890625, 37.933359375, 57.1550390625, 85.98755859375, 129.236337890625,
    194.1095068359375, 291.41926025390626, 437.3838903808594, 656.3308355712891,
    984.7512533569336, 1477.3818800354004, 2216.3278200531004, 3324.7467300796507,
    4987.375095119476, 7481.317642679214, 11222.231464018821, 16833.602196028234,
    25250.65829404235, 37876.24244106352, 56814.61866159528, 85222.18299239293,
    127833.5294885894, 191750.54923288408, 287626.0788493261, 431439.3732739892,
    647159.3149109838, 970739.2273664756, 1456109.0960497134, 2184163.8990745707,
    3276246.103611856, 4914369.410417783, 7371554.370626675]
increment = len(buckets) + 1
categorical_base = increment * 13

# remove cols from counts if count is < 15, translate values to IDs
curr_id = categorical_base

def increment_and_return():
    global curr_id
    curr_id += 1
    return curr_id - 1

counts = [{val : increment_and_return() for val, count in a.items() if count >= 15} for a in counts]

def find_bucket(num):
    i = 0;
    while i < len(buckets) and num >= buckets[i]:
        i += 1
    return i;

def preprocess(rows, cols, data):
    for i in range(len(data)):
        col = cols[i]
        val = data[i]

        # bucketize integer values
        if col < 13:
            bucket = find_bucket(val)
            cols[i] = bucket + increment*col
            data[i] = 1

        # ignore rare categorical features and convert to ID, including bias term
        else:
            # default 0, out of bound col index if destined to be removed due to low count
            entry = 0
            index = curr_id
            if val in counts[col]:
                index = counts[col][val]
                entry = 1

            cols[i] = index
            data[i] = entry

preprocess(rows, cols, data)
preprocess(rows_test, cols_test, data_test)

# make sure all vals are now either 1 or 0. 0's will be removed after turning into csr_matrix
for i in range(train_len):
    assert(data[i] == 1 or (data[i] == 0 and cols[i] == curr_id))

X = csr_matrix((data, (rows, cols)))
X.eliminate_zeros()
data, rows, cols = None, None, None

X_test = csr_matrix((data_test, (rows_test, cols_test)))
X_test.eliminate_zeros()
data_test, rows_test, cols_test = None, None, None

y = np.array(y)
y_test = np.array(y_test)

print('data finished preprocessing')
curr_time = time.time() - prev_time
print('time for preprocess data step = %s seconds' % curr_time)
prev_time = time.time()
print()

# Get sklearn's value
lr = LogisticRegression().fit(X,y)

print('sklearn log loss = ' + str(lr.score(X_test,y_test)))

# Start training step

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

# log loss function for loss of prediction values
def logloss(predictions, y):
    total = 0.0
    for i, yi in enumerate(y):
        p = max(min(predictions[i], 1. - 10e-12), 10e-12)
        total -= log(p) if yi == 1. else log(1. - p)
    return total / len(y)

# run SDCA training for some number of epochs
for i in range(1,21):
    w, a_0 = sdca.train(X, y, a_0, epochs_per_data, lamb = lamb)

    epochs.append(i * epochs_per_data)
    
    print('finished epoch ' + str(i * epochs_per_data))
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
    print()

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

# TODO: sklearn