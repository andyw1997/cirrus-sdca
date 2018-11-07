import numpy as np
from sklearn.preprocessing import normalize
import csv
from sdca import SDCA
import matplotlib.pyplot as plt
import sklearn.linear_model as linear_model

data = []

def get_first_num(range_str):
    return int(range_str[:range_str.index('-')])

classification = {'no-recurrence-events':0, 'recurrence-events':1}
def age(range_str):
    return int(eval(range_str[0]) - eval('1'))
menopause = {'lt40':0, 'ge40':1, 'premeno':2}
def tumor_size(range_str):
    return get_first_num(range_str) / 5
def inv_nodes(range_str):
    return get_first_num(range_str) / 3
node_caps = {'yes':0, 'no':1}
deg_malig = {'1':0, '2':1, '3':2}
breast = {'left':0, 'right':2}
breast_quad = {'left_up':0, 'left_low':1, 'right_up':2,    'right_low':3, 'central':4}
irradiat = {'yes':0, 'no':1}

# process csv, encoding each attribute with labels (self generated to preserve ordering of ranges)
with open('breast-cancer.csv', 'r') as data_file:
    reader = csv.reader(data_file)
    for row in reader:
        data.append([
            classification[row[0]],
            age(row[1]),
            menopause[row[2]],
            tumor_size(row[3]),
            inv_nodes(row[4]),
            node_caps[row[5]],
            deg_malig[row[6]],
            breast[row[7]],
            breast_quad[row[8]],
            irradiat[row[9]]])

data = np.array(data)

y = data[:,:1].T[0].astype('float')
X = data[:,1:].astype('float')
X = np.hstack((X, np.array([[1 for i in range(len(X))]]).T)) # add intercept
a_0 = np.array([0.0 for i in y]).astype('float')
lamb = 0.0001
X = normalize(X, axis=1)

epochs_per_data = 5
sdca = SDCA('smooth_hinge')
epochs = []
percent_no_recurrence_incorrect = []
percent_recurrence_incorrect = []
total_percent_incorrect = []

for i in range(1,51):
    w, a_0 = sdca.train(X, y, a_0, epochs_per_data, lamb = lamb)

    epochs.append(i * epochs_per_data)
    
    print('finished epoch ' + str(i * epochs_per_data))

    pred_y = sdca.predict(X)
    incorrect = [0,0]
    correct = [0,0]
    for i in range(len(y)):
        if pred_y[i] != y[i]:
            incorrect[int(y[i])] += 1
        else:
            correct[int(y[i])] += 1

    total_0 = incorrect[0] + correct[0]
    total_1 = incorrect[1] + correct[1]

    percent_no_recurrence_incorrect.append(incorrect[0]/total_0)
    percent_recurrence_incorrect.append(incorrect[1]/total_1)
    total_percent_incorrect.append((incorrect[0] + incorrect[1])/(total_0 + total_1))

print("total percent incorrect = " + str(total_percent_incorrect[-1]))

# Compare to sklearn implementation
sklearn_model = linear_model.LogisticRegression()
sklearn_model.fit(X, y)
pred_y = sklearn_model.predict(X)
incorrect = [0,0]
correct = [0,0]
for i in range(len(y)):
    if pred_y[i] != y[i]:
        incorrect[int(y[i])] += 1
    else:
        correct[int(y[i])] += 1

total_0 = incorrect[0] + correct[0]
total_1 = incorrect[1] + correct[1]
percent_incorrect_sklearn = (incorrect[0] + incorrect[1])/(total_0 + total_1)
print("total percent incorrect sklearn = " + str(percent_incorrect_sklearn))
print("parameters = " + str(sklearn_model.get_params()))

plt.figure(1)
plt.plot(epochs, total_percent_incorrect, color='r')
plt.xlabel('num epochs')
plt.ylabel('percent incorrect')
plt.title('total percent incorrect across both classes')

plt.figure(2)
plt.plot(epochs, percent_recurrence_incorrect, color='g')
plt.xlabel('num epochs')
plt.ylabel('percent incorrect')
plt.title('percent incorrect for recurrence-events')

plt.figure(3)
plt.plot(epochs, percent_no_recurrence_incorrect, color='b')
plt.xlabel('num epochs')
plt.ylabel('percent incorrect')
plt.title('percent incorrect for no-recurrence-events')

plt.show()