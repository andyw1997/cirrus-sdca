import numpy as np
from sklearn.preprocessing import normalize
import csv
from sdca import SDCA
import matplotlib.pyplot as plt
import sklearn.linear_model as linear_model

data = []
test_data = []

# process csv, encoding each attribute with labels (self generated to preserve ordering of ranges)
with open('income_data_clean.csv', 'r') as data_file:
    reader = csv.reader(data_file)
    first = True
    for row in reader:
        if not first:
            data.append(row)
        else:
            first = False

with open('adult_income_test_clean.csv', 'r') as data_file:
    reader = csv.reader(data_file)
    first = True
    for row in reader:
        if not first:
            test_data.append(row)
        else:
            first = False

data = np.array(data)
test_data = np.array(test_data)

y = data[:,-1:].T[0].astype('float') - 1.0
X = data[:,:-1].astype('float')
X = np.hstack((X, np.array([[1 for i in range(len(X))]]).T)) # add intercept
X = normalize(X, axis=1)

y_test = test_data[:,-1:].T[0].astype('float') - 1.0
X_test = test_data[:,:-1].astype('float')
X_test = np.hstack((X_test, np.array([[1 for i in range(len(X_test))]]).T))
X_test = normalize(X_test, axis=1)

a_0 = np.array([0.0 for i in y]).astype('float')
lamb = 0.00001

epochs_per_data = 2
sdca = SDCA('smooth_hinge')
epochs = []
percent_low_incorrect = []
percent_high_incorrect = []
total_percent_incorrect = []

percent_low_incorrect_test = []
percent_high_incorrect_test = []
total_percent_incorrect_test = []

for i in range(1,21):
    w, a_0 = sdca.train(X, y, a_0, epochs_per_data, lamb = lamb)

    epochs.append(i * epochs_per_data)
    
    print('finished epoch ' + str(i * epochs_per_data))
    print(w)

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

    percent_low_incorrect.append(incorrect[0]/total_0)
    percent_high_incorrect.append(incorrect[1]/total_1)
    total_percent_incorrect.append((incorrect[0] + incorrect[1])/(total_0 + total_1))

    pred_y = sdca.predict(X_test)
    incorrect = [0,0]
    correct = [0,0]
    for i in range(len(y_test)):
        if pred_y[i] != y_test[i]:
            incorrect[int(y_test[i])] += 1
        else:
            correct[int(y_test[i])] += 1

    total_0 = incorrect[0] + correct[0]
    total_1 = incorrect[1] + correct[1]

    percent_low_incorrect_test.append(incorrect[0]/total_0)
    percent_high_incorrect_test.append(incorrect[1]/total_1)
    total_percent_incorrect_test.append((incorrect[0] + incorrect[1])/(total_0 + total_1))

print("total training percent incorrect = " + str(total_percent_incorrect[-1]))
print("total test percent incorrect = " + str(total_percent_incorrect_test[-1]))

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
plt.plot(epochs, total_percent_incorrect_test, color='c')
plt.xlabel('num epochs')
plt.ylabel('percent incorrect')
plt.title('total test percent incorrect across both classes')

plt.figure(2)
plt.plot(epochs, total_percent_incorrect, color='r')
plt.xlabel('num epochs')
plt.ylabel('percent incorrect')
plt.title('total training percent incorrect across both classes')

plt.figure(3)
plt.plot(epochs, percent_high_incorrect, color='g')
plt.xlabel('num epochs')
plt.ylabel('percent incorrect')
plt.title('training percent incorrect for high income')

plt.figure(4)
plt.plot(epochs, percent_low_incorrect, color='b')
plt.xlabel('num epochs')
plt.ylabel('percent incorrect')
plt.title('training percent incorrect for low income')

plt.show()