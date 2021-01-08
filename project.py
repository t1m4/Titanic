import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rc("font", size=14)
dataset = pd.read_csv('titanic_data.csv', sep=',')
dataset.head()
# print(dataset)
# print(dataset.Fare)
len_dataset = len(dataset)
survived = np.sum(dataset["Survived"].to_numpy())
# print(f"All - {len_dataset}\nSurvived - {survived}\nDied - {len_dataset - survived}")


targets = dataset['Survived'].to_numpy()
data = dataset.iloc[:, 1:].to_numpy()

def normalize_dataset(dataset):
    return (dataset - np.mean(dataset, axis=0)) / np.std(dataset, axis=0)
data = normalize_dataset(data)



def split_dataset(targets, dataset, split=0.8):
    index = np.arange(dataset.shape[0])
    np.random.shuffle(index)
    index_train = index[:int(dataset.shape[0] * split)]
    index_test = index[int(dataset.shape[0] * split):]

    datas_train = dataset[index_train]
    targets_train = targets[index_train]

    datas_test = dataset[index_test]
    targets_test = targets[index_test]

    return datas_train, targets_train, datas_test, targets_test
datas_train, targets_train, datas_test, targets_test = split_dataset(targets, data)

# print(f"Train - {len(datas_train)}")
# print(f"Test - {len(datas_test)}")


def sigmoid(input, weight):
    return 1/(1 + np.exp(-input @ weight))

def get_prediction(input, weight):
    return 1/(1 + np.exp(-input @ weight)) >= 0.5

def get_loss(data, targets, weight):
    sigmoid_data = sigmoid(data, weight)
    return - np.sum(np.log(np.power(sigmoid_data, targets) * np.power(1 - sigmoid_data, 1 - targets)))


def get_accuracy(datas, targets, weight):
    predictions = get_prediction(datas, weight)
    return np.sum(predictions == targets) / len(targets)

def get_precision(datas, targets, weight):
    predictions = get_prediction(datas, weight)
    tp = np.sum((targets == 1) & (predictions == 1))
    tn = np.sum((targets == 0) & (predictions == 0))
    fn = np.sum(targets == 1) - tp
    fp = np.sum(targets == 0) - tn

    return tp / (tp + fn)

def get_recall(datas, targets, weight):
    predictions = get_prediction(datas, weight)
    tp = np.sum((targets == 1) & (predictions == 1))
    tn = np.sum((targets == 0) & (predictions == 0))
    fn = np.sum(targets == 1) - tp
    fp = np.sum(targets == 0) - tn

    return tp / (tp + fp)


def start(datas_train, targets_train, weight, learning_rate, loops, sigma=0.001):
    best_weight = np.copy(weight)
    best_loss = np.inf
    loss_list = []

    for i in range(loops):
        prediction = get_prediction(datas_train, weight)
        grad = (prediction - targets_train).T @ datas_train
        weight -= learning_rate * grad
        current_loss = get_loss(datas_train, targets_train, weight)
        if current_loss < best_loss:
            best_weight = np.copy(weight)
            best_loss = current_loss
        loss_list.append(current_loss)
        # print(f"i - {i + 1}, loss - {np.int32(current_loss)}")
    # print(f"Best loss - {np.int32(best_loss)}")
    return {'best_weight': best_weight, 'loss': np.int32(loss_list)}

learning_rate = 0.01
loops = 20
sigma = 0.001

weight = sigma * np.random.randn(datas_train.shape[1])

start_time = time.time()
res = start(datas_train, targets_train, weight, learning_rate, loops, sigma=0.001)

# fig = plt.figure()
# plt.plot(np.arange(1, loops + 1), res['loss'], color='black')
# plt.xlabel('number', fontsize=15)
# plt.ylabel('loss', fontsize=15)
# plt.show()

# print(f"Test loss {np.int32(get_loss(datas_test, targets_test, res['best_weight']))}")
# print(f"Best weight {res['best_weight']}")
print(f"Time - {time.time() - start_time}")
#
# print(f"Train:\nAccuracy {np.int32(get_accuracy(datas_train, targets_train, res['best_weight']) * 100)}")
# print(f"Precisioon {np.int32(get_precision(datas_train, targets_train, res['best_weight']) * 100)}")
# print(f"Recall {np.int32(get_recall(datas_train, targets_train, res['best_weight']) * 100)}\n")
#
# print(f"Test:\nAccuracy {np.int32(get_accuracy(datas_test, targets_test, res['best_weight']) * 100)}")
# print(f"Precisioon {np.int32(get_precision(datas_test, targets_test, res['best_weight']) * 100)}")
# print(f"Recall {np.int32(get_recall(datas_test, targets_test, res['best_weight']) * 100)}\n")


my_vectors = np.array([
    1, # class
    0, # sex
    20,# age
    1, # subling
    2, # parents
    48 # fare
])
my_vector = normalize_dataset(my_vectors)
print(f"Answer: {get_prediction(my_vector, res['best_weight'])}")
