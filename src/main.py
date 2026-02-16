import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNN
from LinearRegression import LinearRegression
from LogisticRegression import LogisticRegression
from DecisionTree import DecisionTree
from RandomForest import RandomForest

# cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

# iris = datasets.load_iris()
# X, y = iris.data, iris.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# plt.figure()
# plt.scatter(X[:,2],X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
# plt.show()


# clf = KNN(k=5)
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)

# print(predictions)

# acc = np.sum(predictions == y_test) / len(y_test)
# print(acc)

# X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# fig = plt.figure(figsize=(8,6))
# plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
# plt.show()

# reg = LinearRegression(lr=1e-2)
# reg.fit(X_train,y_train)
# predictions = reg.predict(X_test)

# def mse(y_test, predictions):
#     return np.mean((y_test-predictions)**2)

# mse = mse(y_test, predictions)
# print(mse)

# y_pred_line = reg.predict(X)
# cmap = plt.get_cmap('viridis')
# fig = plt.figure(figsize=(8,6))
# m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
# m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
# plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
# plt.show()


bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# clf = LogisticRegression(lr=0.001)
# clf = DecisionTree(max_depth=100)
clf = RandomForest()

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)


def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)


acc = accuracy(y_test, predictions)
print(acc)
