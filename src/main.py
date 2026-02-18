import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from algorithms.KNN import KNN
from algorithms.LinearRegression import LinearRegression
from algorithms.LogisticRegression import LogisticRegression
from algorithms.DecisionTree import DecisionTree
from algorithms.RandomForest import RandomForest
from algorithms.NaiveBayes import NaiveBayes
from algorithms.PCA import PCA

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


# bc = datasets.load_breast_cancer()
# X, y = bc.data, bc.target
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=1234
# )

# clf = LogisticRegression(lr=0.001)
# clf = DecisionTree(max_depth=100)
# clf = RandomForest()

# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)


# def accuracy(y_test, y_pred):
#     return np.sum(y_test == y_pred) / len(y_test)


# acc = accuracy(y_test, predictions)
# print(acc)


# def accuracy(y_true, y_pred):
#         accuracy = np.sum(y_true == y_pred) / len(y_true)
#         return accuracy

# X, y = datasets.make_classification(
#     n_samples=1000, n_features=10, n_classes=2, random_state=123
# )
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=123
# )

# nb = NaiveBayes()
# nb.fit(X_train, y_train)
# predictions = nb.predict(X_test)

# print("Naive Bayes classification accuracy", accuracy(y_test, predictions))

# data = datasets.load_digits()
data = datasets.load_iris()
X = data.data
y = data.target

# Project the data onto the 2 primary principal components
pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

plt.scatter(
    x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()