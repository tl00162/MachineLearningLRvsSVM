import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
from matplotlib.colors import ListedColormap


wine = load_wine()

wine.target[[10, 80, 140]]
print(list(wine.target_names))

X = wine.data[:, [0, 1]]
y = wine.target
print("X ", X)
print("y ", y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')

    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='yellow', edgecolor='black', alpha=1.0, linewidth=1, marker='o',
                    s=100, label='test set')
                 
C_values = [10, 100, 1000, 5000]
for C in C_values:
    lr = LogisticRegression(C=C, random_state=0)
    lr.fit(X_train_std, y_train)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined_std, y_combined, classifier=lr
                          , test_idx=range(len(y_train), len(y_train) + len(y_test)))
    plt.title(f'Logistic Regression (C={C})')
    plt.xlabel('Alcohol (Standardized)')
    plt.ylabel('Malic Acid (Standardized)')
    plt.legend(loc='upper left')
    plt.show()

gamma_values = [0.1, 10]
for gamma in gamma_values:
    for C in C_values:
        svm = SVC(kernel='rbf', random_state=0, gamma=gamma, C=C)
        svm.fit(X_train_std, y_train)
        X_combined_std = np.vstack((X_train_std, X_test_std))
        y_combined = np.hstack((y_train, y_test))
        plot_decision_regions(X_combined_std, y_combined, classifier=svm, 
                              test_idx=range(len(y_train), len(y_train) + len(y_test)))
        plt.title(f'SVM (RBF kernel, C={C}, gamma={gamma})')
        plt.xlabel('Alcohol (Standardized)')
        plt.ylabel('Malic Acid (Standardized)')
        plt.legend(loc='upper left')
        plt.show()

accuracy_lr = []
params_lr = []
for c in np.arange(-4., 4.):
    lr = LogisticRegression(C=10.**c, random_state=0)
    lr.fit(X_train_std, y_train)
    y_pred = lr.predict(X_test_std)
    accuracy_lr.append(accuracy_score(y_test, y_pred))
    params_lr.append(10**c)
plt.plot(params_lr, accuracy_lr, linestyle='--', label='Logistic Regression Accuracy')
plt.xscale('log')
plt.xlabel('C (log scale)')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()

for gamma in gamma_values:
    accuracy_svm = []
    params_svm = []
    for c in np.arange(-4., 4.):
        svm = SVC(kernel='rbf', random_state=0, gamma=gamma, C=10.**c)
        svm.fit(X_train_std, y_train)
        y_pred = svm.predict(X_test_std)
        accuracy_svm.append(accuracy_score(y_test, y_pred))
        params_svm.append(10**c)
    plt.plot(params_svm, accuracy_svm, linestyle=':', label=f'SVM Accuracy (gamma={gamma})')

plt.xscale('log')
plt.xlabel('C (log scale)')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()
