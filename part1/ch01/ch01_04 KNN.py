# k - Nearest Neighbor(kNN)

# classification task에서 주로 사용되는 알고리즘
#     1.학습 데이터로부터 파라미터를 학습하지 않음(non - parametric)
#     2. 새로운 데이터가 들어오면 가장 가까운 k개의 학습 데이터와 비교
#     3. k개의 데이터 중 가장 많은 class로 분류되는 점
#         3.1. k개의 설정에 따라서 분류가 달라지게 된다는 것을 의미함
#         3.2 단순히 다수결로 정하기 보다는, 입력 데이터와 인접 데이터 간의 거리를 이용하여 가중치를 주는 방법도 자주 사용
#
#     4. Regression을 위해서 사용될 때, 인접 학습 데이터의 평균값을 사용
#     5. Hyperpapamet k는 상황에 맞춰서 설정해야 함.


import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

# 이항 분류를 위해서 세토사 클래스(class = 0)에 해당하는 데이터를 제외하고 수행
X = X[y != 0]
y = y[y != 0]

# Decision boundary 그리기 -> 결정 경게를 그리기
def make_meshgrid(x, y, h = 0.02): # 입력된 데이터들의 모든 조합으로 만들어내는 경우의 수
    x_min, x_max = x.min() -1, x.mean() + 1
    y_min, y_max = y.min() -1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(clf, xx, yy, **params): # predict 함수를 통해 만들어낸 모든 feature map을 수행
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = plt.contourf(xx, yy, Z, **params)

    return out


X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1) # 특징 조합 생성

for k in [3, 5, 7, 15]:
    model = KNeighborsClassifier(k)
    model.fit(X, y)

    # Decision boundary 시각화
    plot_contours(model, xx, yy, cmap = plt.cm.coolwarm, alpha = 0.8)

    # Training data plot
    plt.scatter(X0, X1, c = y, cmap = plt.cm.coolwarm, s = 20, edgecolors = "k")

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy. max())
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.title(f"kNN (k = {k}")
    plt.xticks(())
    plt.yticks(())
    plt.show()
