# MNIST를 활용한 MLP Image Classification
# Image Classification task를 위하여 MLP 모델 구현
# MNIST -> 28 x 28개의 feature를 입력 받고 10 개의 클래스 중 하나로 분류
# Cross Entropy loss 사용 -> PyTorch에서는 내부적으로 Soft log와 NLL을 수행하기 때문에 자연스럽게 Softmax 연산을 수행하면서 미분의 안정성을 가져갈 수 있음.

from sklearn.datasets import fetch_openml
import numpy as np # padas 형태인 MNIST 데이터를 NUMPY로 변경 -> 이후, Torch로 변경
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

datasets = fetch_openml('mnist_784', parser = 'auto')

# Pandas 데이터를 numpy로 바꾼 뒤, tensor로 변경함 이때, basic은 floast32임을 유의
X = torch.tensor(datasets.data.to_numpy(np.float32))
y = torch.tensor(datasets.target.to_numpy(np.int64))

# 시각화 부분
for i in range(5):
    image = X[i].reshape(28, 28) # Flatten된 데이터셋이기에 reshape을 통해서 28 x 28로 설정함
    plt.imshow(image, cmap = 'binary')
    plt.axis('off')
    plt.show()


class MLP(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_units) # input_shape = 784, output_shape = hidden_units
        self.fc2 = nn.Linear(hidden_units, 10) # input_shape = hideen_units, output_shape = 10

    def forward(self, x):
        x = F.relu(self.fc1(x)) # activation function(fully connected layer1)
        x = self.fc2(x) # Cross Entropy 적용으로 인해서 자동으로 내부적으로 softmax 함수가 적용됨

        return x

def train(model, optimizer, criterion):

    X_dev = X.to(device)
    y_dev = y.to(device)

    for epoch in range(100): # epoch 100번 설정
        y_pred = model(X_dev) # training data에 대한 predict

        loss = criterion(y_pred, y_dev) # model이 수행한 predict에 대해서 loss function 수행
        print(f'Epoch: {epoch}, Loss: {loss}')

        optimizer.zero_grad() # 이전에 수행한 역전파 기울기 초기화
        loss.backward() # 오차역전파 -> gradient 구하기
        optimizer.step() # 오차역전파로 구한 gradient를 통해서 parameters 업데이트

model = MLP(1000).to(device)# 뉴런 수 1000개
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) # 경사하강법 SGD 활용
criterion = nn.CrossEntropyLoss() # loss function

train(model, optimizer, criterion)

y_pred = model(X[:5].to(device))
print(y_pred.shape)
print(y_pred.argmax(1))