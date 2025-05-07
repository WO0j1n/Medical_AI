# PyTorch을 이용한 CNN 학습

# Image Classification task를 위한 CNN
# MNIST 데이터셋을 이용 28 x 28개의 feature을 입력 받고 10개의 클래스 중 하나로 분류
# 2개의 Conv Layer와 Max Pooling Layer 그리고 2개의 FC layer를 가짐
# Cross-entropy Loss 사용

# pandas -> numpy -> tensor로 데이터 변경

# Data preprocessing -> 1차원 데이터를 N x C X H x W(데이터 수, 채널 수, 높이, 가로) -> convolution layer의 입력에 맞추기 위해서 수행

# 2차원 이미지를 사용 -> conv2d 사용

# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
# in_channels, out_channels -> (N, C, H, W) == (데이터 수, 채널 수, 높이, 가로)의 shape로 입력 -> 그렇기 때문에 Data preprocessing을 수행

# torch.nn.MaxPool2d(kernel_size, stride)
    # stride를 설정하지 않으면 kernel_size와 같은 크기로 수행이 됨


from sklearn.datasets import fetch_openml
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

datasets = fetch_openml('mnist_784', parser = 'auto')

# pandas -> numpy -> tensor , tensor의 경우 기본적으로 float를 지원
X = torch.tensor(datasets.data.to_numpy(np.float32))
y = torch.tensor(datasets.target.to_numpy(np.int64))

X = X.reshape(-1, 1, 28, 28) # -1의 의미: 데이터 전체, (N, C, H, W)로 reshape 수행
print(X.shape)

# 이미지 시각화 5개
for i in range(5):
    image = X[i, 0]
    plt.imshow(image, cmap = 'binary')
    plt.axis('off')
    plt.show()

class CNN(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3) # 입력 채널의 경우, MNIST는 흑백이기에 1, 출력 채녈의 경우 6, 커널 사이즈 3*3으로 수행
        self.pool = nn.MaxPool2d(2,2) # MaxPooling의 경우 커널 2*2, stride = 2
        self.conv2 = nn.Conv2d(6, 16, 3) # 이전 채널에서 6으로 전달, 출력 채널의 경우 16 커널 사이즈 3*3으로 수행
        self.fc1 = nn.Linear(16*5*5, hidden_units) # Flatten 이후 수행하기 때문에 입력은 16 * 5 * 5로 수행
        self.fc2 = nn.Linear(hidden_units, 10) # 숫자 1~10의 Classification Task이기에 10으로 수행

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x))) # pooling layer의 경우, 파라미터 학습 없이 수행이 가능한 연산이기 때문에 하나만 만들고 활용하기
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # FC layer로 전달하기 전에 Flatten으로 수행을 한 다음에 전달
        x = F.relu(self.fc1(x)) # Activation function은 relu로 수행
        x = self.fc2(x) # loss function을 CrossEntropyLoss로 수행하기 때문에 PyTorch의 경우 내부적으로 자동으로 softmax 연산 지원

        return x

def train(model, optimizer, criterion): # 학습 과정 수행
    X_mps = X.to(device)
    y_mps = y.to(device)

    for epoch in range(100): # epoch 100으로 설정
        y_pred = model(X_mps) # training data data에 대한 학습 수행

        loss = criterion(y_pred, y_mps) # model의 predict의 대한 loss 구하기
        print(f'Epoch : {epoch}, Loss: {loss}')

        optimizer.zero_grad() # 이전 역전파로 구한 기울기 0으로 초기화
        loss.backward() # 역전파로 기울기 구하기
        optimizer.step() # 역전파로 구한 기울기를 활용하여 params update

model = CNN(100).to(device) # hidden layer 뉴런 수 100으로 설정
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) # 경사하강법을 SGD를 활용
criterion = nn.CrossEntropyLoss() # Loss Function으로 Cross Entropy Loss로 수행

train(model, optimizer, criterion) # 학습 수행

y_pred = model(X[:5].to(device))
print(y_pred.shape)
print(y_pred.argmax(1))