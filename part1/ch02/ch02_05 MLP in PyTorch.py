# Pytorch를 이용하여 MLP 학습
#     - 입력 샘플의 4개의 특징을 모두 입력받고 3개의 클래스 중 하나로 분류
#     - 1개의 은닉층을 가지고, 은닉층에서 입력 특징보다 더 많은 특징을 추출
#     - Cross-entropy loss를 사용

# Pytorch의 경우, Neural Network는 기본적으로 float32 다룸

# PyTorch의 경우, Fully Connected Layer 선언하는 방법
# tensorflow.keras.layers.Dense()
# torch.nn.Linear(input_features, output_features, bias = True, device = None, dtype = None)
#     - 선형 연산만 해주는 것으로 간단한 연산이 이루어짐
#     - input/ output features 필수
#     - device = CPU, GPU

 # ReLu 함수 사용하기
# import tensorflow as tf
# from tensorflow.keras import models, layers
# model = models.Sequential()
# model.add(layers.Dense(10, activation = 'relu' ))
# tf.keras.activations.relu()
# torch.nn.ReLU(inplace = False)
#     - inplace 연산을 할것인지 하지 않을 것인지 수행
#     - 일반적으로는 False로 설정
# torch.nn.functional.relu(input, inplace = False)
#     - 함수로 가져와서 사용하는 경우의 수가 더 많음

# 모델, 경사하강법, 손실함수 설정
import tensorflow as tf
import torch.nn

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(10, activation = 'relu'))
# model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(lr = 0.0001), metrics = ['accuracy'])

# import torch
# torch.optim.SGD(params, lr, momentum, dampening, weight_decay = 0, nesterov = False, maximize=False, foreach=None, differentiable=False)
#     - parameter(필수), learning_rate(필수), momentum(알아두기), weight_decay(알아두기)

# 손실함수
# torch.CrossEntropyLoss(weight, size_average, ignore_index, reduction, reduction, label_smoothing)
#     - weight(각 데이터 입력 신호의 가중치를 두어 중요도 설정), reduction(Mean, sum -> 손실의 평균, 합을 반환)

# Pytorch의 경우 Softmax function의 경우, Cross-Entropy 연산이 Log Softmax와 NLL Loss를 적용하는 것과 같은 일을 수행
#     - Negative Log Likelihood -> Cross entropy 공식과 비슷함
#     - Softmax 함수 기능을 Loss function으로 수행한다고 보면 돼 -> softamx의 값이 작아져 미분이 불안해지는 현상 방지
#     - Log Softmax, NLL의 경우, 미분이 안정함

# optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
# criterion = torch.nn.CrossEntropyLoss()

# Loss를 줄이기 위해서 hudden layer을 깊게 하거나, 모델의 구조를 더욱 복작합게 할 수 있지만 가장 쉬운 건 Node 수를 키우는 것


from sklearn import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F

datasets = datasets.load_iris()

X = torch.tensor(datasets.data, dtype = torch.float32)
y = torch.tensor(datasets.target)

class MLP(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.fc1 = nn.Linear(4, hidden_units) # features 총 4개 output_ feature를 hidden_units로 설정
        self.fc2 = nn.Linear(hidden_units, 3)# Classification이 총 3개이기에 output_features를 3으로 설정

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x) # torch.nn.functional.relu(), activation function
        x = self.fc2(x)
        # x = torch.nn.functional.relu(self.fc2(x)) -> 맨 마지막 layer에는 적용하지 않음. Cross Entropy를 통해서 내부에서 softmax가 수행이 됨

        return x

def train(model, optimizer, criterion): # model training ->  model, optimizer, criterion(loss function)
    for epoch in range(100): # 에폭 100으로 설정
        y_pred = model(X) # 근사 함수 수행

        loss = criterion(y_pred, y) # loss 값 구하기
        print(f'Epoch: {epoch} / Loss: {loss}')

        optimizer.zero_grad() # 이전 역전파를 수행하면서 구한 기울기 초기화 .zero_grad()
        loss.backward() # backpropagation
        optimizer.step() # backpropagation으로 수행한 기울기를 이용해서 parameter update 수행 (w = w - lr * grad를 의미)


model = MLP(5) # 뉴런 수를 5로 전달
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1) # Optimizer로 SGD를 사용했으며 learning_rate를 0.1로 수행
criterion = nn.CrossEntropyLoss() # Loss Function

train(model, optimizer, criterion)


model2 = MLP(100)
optimizer2 = torch.optim.SGD(model2.parameters(), lr = 0.1)
criterion2 = torch.nn.CrossEntropyLoss()

train(model2, optimizer2, criterion2)

model3 = MLP(1000)
optimizer3 = torch.optim.SGD(model3.parameters(), lr = 0.1)
train(model3, optimizer3, criterion)