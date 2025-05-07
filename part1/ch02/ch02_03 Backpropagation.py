# Loss Function
#     - 샘플에 대해서 원하는 값 y와 NN의 출력 predict의 차이를 나타내는 함수
#     - Task 별로 다음과 같은 손실함수들을 대표적으로 사용
#         - Classification Task: Cross-entropy
#         - Regression Task: MSE(Mean Squared Error)
#         - Detection Task: Soft IoU

# Kullback - Leibler divergence
#     - 확률 분포 P를 다른 확률 분포 Q로 근사했을 때 정보량의 손실
#     - KL-divergnece -> 분포P와 분포 Q의 Cross Entropy와 분포 P의  entropy의 차이로 이루어짐
#     - P: 우리가 풀어야 하는 확률 분포  -> Ground Truth Distribution
#     - Q: Neural Network의 Predict Distribution
#     - KL-divergence를 작게 하기 위해서는 Neural Network의 Predict가 잘 되어져야 하며 이는 Cross Entropy H(P, Q)를 최소화하는 것

# Mean Squared Error, MSE
    # - Regression Task 등에서는 대체로 MSE를 사용
    # - 풀어야 하는 확률 분포 Gaussian Distribution인 경우 사용
    # - 이때 MSE를 최소화하는 것으로 Neural Network는 원래 확률 분포를 예측하도록 학습

# 경사하강법(Gradient Descent) - Loss Function의 값을 최소화하는 방법
#     - 손실의 기술기를 게산하여, 손실이 낮아지는 방향으로 Weight를 업데이터
#     - 학습률을 잘 설정하는 것이 중요 (Learning rate) -> Hyper Parameter로 직접 설정해야 함.
#     - 기울기를 계산하기 위해서, Neural Network는 미분 가능해야 함.
#     - 다양한 방식의 Gradient Descent 알고리즘이 존재
#         - SGD
#         - Adam
#         ...

# 대리 함수(Surrogate Function)
#     - Neural Network에서 사용하는 함수가 Gradient Descent에서 사용하기 어려운 경우 대신 사용
#     - 예로 단위 계단 함수(Unit Step Function) 대신 아래의 함수들을 대신 사용
#         - Sigmoid
#         - Tanh
#         - ReLU
#     - 다른 대리 함수로는 다음의 함수가 존재
#         - Softmax -> Argmax 함수를 soft하게 변형한 함수

# Backpropagation, 역전파
#     - 기울기를 수치적으로 쉽게 계산하기 위한 방법
#     - hidden layer의 증가로 해석적으로 기울기를 직접 구하기 어려움
#     - 손실 함수를 미분하여 출력층에서 입력층으로 전달


