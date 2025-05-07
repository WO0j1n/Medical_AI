# Multi-Layer Perceptron
#     1. Perceptron을 다층(Multi-Layer)로 쌓은 모델
#     2. 단일 Perceptron을 여러 층 쌓은 것으로 즉, 여러 개의 선형 조합을 순차적으로 수행
#     3. Input Layer - Hidden Layer - Output Layer로 구성
#     4. 각 노드와 모두 연결한 layer를 -> Fully Connected Layer라고 함.
#     5. MLP는 이러한 Fully-Connected Layer가 여러 층 쌓여서 만들어진 것

# 비선형성
#     1. MLP의 경우 FC Layer로 여러 층 쌓았을 때, 선형 조합을 여러 층 쌓았음에도 이는 하나의 선형 조합으로 표현할 수 있음.
#     2. 비선형성을 띄게하기 위해 활성화 함수를 사용
#     3. 따라서 Hidden layer의 Node의 출력에 활성화 함수를 적용해서 Non-Linear 성질을 가져

# Activation function
#     1. Sigmoid
#         - 출력값 항상 0에서 1사이의 값을 갖는다
#           - 입력값의 절댓값이 큰 경우 기울기가 0에 가까워짐 -> Gradient Vanishing
#           - 지수함수를 계산 -> 연산량 소모 큼
#     2. tanh
#         - 출력 값이 항상 -1에서 1 사이로 지정 -> 굉장히 작은 값
#         - 입력 값의 절댓값이 큰 경우 기울기가 0에 가까워짐 ->  Gradient Vanishing
#         - 지수함수를 계사해줘야 함
#     3. ReLU
#         - 출력 값이 양의 방향으로는 제한이 없음
#         - 계산 효율이 굉장히 높음 -> max 연산만 수행하기 때문
#         - 입력값이 음수인 경우, 값의 크기와 상관없이 항상 출력이 0 -> 음수인 경우 기울기 0이지만 계산 효율성 때문에 인기 많음
#     4. Leakly ReLU 
#         - ReLU의 대부분의 특징을 가지고 있음
#         - 입력값이 음수인 경우에도 출력값이 0으로 고정되지 않음 -> ReLU 단점을 보완했으나 매번 좋은 성능을 가져오는 것은 아님
#     5. Maxout
#     6. ELU

# Feature Extraction
#     - Hidden layer의 뉴런을 새로운 특징을 추출하는 함수로 볼 수 있음
#     - 저수준의 데이터를 고수준의 데이터의 특징으로 변환하면서 학습
#     - low level :픽셀 값
#     - high level: 꼬리의 길이, 혀의 길이
#     - 점차 학습을 통해서 Hidden layer을 통해서 새롭고 고차원 특징을 추출할 수 있게 된다.

# 출력의 변환
#     - 원하는 출력 형태에 따라 출력을 변환해줄 필요가 있음
#     - 가우시안 분포: 선형 함수
#     - 베르누이 분포: 시그모이드 함수 -> Binary Classification
#     - 멀티누이(Multinoulli) 분포:  Softmax 함 -Multi Classification

# Universal Approximation Theorem(UAT)
#     - 하나의 hidden layer를 갖고, 시그모이드 형식의 activation function을 사용하는 NN은 임의의 연속인 다변수 함수를 근사할 수 있음
#         - 많은 layer 없이 하나의 hidden layer로도 다양한 함수 근사할 수 있음.
#
#     Q: 그럼 하나의 은닉층만 사용하면 되는 것이지 왜 여러 hidden layer을 써야하냐
#         -> 근사가 가능함을 알려주는 것이지 근사를 하기 위해 얼마나 많은 뉴런과 가중치가 필요한지는 말하지 않음
#     - Neural Network(NN)의 범용성에 대한 이론적 뒷받침
#     - 다양한 형태의 이론적 분석들이 이루어지고 있음
#     - 다만, 얼마나 많은 뉴런이 필요한지, weight를 어떻게 설정해야 하는지 등은 모름
#     - 실제론, 모델의 잡성을 늘리기 위하여 여러 개의 hidden layer를 사용함.