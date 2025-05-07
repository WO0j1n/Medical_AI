# Perceptron -> 단일 layer로 현재 잘 쓰이지는 않지만 딥러닝 모델의 근간이 됨
# neural network -> 인간의 신경세포를 모방한 알고리즘(Neuron)
# 함수 근사
#     1. 원하는 함 f(*)를 근사할 수 있도록 weight W를 설계
#     2. Input signal과 weight를 곱한 뒤, 더 더하여 즉, 선형 조합을 구성
#     3. 선형 조합의 결과가 임계값보 크면 뉴런이 활성화

# Perceptron의 경우, Bianry Classification을 위해 설계
#     1. 데이터가 Linear separable하다면, 원하는 함수 근사 가능
#     2. XOR 문제의 경우, 비선형 문제이기에 단일 Perceptron으로는 해결할 수 없음.
#     3. 즉, 직선이 두 개가 있어야 하기에 단일 Perceptron에서 Multi-Layer Perceptron이 제안됨.

# Real World의 경우 모든 data는 NOn-Linear 성격을 띄고 있어서 문제