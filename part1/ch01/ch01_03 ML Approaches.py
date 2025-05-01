# 머신러닝을 통한 문제 접근
#
# 1. 다양한 학습 방법
#     - Supervised Learning : 정답이 주어졌을 때, 모델의 출력이 정답과 유사해지도록 학습
#     - Unsupervised Learning: 정답이 주어지지 않았을 때, 데이터내의 특정 구조를 찾도록 학습
#     - Reinforcement Learning: 보상을 최대로 하도록, 일련의 행동을 정답 없이 학
#
# 2. Supervised Learning
#     - 모든 샘플에대해서 지도(supervision)가 주어졌을 때, 모델을 학습하는 방법
#     - 조직 영상에서 세포의 영역을 검출
#     - 모든 샘플에 대해서, 세포의 영역이 정답으로 주어지고 모델을 학
#
# 3. Weakly Supervised Learning
#     - 모든 샘플들에 대해서 제한적인 지도가 주어졌을 때, 모델을 학흡하는 방법
#     - 조직 영상에서 세포의 영역을 검출(Object segmentation)
#     - 모든 샘플들에 대해서, 세포의 영역 일부분만이 정답으로 주어지고 모델을 학습
#
# 4. Semi Supervised Learning
#     - 일부분의 샘플들에 대해서만 지도가 주어졌을 때, 모델을 학습하는 방법
#     - 조직 영상에서 세포의 영역을 검출(Object Segmentation)
#     - 일부 샘플들에 대해서, 세포의 영역이 정답으로 주어지고 모델을 학
#
# 5. Unsupervised Learning
#     - 지도가 없는 혹은 구하기 불가능한 경우, 데이터 내의 구조를 찾기 위한 학습 방법
#     - Whloe Slide Image로부터 사이즈가 너무 크기에 이를 이미지 패치들을 특정 기준으로 군집화(Clustering)
#     - 정답없이 샘플들만 주어져서, 정해진 기준에 따라서 task 해결
#
# 6. Self-Supervised Learning
#     - 풀고자 하는 작업을 해결할 때 도움이 될만한 표현을 학습하기 위한 방법
#     - 흉부 X-ray 영상으로부터 암진단
#     - 정답 없는 대량의 샘플들로 표현 학습
#     - 이후, 정답이 있는 소량의 샘플들로 지도학습