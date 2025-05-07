# Pooling Layer의 종류
#     - CNN에서는 Convolution Layer뿐만 아니라 다양한 Pooling Layer도 사용
#     - 데이터의 공간적인 특성을 유지하면서도, 특징의 크기를 줄여줌
#
#     - Max Pooling
#     - Average Pooling
#     - Global Average Pooling

# Max Pooling
#     - 특정 위치에서 가장 강한 신호만을 출력
#
#     - Conv Layer와 마찬가지로 슬라이딩 방식으로 작동
#     - 보통은 2 x 2 크기의 커널로 stride 2로 사용
#     - 따라서 출력의 크기는 1/4로 줄어듦

# Average Pooling
#     - 특정 위치에서 신호들의 편균을 구해 출력
#
#     - Max Pooling과 비슷한 특성을 지님
#     - 단, Max Pooling에 만큼 자주 사용되지는 않음 -> 주로 Global Average Pooling으로 더 많이 사용됨

# 그러면 왜 우리는 풀링 레이어를 사용하는 것일까
#     - Small Translation Invariance(Invariance-> Input의 변화가 있어도 출력의 변화가 없도록하는 성질)
#         - Pooling Layer을 사용함으로써 입력에 약간의 변화가 있더라도 출력은 동일
#         - Shift Invariance
#         - Rotational Invariance
#         - Scale Invariance
#         =>  위치, 회전, 확대 변형의 변형에도 강인한 성징을 부여할 수 있음

    # - Computation Efficiency
    #     특별한 파라미터 없이 정해진 연산으로 특징의 크기를 감소
    #
    #     - 대부분의 경우 Conv Layer보다 훨씬 연산량이 적음
    #     - 추가적인 파라미터 학습이 필요 ㅇ벗음
    #     - Overfitting엗 어느 정도 도움이 된다고 알려져 있음

# Global Average Pooling(GAP)
#     - 극단적으로 Pooling을 하는 방식으로, 한 채널에 대해서 평균을 계산
#     - 한장의 Feature map에 대해서 수행
#     - depth에 대해서 수행이 되어지기에 동일한 길이의 결과가 나옴
#     - depth의 길이와 Classification을 수행하는 Class의 수가 같은 경우 FC Layer 없이 GAP를 통해서 바로 출력을 만들어내는 것이 가능함.
#         - 파라미터를 대폭 줄어들고 성능 향상이 가능
