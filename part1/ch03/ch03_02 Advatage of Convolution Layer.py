# 귀납적 편향 (Inductive Bias)
#     - 일반화 성능을 높이기 위해 학습한 편향
#     - Task마다 더 좋은 성능을 얻기 위하여, inductive bias를 사전 정보로써 가정
#     - 이미지 데이터는 다음과 같은 특징들을 가지고 있음
#
#     - Stationarity of statistics
#     - Locality of pixcel dependencies


# Stationarity of statistics
#     - 지역적인 통계량이 비슷한 영역들이 이미지에 다수 존재함
#
#     - 이미지에는 비슷한 물체들이 다수 존재할 확률이 높음
#     - 따라서 하나의 특징 추출 방식으로 다양한 위치에서 특징을 추출해 낼 수 있음 -> low - level부터 high - level까지
#     - Stationarity은 주로 시계열 데이터에서 많이 사용되는 특징 혹은 가정으로 정상성이라고 함
#     - 시간적 순서, 공간적 정보에서 벗어나 일관적인 성질을 의미
#     - 하나의 필터/커널로 입력의 이미지에서 여러 물체에 적용할 수 있음

# Locality of pixel dependencies
#     - 근접한 픽셀들끼리는 종속성을 가지며, 지역적인 특징을 만듦
#     - 하나의 특징을 구성하는 픽셀들은 근접해 있음
#     - 따라서 지역적인 부분들에 대해서만 특징 추출
#     - 코를 파악하기 위해서는 코 근접 픽셀을 보면 되는 것이지 귀 주변 픽셀을 보지는 않음
#     - 고양이의 일부 얼굴 사진만으로도 얼추 추정할 수 있듯이 종속성을 가지고 있음
#     => 멀리 떨어져 있는 픽셀들의 특징이 아닌 근접한 local 적인 특징만 봐도 괜찮을 수도 있다는 것을 의미

# Convolution layer의 경우, Stationarity of statistics + Locality of pixel dependencies 을 Inductive bias로 가지고 있음
#     - Convolution layer의 장점
#         - Sparse Interaction
#         - Parameter Sharing
#         - Equivariant representation

# Sparse Interaction, 희소 상호작용
#     - 새로운 특징 추출을 위해 계산되는 특징의 수가 한정적임
#     - 커널의 크기는 일반적으로 이미지의 크기보다 매우 작음
#     - Fully Connected Layer는 하나의 특징 추출을 위해서 모든 Pixel의 정보를 이용
#     - Convolution Layer의 경우 Kernel size만큼의 pixel 정보만들 이용 -> 연산량 감소
#         - Input Image = 1024 * 1024인 경우,
#             - FC Layer의 경우 2^20의 크기 연산량 요구
#             - Convolution Layer의 경우, Kernel size가 3 * 3인 경우, Input Size의 크기와 상관없이 Kernel size만큼 9번의 연산량을 가짐
#             - 즉, Convolution Layer를 통해서, Local feature에 집중
#     - 이미지의 지역성(Locality) 특징의 활용

# Parameter Sharing, 매개변수 공유
#     - 하나의 특징 추출함수로 모든 위치에서 사용
#     - Fully Connected Layer는 매 위치에서 각기 다른 특징 추출 함수가 필요
#     - Convolution Layer에서는 고정된 커널로 모든 위치에서 같은 특징 추출
#     - 이미지의 정상성(Stationary) 특징의 활용
#     - FC Layer의 경우, 모든 픽셀에 대해서 다른 파라미터가 필요 -> Convolution Layer의 경우 커널 사이즈의 해당하는 크기의 params만 필요

# Equivariant representation, 등변 표현
#     - 입력 이미지가 이동하면, 결과도 같은 방식으로 변형
#     - 입력 이미지가 이동한다면 특징도 같은 방식으로 이동
#     - 확대나 회전 등 다른 변형에 대해서는 등변성(Equivariant)을 가지고 있지 않음.
#     - 이동에 대해서만 등변성을 가지고 있음 -> 위치 변형에 대한 등변성이라고 표현함
#     - 이는 Object Detection에서 객체가 이동함에도 추적이 가능