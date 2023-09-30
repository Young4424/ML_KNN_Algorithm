import numpy as np
from collections import Counter
from scipy.spatial import distance
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class KNN:

    # 클래스 생성자에는 k의 기본값을 3으로 지정한다.

    def __init__(self, k = 3):
        self.k = k

    # feature가 x이고, target이 y인 것을 클래스 변수에 저장한다.  
    def fit(self,x,y):
        self.features = x
        self.targets = y

    # euclidean 거리를 계산한다.

    def euclidean_dist(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    # 테스트 데이터와 모든 훈련 데이터간의 거리를 계산하고, 가장 가까운 k-nearest neigbor를 반환한다.

    def get_neighbors(self,test_data):
        distances = []

        for index, training_data in enumerate(self.features):

            # 테스트 데이터와 트레이닝 데이터간의 거리를 계산한다.
            dist = self.euclidean_dist(training_data, test_data)
            distances.append(self.targets[index], dist)

        # 거리 순으로 오름차순 정렬, lambda를 통해 각 튜플 'x'의 두번째 요소 반환
        distances.sort(key=lambda x : x[1])
        



    # 주어진 이웃에 대해 가장 일반적인 클래스 레이블을 찾아서 반환한다.

    # 주어진 이웃에 대해 거리의 역수를 사용하여 가중치를 부여하고, 가중치가 가장 높은 클래스 레이블을 반환한다.

    # 테스트 데이터 세트의 각 