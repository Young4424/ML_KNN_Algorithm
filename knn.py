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

    def get_neighbors(self,test_instance):
        distances = []

        for index, training_instance in enumerate(self.features):

            # 테스트 데이터와 트레이닝 데이터간의 거리를 계산한다. 튜플을 distance에 추가
            dist = self.euclidean_dist(training_instance, test_instance)
            distances.append((self.targets[index], dist))


        # 거리 순으로 오름차순 정렬, lambda를 통해 각 튜플 'x'의 두번째 요소 반환
        distances.sort(key=lambda x : x[1])

        # 처음 K개의 항목을 가져온다.
        neighbors = distances[:self.k]
        return neighbors

    # 주어진 이웃에 대해 가장 일반적인 클래스 레이블을 찾아서 반환한다.

    def majority_vote(self,neighbors):
        class_votes = Counter() # 각 클래스에 대한 가중치 투표를 보유할 Counter
        for neighbor in neighbors:
            class_votes[neighbor[0]] += 1 # 이웃의 클래스에 대한 투표 수를 증가시킨다.

        return class_votes.most_common(1)[0][0] # 가장 많은 투표를 받는 클래스를 반환한다.



    # 주어진 이웃에 대해 거리의 역수를 사용하여 가중치를 부여하고, 가중치가 가장 높은 클래스 레이블을 반환한다.

    def weighted_majority_vote(self,neighbors):
        class_votes = Counter() # 각 클래스에 대한 가중치 투표를 보유할 Counter
        for neighbor in neighbors:
            
            # 이웃의 클래스에 대한 투표 수를 증가시킨다. 거리의 역수로 가중치를 부여한다.
            class_votes[neighbor[0]] += 1 / (neighbor[1] + 1e-10) # 0으로 나누는 것을 피하기 위함

        return class_votes.most_common(1)[0][0]
    
    def predict(self, test_data, vote_type='majority'):
        predictions = [] # 에측을 저장할 리스트
        for test_instance in test_data:
            neighbors = self.get_neighbors(test_instance) # 테스트 인스턴스에 대한 가장 가까운 이웃을 가져온다.
            if vote_type == 'majority':
                # 다수결 투표로 사용하여 테스트 인스턴스의 클래스 결정한다. 
                vote_result = self.majority_vote(neighbors)

            else:
                # 가중치가 부여된 다수결 투표를 사용하여 테스트 인스턴스의 클래스를 결정한다.
                vote_result = self.weighted_majority_vote(neighbors)

            predictions.append(vote_result) # 결과를 저장한다.

        
        return predictions # 에측의 리스트를 반환한다.

        




