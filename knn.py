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

    
    def predict(self, test_data, true_labels, class_names):
        for i, test_instance in enumerate(test_data):
            neighbors = self.get_neighbors(test_instance)
            vote_result = self.majority_vote(neighbors)
            
            # Convert numerical class labels to string class names
            computed_class = class_names[vote_result]
            true_class = class_names[true_labels[i]]

            print(f'Test Data Index : {i} Computed class : {computed_class}, True class : {true_class}')

    