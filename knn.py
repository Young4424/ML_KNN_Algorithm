import numpy as np
from collections import Counter

class KNN:

    # 클래스 생성자에는 k의 기본값을 3으로 지정한다. k는 가장 가까운 이웃의 수를 나타냄

    def __init__(self, k = 3):
        self.k = k 

    # feature가 x이고, target이 y인 것을 클래스 변수에 저장한다. 
    # x : 꽃받침의 길이(sepal length), 꽃받침의 폭(sepal width), 꽃잎의 길이 (petal length), 꽃잎의 폭(petal width)
    
    def fit(self,x,y):
        self.features = x[:, 0:4]
        self.targets = y

    # euclidean 거리를 계산한다.

    def euclidean_dist(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    # 테스트 데이터와 모든 훈련 데이터간의 거리를 계산하고, 가장 가까운 k-nearest neigbor를 반환한다.

    def get_neighbors(self,test_instance):

        distances = [] # 테스트 인스턴스와, 각 트레이닝 인스턴스 사이의 거리가 저장된다.
        for index, training_instance in enumerate(self.features):
            
            # 각 트레이닝 인스턴스와 테스트 인스턴스 사이의 거리를 계산한다.
            dist = self.euclidean_dist(training_instance, test_instance)

            # 거리와 해당 트레이닝 인스턴스의 레이블을 튜플로 묶어 리스트에 추가한다.
            distances.append((self.targets[index], dist))

        # 거리 순으로 오름차순 정렬하기 위해, lambda를 이용해서, 가장 가까운 이웃이 리스트 앞쪽에 위치하도록 한다.
        distances.sort(key=lambda x : x[1])

        # 가장 가까운 k개의 이웃을 선택한다.
        neighbors = distances[:self.k]
        return neighbors

    # 주어진 이웃에 대해 가장 일반적인 클래스 레이블을 찾아서 반환한다.

    def majority_vote(self,neighbors):
        class_votes = Counter() # 각 클래스에 대한 투표수를 기록하는 카운터를 생성한다.
        for neighbor in neighbors:
            
            # 각 이웃이 어떤 클래스에 속하는지 확인 후, 해당 클래스의 투표 수를 증가시킨다.
            class_votes[neighbor[0]] += 1 

        return class_votes.most_common(1)[0][0] # 가장 많은 투표를 받는 클래스를 반환한다.


    def predict(self, test_data, true_labels, class_names,weighted=False):
        for i, test_instance in enumerate(test_data):
            
            # 각 테스트 인스턴스에 대해 가장 가까운 이웃을 찾는다.
            neighbors = self.get_neighbors(test_instance)

            if weighted:
                vote_result = self.weighted_majority_vote(neighbors)

            else:
                vote_result = self.majority_vote(neighbors)

    
            # 숫자 레이블을 클래스 이름으로 바꾼다.
            
            computed_class = class_names[vote_result]
            true_class = class_names[true_labels[i]]

            print(f'Test Data Index : {i} Computed class : {computed_class}, True class : {true_class}')
