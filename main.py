# main.py
import numpy as np
from sklearn.datasets import load_iris
from knn import KNN

# 데이터셋 로드하기
iris = load_iris()
data = iris.data
labels = iris.target

# 트레이닝, 테스트 데이터 초기화 하기
train_data = []
train_labels = []
test_data = []
test_labels = []

# 규칙에 따라, 데이터를 나누고 반복하기 

for i in range(len(data)):
    if (i + 1) % 15 == 0:  # 15번쨰 데이터 마다 테스트 데이터로 추가함
        test_data.append(data[i])
        test_labels.append(labels[i])

    else:  # 그 이외의 경우에는 트레이닝 데이터로 추가됨
        train_data.append(data[i])
        train_labels.append(labels[i])

# 리스트를 넘파이 배열로 변환하기 (더 다루기 쉽게)
train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

def main():
    knn = KNN(k=3)
    knn.fit(train_data, train_labels)
    class_names = iris['target_names']
    knn.predict(test_data, test_labels, class_names)  # Passing true labels for comparison

if __name__ == "__main__":
    main()
