import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

class Node:
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right

class C45DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y, depth=0):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return Node(value=np.bincount(y).argmax())

        num_samples, num_features = X.shape
        best_info_gain_ratio = 0
        best_feature = None
        best_threshold = None

        entropy_parent = self._entropy(y)

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                y_left = y[X[:, feature] <= threshold]
                y_right = y[X[:, feature] > threshold]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                entropy_left = self._entropy(y_left)
                entropy_right = self._entropy(y_right)

                info_gain = entropy_parent - (len(y_left) / num_samples) * entropy_left - (len(y_right) / num_samples) * entropy_right
                split_info = -(len(y_left) / num_samples) * np.log2(len(y_left) / num_samples) - (len(y_right) / num_samples) * np.log2(len(y_right) / num_samples)

                if split_info == 0:
                    continue

                info_gain_ratio = info_gain / split_info

                if info_gain_ratio > best_info_gain_ratio:
                    best_info_gain_ratio = info_gain_ratio
                    best_feature = feature
                    best_threshold = threshold

        if best_info_gain_ratio == 0:
            return Node(value=np.bincount(y).argmax())

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left_tree = self.fit(X[left_indices], y[left_indices], depth + 1)
        right_tree = self.fit(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left_tree, right=right_tree)

    def predict(self, node, X):
        if node.value is not None:
            return node.value

        if X[node.feature] <= node.threshold:
            return self.predict(node.left, X)
        else:
            return self.predict(node.right, X)

    def _entropy(self, y):
        p = np.bincount(y) / len(y)
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

def main():
    adult = pd.read_csv('adult.data')
    adult_test = pd.read_csv('adult.test')

    X = adult.iloc[:, :-1].values
    y = LabelEncoder().fit_transform(adult['income'])

    test_X = adult_test.iloc[:, :-1].values
    test_y = LabelEncoder().fit_transform(adult_test['income'])

    clf = C45DecisionTree(max_depth=5)
    tree = clf.fit(X, y)
    predicted = [clf.predict(tree, x) for x in test_X]

    accuracy = metrics.accuracy_score(test_y, predicted)
    print("Accuracy:", accuracy)

    # Print classification report with replaced labels
    report = classification_report(test_y, predicted, target_names=['<=50K', '>50K'])
    print(report)

if __name__ == '__main__':
    main()
