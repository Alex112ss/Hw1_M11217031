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

class ID3DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y, depth=0):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return Node(value=np.bincount(y).argmax())

        num_samples, num_features = X.shape
        best_gini = 1.0
        best_feature = None
        best_threshold = None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                y_left = y[X[:, feature] <= threshold]
                y_right = y[X[:, feature] > threshold]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gini = (len(y_left) / num_samples) * self._gini(y_left) + (len(y_right) / num_samples) * self._gini(y_right)

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        if best_gini == 1.0:
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

    def _gini(self, y):
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p ** 2)

def main():
    adult = pd.read_csv('adult.data')
    adult_test = pd.read_csv('adult.test')

    X = adult.iloc[:, :-1].values
    y = LabelEncoder().fit_transform(adult['income'])

    test_X = adult_test.iloc[:, :-1].values
    test_y = LabelEncoder().fit_transform(adult_test['income'])

    clf = ID3DecisionTree(max_depth=5)
    tree = clf.fit(X, y)
    predicted = [clf.predict(tree, x) for x in test_X]

    # Replace 0 with '<=50K' and 1 with '>50K'
    predicted_labels = ['<=50K' if label == 0 else '>50K' for label in predicted]
    test_y_labels = ['<=50K' if label == 0 else '>50K' for label in test_y]

    accuracy = metrics.accuracy_score(test_y_labels, predicted_labels)
    print("Accuracy:", accuracy)
    report = classification_report(test_y_labels, predicted_labels)
    print(report)

    # Create a DataFrame with the true and predicted labels
    results_df = pd.DataFrame({'True_Class': test_y_labels, 'Predicted_Class': predicted_labels})

    # Save the results to an Excel file
    results_df.to_excel('classification_results.xlsx', index=False)

if __name__ == '__main__':
    main()
