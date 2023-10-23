import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report

def OneHot_DecisionTree():
    adult = pd.read_csv('adult.data')
    adult_test = pd.read_csv('adult.test')
    
    X = adult.iloc[:,:-1]
    y = adult['income']
    
    test_X = adult_test.iloc[:,:-1]
    test_y = adult_test['income']
    
    #=======================================================================================================================================================
    
    # handle_unknown='ignore'，轉換時用全0代替，维度保持和正常的一致
    ohe = OneHotEncoder(handle_unknown='ignore')
    
    ohe.fit(X)
    
    X_train_ohe = ohe.transform(X).toarray()
    test_X_ohe = ohe.transform(test_X)
    
    ohe_df = pd.DataFrame(X_train_ohe, columns=ohe.get_feature_names_out(X.columns))

    ohe_df.head()
    
    #=======================================================================================================================================================
    
    clf = tree.DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=4, min_samples_leaf=4, min_samples_split=4)
    # 建立分類器
    adult_clf = clf.fit(X_train_ohe, y)       
    
    #=======================================================================================================================================================
    
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(3, 3), dpi=1000)
    # 輸出分類樹圖片
    tree.plot_tree(adult_clf, feature_names=ohe_df.columns, class_names=np.unique(y).astype('str'), filled=True)
    # 儲存分類器圖片, bbox_inches=”tight” : 可防止截斷圖像
    plt.savefig('tree.png', format='png', bbox_inches="tight")
    plt.show()
    
    #=======================================================================================================================================================
    
    # train data & test data
    test_y_pre = adult_clf.predict(X_train_ohe)
    test_y_predicted = adult_clf.predict(test_X_ohe)
    
    # train data & test data 與 train data 做比較
    accuracy1 = metrics.accuracy_score(y, test_y_pre)
    accuracy = metrics.accuracy_score(test_y, test_y_predicted)
    
    # 輸出比較值
    print(accuracy1)
    print(accuracy)
    print(classification_report(test_y, test_y_predicted))
    
    #=======================================================================================================================================================
    
    # 將比較結果寫入 csv 檔中
    dict = {'test_y': test_y, 'train_test_y': test_y_predicted}
    df = pd.DataFrame(dict)
    df.to_csv('Ans.csv', index=False)

if __name__ == '__main__':
    with open('Ans.csv', 'w', encoding='utf-8') as csvfile:
        OneHot_DecisionTree()
