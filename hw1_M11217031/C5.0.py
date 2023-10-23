import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from sklearn.metrics import classification_report
from rpy2.robjects.packages import importr

# 載入C50套件
c50 = importr('C50')

# 載入訓練數據作為pandas DataFrame
train_data = pd.read_csv('adult.data')

# 載入測試數據作為pandas DataFrame
test_data = pd.read_csv('adult.test')

# 將 'income' 列轉換為分類因子
train_data['income'] = pd.Categorical(train_data['income'])
test_data['income'] = pd.Categorical(test_data['income'])

# 將訓練數據轉換為R的數據框
with localconverter(robjects.default_converter + pandas2ri.converter):
    train_data_r = robjects.conversion.py2rpy(train_data)

# 將測試數據轉換為R的數據框
with localconverter(robjects.default_converter + pandas2ri.converter):
    test_data_r = robjects.conversion.py2rpy(test_data)

# 定義模型的公式
formula = robjects.Formula("income ~ .")

# 使用訓練數據構建C5.0模型
model = c50.C5_0(formula=formula, data=train_data_r)

# 對測試數據進行預測
predictions = robjects.packages.importr('C50').predict_C5_0(model, newdata=test_data_r)

# 將預測結果轉換回Python
with localconverter(robjects.default_converter + pandas2ri.converter):
    predictions_python = robjects.conversion.rpy2py(predictions)

# 真實標籤
true_labels = test_data['income'].tolist()

# 生成分類報告
report = classification_report(true_labels, predictions_python, target_names=['<=50K', '>50K'])
print(report)

# 将真实标签和预测标签保存到Excel文件
data = {'test_y': true_labels, 'train_test_y': predictions_python}
df = pd.DataFrame(data)
df.to_excel('classification_results.xlsx', index=False)
