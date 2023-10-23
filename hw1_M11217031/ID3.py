import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from sklearn.metrics import classification_report
from rpy2.robjects.packages import importr

# 載入 'rpart' 套件以進行 ID3 分類
rpart = importr('rpart')

# 載入訓練數據作為 pandas DataFrame
train_data = pd.read_csv('adult.data')

# 載入測試數據作為 pandas DataFrame
test_data = pd.read_csv('adult.test')

# 將 'income' 列轉換為分類因子
train_data['income'] = pd.Categorical(train_data['income'])
test_data['income'] = pd.Categorical(test_data['income'])

# 將字串屬性轉換為因子（分類）
for col in train_data.select_dtypes(include=['object']).columns:
    train_data[col] = pd.Categorical(train_data[col])
    test_data[col] = pd.Categorical(test_data[col])

# 將訓練數據轉換為 R 的數據框
with localconverter(robjects.default_converter + pandas2ri.converter):
    train_data_r = robjects.conversion.py2rpy(train_data)

# 將測試數據轉換為 R 的數據框
with localconverter(robjects.default_converter + pandas2ri.converter):
    test_data_r = robjects.conversion.py2rpy(test_data)

# 定義模型的公式
formula = robjects.Formula("income ~ .")

# 使用 'rpart' 套件中的 'rpart' 函數來建立模型（ID3）
model = rpart.rpart(formula=formula, data=train_data_r, method="class")

# 在測試數據上執行預測
predictions = robjects.r.predict(model, newdata=test_data_r, type="class")

# 將預測結果轉回 Python
with localconverter(robjects.default_converter + pandas2ri.converter):
    predictions_python = robjects.conversion.rpy2py(predictions)

# 真實標籤
true_labels = test_data['income'].tolist()

# 生成分類報告
report = classification_report(true_labels, predictions_python, target_names=['<=50K', '>50K'])
print(report)

# 將真實標籤和預測標籤保存到 Excel 文件
data = {'真實標籤': true_labels, '預測標籤': predictions_python}
df = pd.DataFrame(data)
df.to_excel('classification_results_id3.xlsx', index=False)
