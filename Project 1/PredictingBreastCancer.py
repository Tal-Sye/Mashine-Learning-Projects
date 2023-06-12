import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from google.colab import drive
drive.mount('/content/drive')
path = '/content/breastcancer.csv'
data = pd.read_csv(path)
del data['id']
yes_cancer = len(data[data['result'] == 1])
no_cancer = len(data[data['result'] == 0])
print("Percentage of Subjects with Cancer = ", (yes_cancer/568)*100)
print("Percentage of Subjects without Cancer = ", (no_cancer/568)*100)
sorted_features = data.drop(['result'], 1)
sorted_result = data['result']
X_train, X_test, y_train, y_test = train_test_split(sorted_features, sorted_result)
model = RFE(estimator = LogisticRegression(), n_features_to_select = 3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)
confusion_matrix(y_test, y_pred)
model.score(X_test, y_test)