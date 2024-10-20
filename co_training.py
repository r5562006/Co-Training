from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# 生成數據
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_unlabeled, y_train, _ = train_test_split(X, y, test_size=0.9, random_state=42)
X_unlabeled, X_test, _, y_test = train_test_split(X_unlabeled, y, test_size=0.5, random_state=42)

# 初始模型訓練
model1 = RandomForestClassifier()
model2 = RandomForestClassifier()
model1.fit(X_train[:, :10], y_train)
model2.fit(X_train[:, 10:], y_train)

# 共訓練迭代
for _ in range(10):
    y_unlabeled_pred1 = model1.predict(X_unlabeled[:, :10])
    y_unlabeled_pred2 = model2.predict(X_unlabeled[:, 10:])
    high_confidence_idx1 = np.where(model1.predict_proba(X_unlabeled[:, :10]).max(axis=1) > 0.9)[0]
    high_confidence_idx2 = np.where(model2.predict_proba(X_unlabeled[:, 10:]).max(axis=1) > 0.9)[0]
    X_train = np.vstack((X_train, X_unlabeled[high_confidence_idx1]))
    y_train = np.hstack((y_train, y_unlabeled_pred1[high_confidence_idx1]))
    X_train = np.vstack((X_train, X_unlabeled[high_confidence_idx2]))
    y_train = np.hstack((y_train, y_unlabeled_pred2[high_confidence_idx2]))
    X_unlabeled = np.delete(X_unlabeled, np.union1d(high_confidence_idx1, high_confidence_idx2), axis=0)
    model1.fit(X_train[:, :10], y_train)
    model2.fit(X_train[:, 10:], y_train)

# 評估模型
y_test_pred = model1.predict(X_test[:, :10])
print("Accuracy:", accuracy_score(y_test, y_test_pred))