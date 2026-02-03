import sklearn as sk
import pandas as pd

data = sk.datasets.load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

x = df[data.feature_names]
y = df["target"]

x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, test_size=0.2, random_state=42)

model = sk.tree.DecisionTreeClassifier(random_state=9)
model.fit(x_train, y_train)

predict = model.predict(x_test)
print(f"precisão da árvore de decisão: {sk.metrics.accuracy_score(y_test, predict)}")
print(sk.metrics.classification_report(y_test, predict))

print("################")

model_rf = sk.ensemble.RandomForestClassifier(n_estimators=100,random_state=42)
model_rf.fit(x_train, y_train)

predict_rf = model_rf.predict(x_test)
print(f"precisão da floresta aleatória: {sk.metrics.accuracy_score(y_test, predict_rf)}")
print(sk.metrics.classification_report(y_test, predict_rf))

print("apos o commit")