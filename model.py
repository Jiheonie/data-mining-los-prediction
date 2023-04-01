from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from set_data import X, y

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

# model = MLPClassifier(random_state=10)
model = RandomForestClassifier(random_state=10)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# print(y_pred)

cmatrix = confusion_matrix(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='macro')
recall = metrics.recall_score(y_test, y_pred, average='macro')
f1 = metrics.f1_score(y_test, y_pred, average='macro')
accuracy = metrics.accuracy_score(y_test, y_pred, normalize=True)

print("Ma trận nhầm lẫn: ")
print(cmatrix)
print("Độ chính xác (precision): {:7.2f}%".format(precision*100))
print("Độ triệu hồi (recall): {:7.2f}%".format(recall*100))
print("Độ đo F1 (F1-measure): {:7.2f}%".format(f1*100))
print("Độ chính xác (accuracy): {:7.2f}%".format(accuracy*100))