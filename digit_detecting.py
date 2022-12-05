from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

digits = load_digits()

# plt.gray()
# plt.matshow(digits.images[0])
# plt.show()

x = digits.data
y = digits.target

# split data for training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = LogisticRegression()
model.fit(x_train, y_train)

# manual testing
# print('target value', digits.target[1700])
# result = model.predict([digits.data[1700]])
# print('test res', result)

accuracy = model.score(x_test, y_test)
print('model accuracy', accuracy)

# confusion matrics
y_predicted = model.predict(x_test)
confusion = confusion_matrix(y_test, y_predicted)
# print(confusion)

plot_confusion_matrix(model, x_test, y_test)
plt.show()