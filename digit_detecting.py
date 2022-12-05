from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

digits = load_digits()

# plt.gray()
# plt.matshow(digits.images[0])
# plt.show()

x = digits.data
y = digits.target

# split data for training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(x_train.shape)