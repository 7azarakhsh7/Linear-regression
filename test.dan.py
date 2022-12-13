import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from mpmath import *
from matplotlib import animation

data = np.loadtxt('house_price.txt', delimiter=',')

X = data[:, 0]
y = data[:, 1]

# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train.reshape(-1, 1), y_train)
y_pred = model.predict(X_test.reshape(-1, 1))
plt.scatter(X_train, y_train,color='g')
plt.plot(X_test, y_pred,color='k')
plt.show()
print(model.coef_)
print(model.intercept_)
# ---------------------------------------
coef = np.polyfit(X,y,1)
poly1d_fn = np.poly1d(coef)
plt.plot(X,y, 'yo', X, poly1d_fn(X), '--k')
plt.figure(figsize=(10, 6))
print(coef)
# ---------------------------------------

sns.regplot(X,y,color='blue')

# ---------------------------------------
plt.scatter(X, y)
plt.xlabel('size')
plt.ylabel('price')
plt.title('House Dataset')
plt.show()

# --------------------------------------------------

a = 0
b = 0

L = 0.0001  # The learning Rate
e = 1000  # The number of iterations to perform gradient descent

n = float(len(X))  # Number of elements in X

mp.dps=5

# Performing Gradient Descent
for i in range(e):
    Y_pred = a * X + b  # The current predicted value of Y
    D_a = abs(mpf(-2 / n * sum(X * (y - Y_pred))))  # moshtagh jozee a
    D_b = abs(mpf(-2 / n * sum(y - Y_pred)))  # moshtagh e jozee b
    a = abs( mpf(a - L * D_a)) # Update m
    b = abs(mpf(b - L * D_b) ) # Update c

print(a, b)

y_pred = a*X + b
fig = plt.figure(dpi=100, figsize=(10,6))
plt.scatter(X, y)
line = plt.plot(X, y_pred, 'k')


def animate():
    line.set_ydata(y_pred)
    return line


anim = animation.FuncAnimation(fig, animate, np.arange(0, 20), interval=200, repeat_delay=1000)
plt.show(anim)