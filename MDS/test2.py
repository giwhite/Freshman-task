from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt

# 生成模拟数据
X = np.sort(5 * np.random.rand(100, 1), axis=0)
Y = np.sin(X).ravel()

# 定义 SVR 模型
model = SVR(kernel='rbf', C=1e3, gamma=0.1, epsilon=0.1)

# 训练模型
model.fit(X, Y)

# 预测新的数据点
X_test = np.linspace(0, 5, 100)[:, np.newaxis]
Y_test = model.predict(X_test)

# 绘制结果
plt.scatter(X, Y, color='black', label='data')
plt.plot(X_test, Y_test, color='red', label='SVR')
plt.legend()
plt.show()
