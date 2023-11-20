import numpy as np


# 定义激活函数（这里使用ReLU）
def relu(x):
    return np.maximum(0, x)


# 定义均方误差损失函数
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# 定义导数
def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true)



class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        # 計算輸出層的誤差
        delta2 = mse_loss_derivative(y, self.a2)

        # 計算隱藏層的誤差
        delta1 = np.dot(delta2, self.W2.T) * (self.a1 > 0)
        #print(delta1)

        # 計算梯度
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m

        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m
        #print(dW1)

        # 更新参数
        print(self.W2)
        self.W2 -= learning_rate * dW2
        print(self.W2)
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            loss = mse_loss(y, self.a2)
            self.backward(X, y, learning_rate)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss}")

    def predict(self, X):
        self.forward(X)
        return self.a2


X = np.array([[0, 1, 0, 0, 1, 0, 0, 1, 0],
              [1, 1, 1, 1, 0, 1, 1, 1, 1],
              [0, 1, 0, 0, 0, 0, 0, 1, 0],
              [0, 1, 1, 1, 0, 1, 1, 1, 1],
              [0, 1, 1, 1, 0, 1, 1, 1, 0]])

Y = np.array([[0, 1], [1, 0], [0, 1], [1, 0], [1, 0]])

# 模型
model = NeuralNetwork(input_size=9, hidden_size=5, output_size=2)

# 訓練模型
model.train(X, Y, epochs=50, learning_rate=0.00001)

test = np.array([ [0, 1, 0, 0, 0, 0, 0, 1, 0]])

# 預測结果
predictions = model.predict(test)
print("Predictions:", predictions)