import numpy as np


# 定义sigmoid激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 定义sigmoid的导数
def sigmoid_derivative(x):
    return x * (1 - x)


# 初始化神经网络参数
input_size = 2
hidden_size = 3
output_size = 1
learning_rate = 0.0001

# 随机初始化权重
np.random.seed(0)
weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

# 训练数据
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([[0], [1], [1], [0]])
y = np.array([[1, 0], [0, 1]])

# 训练神经网络
for epoch in range(600):
    # 前向传播
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)

    # 计算损失（MSE）
    error = y - output_layer_output
    loss = np.mean(error ** 2)

    # 反向传播
    d_output = error * sigmoid_derivative(output_layer_output)
    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)

    # 更新权重
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# 预测
predicted_output = sigmoid(np.dot(sigmoid(np.dot(X, weights_input_hidden)), weights_hidden_output))
print("Predicted Output:")
print(predicted_output)