import numpy as np
from PIL import Image


# 定義sigmoid激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 定義sigmoid的導数
def sigmoid_derivative(x):
    return x * (1 - x)

# 定義損失函數
def MSE(answer, predict):  # y=predict x=answer
    length = len(predict)
    lossSum = 0
    for i in range(length):
        lossSum += (predict - answer) ** 2
    lossSum *= (1/2)
    print('*******************')
    print('現在是Loss:' + str(lossSum))
    return lossSum / length

# 將RGB二值化
def flaten(pixel):
    arr = []
    for i in range(len(pixel)):
        if (pixel[i] < 100):
            arr.append(int(1))
        else:
            arr.append(int(0))
    return arr

# 初始化神經網路參數
input_size = 784
hidden_size = 10
output_size = 2
learning_rate = 0.0000001

# 隨機初始化權重
np.random.seed(0)
weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

# 訓練數據
X = [[0] * 784 for i in range(200)]
Y = [[0] * 2 for i in range(200)]
for i in range(3,5):
    for j in range(101):
        imagePath = "number/{}/{}.png".format(i,i,j)
        image = Image.open(imagePath)
        input = np.array(image.getdata(), dtype=float)
        print(input)
        input = flaten(input)
        X.append(input)
        if (i == 3) :
            Y.append([1,0])
        else:
            Y.append([0, 1])

# 訓練神經網路
for epoch in range(1001):
    # 向前傳播
    input_layer_hidden = np.dot(X, weights_input_hidden)
    input_layer_hidden = sigmoid(input_layer_hidden)
    hidden_layer_output = np.dot(input_layer_hidden, weights_hidden_output)
    hidden_layer_output = sigmoid(hidden_layer_output)

    # 計算損失（MSE）
    error = Y - hidden_layer_output
    loss = np.mean(error ** 2)

    # 反向傳播
    d_output = error * sigmoid_derivative(hidden_layer_output)
    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(input_layer_hidden)

    # 更新權重
    weights_hidden_output += input_layer_hidden.T.dot(d_output) * learning_rate
    X = np.array(X)
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# 預測
predicted_output = sigmoid(np.dot(sigmoid(np.dot(X, weights_input_hidden)), weights_hidden_output))
print("Predicted Output:")
print(predicted_output)