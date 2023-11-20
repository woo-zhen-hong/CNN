import numpy as np
from PIL import Image


# 定義sigmoid激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 定義sigmoid的導数
def sigmoid_derivative(x):
    return x * (1 - x)

def stable_sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (1 + exp_x)

def stable_sigmoid_der(x):
    return stable_sigmoid(x) * (1 - stable_sigmoid(x))

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
def flaten(dst):
    im = Image.open(dst, 'r')
    width, height = im.size
    pixel = list(im.getdata())
    arr = []
    for i in range(0, len(pixel)):
        arr.append(int(pixel[i][3]))
    return arr

def acc(predicted_output):
    # ans = [0, 0]
    count = 0
    for i in range(0, len(predicted_output)):
        if (i % 2 == 0):
            ans = [1, 0]
        else:
            ans = [0, 1]
        mm = max(predicted_output[i][0], predicted_output[i][1])
        arr = []
        for k in range(0, 2):
            if (predicted_output[i][k] == mm):
                arr.append(1)
            else:
                arr.append(0)
        if (arr == ans):
            count += 1
    # print(str(count) + '/' + str(len(predicted_output)))
    return count / len(predicted_output) * 100

# 初始化神經網路參數
input_size = 784
hidden_size = 150
output_size = 2
learning_rate = 0.006

# 隨機初始化權重
np.random.seed(0)
weights_input_hidden = np.random.uniform(-0.1,0.1,size=(input_size, hidden_size))
weights_hidden_output = np.random.uniform(-0.1,0.1,size=(hidden_size, output_size))
bias_input_hidden = np.random.uniform(-0.1,0.1,size=(200, hidden_size)) #前面的數字是case*num
bias_hidden_output = np.random.uniform(-0.1,0.1,size=(200, output_size)) #前面的數字是case*num

max_weights_input_hidden = np.random.uniform(-0.1,0.1,size=(input_size, hidden_size))
max_weights_hidden_output = np.random.uniform(-0.1,0.1,size=(hidden_size, output_size))
max_bias_input_hidden = np.random.uniform(-0.1,0.1,size=(200, hidden_size)) #前面的數字是case*num
max_bias_hidden_output = np.random.uniform(-0.1,0.1,size=(200, output_size)) #前面的數字是case*num

# 訓練數據
X = [[0] * 784 for i in range(0)]
Y = [[0] * 2 for i in range(0)]
for case in range(101,201):
    for num in range(3,5):
        imagePath = "number/{}/{}.png".format(num,case)
        # image = Image.open(imagePath)
        input = flaten(imagePath)
        X.append(input)
        if (num == 3) :
            Y.append([1,0])
        else:
            Y.append([0,1])

# 訓練神經網路
for epoch in range(2001):
    # 向前傳播1
    input_layer_hidden = np.dot(X, weights_input_hidden)
    input_layer_hidden += bias_input_hidden
    input_layer_hidden = sigmoid(input_layer_hidden)
    hidden_layer_output = np.dot(input_layer_hidden, weights_hidden_output)
    hidden_layer_output += bias_hidden_output
    hidden_layer_output = sigmoid(hidden_layer_output)

    # 向前傳播2
    # input_layer_hidden_1 = np.dot(X, weights_input_hidden)
    # input_layer_hidden_2 = stable_sigmoid(input_layer_hidden_1)
    # hidden_layer_output_1 = np.dot(input_layer_hidden_2, weights_hidden_output)
    # hidden_layer_output_2 = stable_sigmoid(hidden_layer_output_1)

    # 計算損失（MSE）1
    error = Y - hidden_layer_output
    loss = np.mean(error ** 2)

    # 計算損失（MSE）2
    # error = Y - hidden_layer_output_2
    # loss = np.mean(error ** 2)

    # 反向傳播1
    d_output = error * sigmoid_derivative(hidden_layer_output)
    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(input_layer_hidden)

    # 反向傳播2
    # d_output = error * stable_sigmoid_der(hidden_layer_output_1)
    # error_hidden = d_output.dot(weights_hidden_output.T)
    # d_hidden = error_hidden * stable_sigmoid_der(input_layer_hidden_1)

    # 更新權重1
    weights_hidden_output += input_layer_hidden.T.dot(d_output) * learning_rate
    bias_hidden_output += d_output * learning_rate
    X = np.array(X) #transfer list to array
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate
    bias_input_hidden += d_hidden * learning_rate

    # 更新權重2
    # weights_hidden_output += input_layer_hidden_2.T.dot(d_output) * learning_rate
    # X = np.array(X)
    # weights_input_hidden += X.T.dot(d_hidden) * learning_rate
    accuracy = 0
    if(loss <= 0.000003):
        break
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")
    if(epoch == 0):
        max_weights_input_hidden = weights_input_hidden
        max_weights_hidden_output = weights_hidden_output
        max_bias_input_hidden = bias_input_hidden
        max_bias_hidden_output = bias_hidden_output
    elif(acc(hidden_layer_output) >= accuracy):
        max_weights_input_hidden = weights_input_hidden
        max_weights_hidden_output = weights_hidden_output
        max_bias_input_hidden = bias_input_hidden
        max_bias_hidden_output = bias_hidden_output
        accuracy = acc(hidden_layer_output)

# 預測=============================================================
W = [[0] * 784 for i in range(0)]
for case in range(201,301):
    for num in range(3,5):
        imagePath = "number/{}/{}.png".format(num,case)
        # image = Image.open(imagePath)
        input = flaten(imagePath)
        W.append(input)

predict_input_layer_hidden = np.dot(W, max_weights_input_hidden)
predict_input_layer_hidden += max_bias_input_hidden
predict_input_layer_hidden = sigmoid(predict_input_layer_hidden)
predict_hidden_layer_output = np.dot(predict_input_layer_hidden, max_weights_hidden_output)
predict_hidden_layer_output += max_bias_hidden_output
predict_hidden_layer_output = sigmoid(predict_hidden_layer_output)
# predicted_output = sigmoid(np.dot(sigmoid(np.dot(W, weights_input_hidden)), weights_hidden_output))
print("Predicted Output:")
print(predict_hidden_layer_output)
print(str(acc(predict_hidden_layer_output)) + "%")


        
