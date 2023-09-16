import numpy as np
import random
from PIL import Image
import torch

def to_bin(dst):
    im = Image.open(dst, 'r')
    width, height = im.size
    pixel = list(im.getdata())
    arr = []
    for i in range(0, len(pixel)):
        count = 0
        for rgb in range(0, 3):
            count += pixel[i][rgb]
        if (count < 470):
            arr.append(int(1))
        else:
            arr.append(int(0))
    return arr


def MSE(x, y):  # y=predict x=answer
    length = len(y)
    lossSum = 0
    for i in range(length):
        lossSum += (y[i] - x[i]) * (y[i] - x[i])
    return lossSum / length


def sigmoid(r):
    return 1 / (1 + np.exp(-r))


def stable_sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (1 + exp_x)


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


def stable_sigmoid_der(x):
    return stable_sigmoid(x) * (1 - stable_sigmoid(x))


def createWeight(inputlen, outputlen):
    weightLayer = [[0] * outputlen for i in range(inputlen)]  # inputlen列outputlen行
    for i in range(inputlen):
        for j in range(outputlen):
            # weightLayer[i][j] = random.random()
            weightLayer[i][j] = random.uniform(0.1, 0.9)
    return weightLayer


def createBias(inputlen):  # 輸入神經元個數
    biasLayer = [0] * inputlen
    for i in range(inputlen):
        # biasLayer[i] = random.random()
        biasLayer[i] = random.uniform(0.1, 0.9)
    return biasLayer


def calculate(input, weightLayer, biasLayer, outputlen):  # outputlen 放第幾個神經元
    sum = 0
    for j in range(len(input)):
        sum += input[j] * weightLayer[j][outputlen]
        # print(input[j])
        # print("{} {} {}".format(input[j] * weightLayer[outputlen][j],outputlen,j))
    sum += biasLayer[outputlen]
    return sum


def fullyConnected(input, weightLayer, biasLayer, outputlen):  # outputlen 放神經元總數
    output = [0] * outputlen
    for i in range(outputlen):
        output[i] = calculate(input, weightLayer, biasLayer, i)
    return output


def activationFunction(input, activation):
    for i in range(len(input)):
        if activation == "sigmoid":
            input[i] = stable_sigmoid(input[i])
            # input[i] = sigmoid(input[i])
        # elif activation == "softmax":
        #     sum = softmaxSum(input)
        #     input[i] = softmax(input[i],sum)
    return input

learningRate = 0.0001

def backLastLayer(input, weightLayer, biasLayer, outputLayer, ans):  # backpropagation on the last layer
    sum = 0
    gradient = [0] * len(input)
    for i in range(len(input)):
        for j in range(len(outputLayer)):
            sum += weightLayer[i][j] * sigmoid_der(outputLayer[j]) * (outputLayer[j] - ans[j])
        gradient[i] = sum
    for a in range(len(input)):
        for b in range(len(outputLayer)):
            weightLayer[a][b] -= gradient[a] * learningRate

        # biasLayer[a] -= gradient[a] * learningRate
    return gradient


def backHiddenLayer(input, weightLayer, biasLayer, outputLayer, derivation):  # backpropagation on the hidden layer
    # derivation is from the previous layer
    # derivation 下一層的微分結果
    sum = 0
    gradient = [0] * len(input)
    for i in range(len(input)):
        for j in range(len(outputLayer)):
            sum += weightLayer[i][j] * sigmoid_der(outputLayer[j]) * derivation[j]
        gradient[i] = sum
    for a in range(len(input)):
        for b in range(len(outputLayer)):
            weightLayer[a][b] -= gradient[a] * learningRate
        # biasLayer[a] -= gradient[a] * learningRate
    return gradient


# =============================================================================
# def flaten(pixel):
#     arr = []
#     for i in range(0, len(pixel)):
#         count = 0
#         for rgb in range(0, 3):
#             count += pixel[i][rgb]
#         if (count < 470):
#             arr.append(int(1))
#         else:
#             arr.append(int(0))
#     return arr
# =============================================================================
def flaten(pixel):
    arr = []
    for i in range(0, len(pixel)):
        if (pixel[i] < 100):
            arr.append(int(1))
        else:
            arr.append(int(0))
    return arr

def calAUC(prob, labels):
    f = list(zip(prob, labels))
    rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]
    rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]
    posNum = 0
    negNum = 0
    for i in range(len(labels)):
        if (labels[i] == 1):
            posNum += 1
        else:
            negNum += 1
    auc = 0
    auc = (sum(rankList) - (posNum * (posNum + 1)) / 2) / (posNum * negNum)
    print(auc)
    return auc

if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print (x)
else:
    print ("MPS device not found.")

# make answer
# def answer(i):
#     arr = []
#     for j in range(1, 3):
#         if (i == j):
#             arr.append(1)
#         else:
#             arr.append(0)
#     return arr
answer = [[0]*2 for i in range(3)]
answer[1] = [1,0]
answer[2] = [0,1]

# set the weightLayer and biasLayer for HiddenLayer 1
layer1Weight = createWeight(400, 30)
layer2Weight = createWeight(30, 2)
layer1Bias = createBias(30)
layer2Bias = createBias(2)

# set the weightLayer and biasLayer for HiddenLayer 2
# layer1Weight = createWeight(400, 15)
# layer2Weight = createWeight(15, 15)
# layer3Weight = createWeight(15, 2)
# layer1Bias = createBias(15)
# layer2Bias = createBias(15)
# layer3Bias = createBias(10)

# derivation for HiddenLayer 1
gradientHiddenLayer2 = [0] * 30
gradientHiddenLayer1 = [0] * 400

# derivation for HiddenLayer 2
# gradientHiddenLayer3 = [0] * 15
# gradientHiddenLayer2 = [0] * 15
# gradientHiddenLayer1 = [0] * 400

# train for HiddenLayer 1
for epoch in range(600):
    for number in range(1, 3):
        for case in range(1, 101):
            imagePath = "1to10-20x20/{}/{}_{}.png".format(number, number, case)
            image = Image.open(imagePath)
            input = np.array(image.getdata(), dtype=float)
            input = flaten(input)
            HiddenLayer1 = fullyConnected(input, layer1Weight, layer1Bias, 30)
            HiddenLayer1 = activationFunction(HiddenLayer1, "sigmoid")
            finalLayer = fullyConnected(HiddenLayer1, layer2Weight, layer2Bias, 2)
            finalLayer = activationFunction(finalLayer, "sigmoid")
            # print(MSE(finalLayer,answer[number % 2]))
            gradientHiddenLayer2 = backLastLayer(HiddenLayer1, layer2Weight, layer2Bias, finalLayer, answer[number])
            gradientHiddenLayer1 = backHiddenLayer(input, layer1Weight, layer1Bias, HiddenLayer1, gradientHiddenLayer2)

# train for HiddenLayer 2
# for epoch in range(300):
#     print(epoch)
#     for number in range(1, 3):
#         for case in range(1, 101):
#             imagePath = "1to10-20x20/{}/{}_{}.png".format(number, number, case)
#             image = Image.open(imagePath)
#             input = np.array(image.getdata(), dtype=float)
#             input = flaten(input)
#             HiddenLayer1 = fullyConnected(input, layer1Weight, layer1Bias, 15)
#             HiddenLayer1 = activationFunction(HiddenLayer1, "sigmoid")
#             HiddenLayer2 = fullyConnected(HiddenLayer1, layer2Weight, layer2Bias, 15)
#             HiddenLayer2 = activationFunction(HiddenLayer2, "sigmoid")
#             finalLayer = fullyConnected(HiddenLayer2, layer3Weight, layer3Bias, 2)
#             finalLayer = activationFunction(finalLayer, "sigmoid")
#             # print(MSE(finalLayer,answer[number % 2]))
#             gradientHiddenLayer3 = backLastLayer(HiddenLayer2, layer3Weight, layer3Bias, finalLayer, answer(number))
#             gradientHiddenLayer2 = backHiddenLayer(HiddenLayer1, layer2Weight, layer2Bias, HiddenLayer2,
#                                                    gradientHiddenLayer3)
#             gradientHiddenLayer1 = backHiddenLayer(input, layer1Weight, layer1Bias, HiddenLayer1, gradientHiddenLayer2)

correct = 0
count = 0
auc = 0.0

# test for HiddenLayer 1 and for more than one test
for i in range(101,111):
    if (i % 2 != 0) :
        number = 1
    else:
        number = 2
    imagePath = "1to10-20x20/{}/{}_{}.png".format(number, number, i)
    image = Image.open(imagePath)
    input = np.array(image.getdata(), dtype=float)
    input = flaten(input)
    HiddenLayer1 = fullyConnected(input, layer1Weight, layer1Bias, 30)
    HiddenLayer1 = activationFunction(HiddenLayer1, "sigmoid")
    finalLayer = fullyConnected(HiddenLayer1, layer2Weight, layer2Bias, 2)
    finalLayer = activationFunction(finalLayer, "sigmoid")
    # print('------------------------------')
    print('現在是數字' + str(number))
    print(finalLayer)

# test for HiddenLayer 1 and for 1 test
# imagePath = "1to10-20x20/2/2_101.png"
# image = Image.open(imagePath)
# input = np.array(image.getdata(), dtype=float)
# input = flaten(input)
# HiddenLayer1 = fullyConnected(input, layer1Weight, layer1Bias, 15)
# HiddenLayer1 = activationFunction(HiddenLayer1, "sigmoid")
# finalLayer = fullyConnected(HiddenLayer1, layer2Weight, layer2Bias, 2)
# finalLayer = activationFunction(finalLayer, "sigmoid")
# print('------------------------------')
# print(finalLayer)

# test for HiddenLayer 2
# imagePath = "1to10/2/2_100.png"
# image = Image.open(imagePath)
# input = np.array(image.getdata(), dtype=float)
# input = flaten(input)
# HiddenLayer1 = fullyConnected(input, layer1Weight, layer1Bias, 15)
# HiddenLayer1 = activationFunction(HiddenLayer1, "sigmoid")
# HiddenLayer2 = fullyConnected(HiddenLayer1, layer2Weight, layer2Bias, 15)
# HiddenLayer2 = activationFunction(HiddenLayer2, "sigmoid")
# finalLayer = fullyConnected(HiddenLayer2, layer3Weight, layer3Bias, 2)
# finalLayer = activationFunction(finalLayer, "sigmoid")
# print('------------------------------')
# print(finalLayer)