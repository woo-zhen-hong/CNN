import numpy as np
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
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
        if(count < 470):
            arr.append(int(1))
        else:
            arr.append(int(0))
    return arr


def MSE(x,y):   #y=predict x=answer
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
    return sigmoid(x) * ( 1 - sigmoid(x))

def stable_sigmoid_der(x):
    return stable_sigmoid(x) * ( 1 - stable_sigmoid(x))

def createWeight(inputlen, outputlen):
    weightLayer = [[0]*outputlen for i in range(inputlen)] #inputlen列outputlen行
    for i in range(inputlen):
        for j in range(outputlen):
            #weightLayer[i][j] = random.random()
            weightLayer[i][j] = random.uniform(0.1, 0.2)
    return weightLayer

def createBias(inputlen):   #輸入神經元個數
    biasLayer = [0]*inputlen
    for i in range(inputlen):
        #biasLayer[i] = random.random()
        biasLayer[i] = random.uniform(0.1, 0.5)
    return biasLayer

def calculate(input,weightLayer,biasLayer,outputlen): #outputlen 放第幾個神經元
    sum = 0
    for j in range(len(input)):
        sum += input[j] * weightLayer[j][outputlen]
        #print(input[j])
        #print("{} {} {}".format(input[j] * weightLayer[outputlen][j],outputlen,j))
    sum += biasLayer[outputlen]
    return sum

def fullyConnected(input, weightLayer, biasLayer, outputlen): #outputlen 放神經元總數
    output = [0] * outputlen
    for i in range(outputlen):
        output[i] = calculate(input,weightLayer,biasLayer,i)
    return output

def activationFunction(input,activation):
    for i in range(len(input)):
        if activation == "sigmoid":
            input[i] = stable_sigmoid(input[i])
            # input[i] = sigmoid(input[i])
        # elif activation == "softmax":
        #     sum = softmaxSum(input)
        #     input[i] = softmax(input[i],sum)
    return input

learningRate = 0.0001

def backLastLayer(input,weightLayer,biasLayer,outputLayer,ans): #backpropagation on the last layer
    sum = 0
    gradient = [0] * len(input)
    for i in range(len(input)):
        for j in range(len(outputLayer)):
            sum += weightLayer[i][j] * sigmoid_der(outputLayer[j]) * (outputLayer[j] - ans[j])
        gradient[i] = sum
    for a in range(len(input)):
        for b in range(len(outputLayer)):
            weightLayer[a][b] -= gradient[a] * learningRate
        #biasLayer[a] -= gradient[a] * learningRate
    return gradient

def backHiddenLayer(input,weightLayer,biasLayer,outputLayer,derivation): #backpropagation on the hidden layer
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
        #biasLayer[a] -= gradient[a] * learningRate
    return gradient

def flaten(pixel):
    arr = []
    for i in range(0, len(pixel)):
        count = 0
        for rgb in range(0, 3):
            count += pixel[i][rgb]
        if(count < 470):
            arr.append(int(1))
        else:
            arr.append(int(0))
    return arr

def calAUC(prob,labels):
    f = list(zip(prob,labels))
    rank = [values2 for values1,values2 in sorted(f,key=lambda x:x[0])]
    rankList = [i+1 for i in range(len(rank)) if rank[i]==1]
    posNum = 0
    negNum = 0
    for i in range(len(labels)):
        if (labels[i] == 1):
          posNum+=1
        else:
          negNum+=1
    auc = 0
    auc = (sum(rankList) - (posNum*(posNum+1))/2)/(posNum*negNum)
    print(auc)
    return auc

#the ansewer of 0/1
# answer = [[0]*2 for i in range(2)]
# answer[0] = [1,0]
# answer[1] = [0,1]
answer = [0,1,0,0,0,0,0,0,0,0]

# set the weightLayer and biasLayer
layer1Weight = createWeight(100,15)
layer2Weight = createWeight(15,15)
layer3Weight = createWeight(15,10)
layer1Bias = createBias(15)
layer2Bias = createBias(15)
layer3Bias = createBias(10)

# train0 = [1, 1, 1, 1, 0, 1, 1, 1, 1]
# train1 = [0, 1, 0, 0, 1, 0, 0, 1, 0]

# train2 = [1, 1, 1, 0, 0, 1, 1, 1, 1]
# train3 = [0, 0, 0, 0, 1, 0, 0, 1, 0]

# test0 = [0, 1, 1, 1, 0, 1, 1, 1, 1]
# test1 = [0, 1, 0, 0, 1, 0, 0, 1, 1]

#derivation
gradientHiddenLayer3 = [0] * 15
gradientHiddenLayer2 = [0] * 15
gradientHiddenLayer1 = [0] * 100

#train
for epoch in range(1000):
    for number in range(1,2):
        for case in range(1,11):
            imagePath = "1to10/{}/{}_{}.png".format(number, number, case)
            image = Image.open(imagePath)
            input = np.array(image.getdata(), dtype=float)
            input = flaten(input)
            HiddenLayer1 = fullyConnected(input,layer1Weight,layer1Bias,15)
            HiddenLayer1 = activationFunction(HiddenLayer1,"sigmoid")
            HiddenLayer2 = fullyConnected(HiddenLayer1, layer2Weight, layer2Bias, 15)
            HiddenLayer2 = activationFunction(HiddenLayer2, "sigmoid")
            finalLayer = fullyConnected(HiddenLayer2,layer3Weight,layer3Bias,10)
            finalLayer = activationFunction(finalLayer,"sigmoid")
            #print(finalLayer)
            #print(MSE(finalLayer,answer[number % 2]))
            gradientHiddenLayer3 = backLastLayer(HiddenLayer2,layer3Weight,layer3Bias,finalLayer,answer)
            gradientHiddenLayer2 = backHiddenLayer(HiddenLayer1,layer2Weight,layer2Bias,HiddenLayer2,gradientHiddenLayer3)
            gradientHiddenLayer1 = backHiddenLayer(input, layer1Weight, layer1Bias, HiddenLayer1,gradientHiddenLayer2)

#train 3x3
# for epoch in range(40):
#     for number in range(4):
#         if (epoch % 4 == 0):
#             input = train0
#         elif (epoch % 4 == 1) :
#             input = train1
#         elif (epoch % 4 == 2):
#             input = train2
#         elif (epoch % 4 == 3):
#             input = train3
#         #HiddenLayer = [0] * 5
#         HiddenLayer1 = fullyConnected(input,layer1Weight,layer1Bias,50)
#         HiddenLayer1 = activationFunction(HiddenLayer1,"sigmoid")
#         HiddenLayer2 = fullyConnected(HiddenLayer1, layer2Weight, layer2Bias, 50)
#         HiddenLayer2 = activationFunction(HiddenLayer2, "sigmoid")
#         finalLayer = fullyConnected(HiddenLayer2,layer3Weight,layer3Bias,2)
#         finalLayer = activationFunction(finalLayer,"sigmoid")
#         #print(finalLayer)
#         #print(MSE(finalLayer,answer[number % 2]))
#         gradientHiddenLayer3 = backLastLayer(HiddenLayer2,layer3Weight,layer3Bias,finalLayer,answer[epoch % 2])
#         gradientHiddenLayer2 = backHiddenLayer(HiddenLayer1,layer2Weight,layer2Bias,HiddenLayer2,gradientHiddenLayer3)
#         gradientHiddenLayer1 = backHiddenLayer(input, layer1Weight, layer1Bias, HiddenLayer1,gradientHiddenLayer2)

correct = 0
count = 0
auc = 0.0

#test
for epoch in range(4):
    for number in range(1,2):
        for case in range(8,11):
            #imagePath = "1to10/{}/{}_{}.png".format(number, number, case)
            imagePath = "1to10/1/1_test.png"
            image = Image.open(imagePath)
            input = np.array(image.getdata(), dtype=float)
            input = flaten(input)
            HiddenLayer1 = fullyConnected(input,layer1Weight,layer1Bias,15)
            HiddenLayer1 = activationFunction(HiddenLayer1,"sigmoid")
            HiddenLayer2 = fullyConnected(HiddenLayer1, layer2Weight, layer2Bias, 15)
            HiddenLayer2 = activationFunction(HiddenLayer2, "sigmoid")
            finalLayer = fullyConnected(HiddenLayer2,layer3Weight,layer3Bias,10)
            #print(finalLayer)
            finalLayer = activationFunction(finalLayer,"sigmoid")
            print(finalLayer)
            max = -999
            for ma in range(10):
                if (finalLayer[ma] > max):
                    max = finalLayer[ma]
            for la in range(10):
                if (finalLayer[la] == max):
                    finalLayer[la] = 1
                else:
                    finalLayer[la] = 0
            #print("預測： {}".format(finalLayer))
            #print("答案： {}".format(answer[number % 2]))
            if (finalLayer == answer):
                correct += 1
                count += 1
            else:
                count += 1
            #print("{}: 數字{} 目前正確率：{}".format(epoch, number, correct / count))

#test 3x3
# for epoch in range(40):
#     for number in range(4):
#         if (number / 2 == 0):
#             input = test0
#         else:
#             input = test1
#         #HiddenLayer = [0] * 5
#         HiddenLayer1 = fullyConnected(input,layer1Weight,layer1Bias,50)
#         HiddenLayer1 = activationFunction(HiddenLayer1,"sigmoid")
#         HiddenLayer2 = fullyConnected(HiddenLayer1, layer2Weight, layer2Bias, 50)
#         HiddenLayer2 = activationFunction(HiddenLayer2, "sigmoid")
#         finalLayer = fullyConnected(HiddenLayer2,layer3Weight,layer3Bias,2)
#
#         #print(finalLayer)
#         finalLayer = activationFunction(finalLayer,"sigmoid")
#         #print(finalLayer)
#         #numAuc = calAUC(finalLayer,answer[nu])
#         #auc += numAuc
#         max = -2
#         for ma in range(2):
#             if (finalLayer[ma] > max):
#                 max = finalLayer[ma]
#         for la in range(2):
#             if (finalLayer[la] == max):
#                 finalLayer[la] = 1
#             else:
#                 finalLayer[la] = 0
#         #print("預測： {}".format(finalLayer))
#         #print("答案： {}".format(answer[number % 2]))
#         if (finalLayer == answer[number % 2]):
#             correct += 1
#             count += 1
#         else:
#             count += 1
#         #print("{}: 數字{} 目前正確率：{}".format(epoch, number % 2, correct / count))

#print(auc/4500)