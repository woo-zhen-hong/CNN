import numpy as np
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

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

def softmax(x,sum):
    return np.exp(x) / sum

def softmaxSum(input):
    sum = 0
    for i in range(len(input)):
        sum += np.exp(input[i])
    return sum

def relu(x):
    return np.maximum(0, x)

def crossEntropy(x,y):
    length = len(x)
    lossSum = 0
    for i in range(length):
        lossSum += (y[i] * np.log(x[i]))
        if i == length - 1:
            lossSum *= (-1)
    L = lossSum / length
    return L

def MSE(x,y):
    length = len(y)
    lossSum = 0
    for i in range(length):
        lossSum += (y[i] - x[i]) * (y[i] - x[i])
    return lossSum / length

def sigmoid(r):
    return 1 / (1 + np.exp(-r))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

def createWeight(inputlen, outputlen):
    weightLayer = [[0]*inputlen for i in range(outputlen)] #outputlen列inputlen行
    for i in range(outputlen):
        for j in range(inputlen):
            #weightLayer[i][j] = random.random()
            weightLayer[i][j] = random.uniform(0.1,0.3)
    return weightLayer

def createBias(inputlen):   #輸入神經元個數
    biasLayer = [0]*inputlen
    for i in range(inputlen):
        #biasLayer[i] = random.random()
        biasLayer[i] = random.uniform(0.1,0.3)
    return biasLayer

def calculate(input,weightLayer,biasLayer,outputlen): #outputlen 放第幾個神經元
    sum = 0
    for j in range(len(input)):
        sum += input[j] * weightLayer[outputlen][j]
        #print(input[j])
        #print("{} {} {}".format(input[j] * weightLayer[outputlen][j],outputlen,j))
    sum += biasLayer[outputlen]
    #print(sum)
    return sum

def fullyConnected(input, weightLayer, biasLayer, outputlen): #outputlen 放神經元總數
    output = [0] * outputlen
    for i in range(outputlen):
        output[i] = calculate(input,weightLayer,biasLayer,i)
    return output

def activationFunction(input,activation):
    for i in range(len(input)):
        if activation == "sigmoid":
            input[i] = sigmoid(input[i])
        elif activation == "softmax":
            sum = softmaxSum(input)
            input[i] = softmax(input[i],sum)
    return input

movement = 0
learningRate = 0.5
def backWardSoftmax(weightLayer,biasLayer,learningRate,input,output,ans): #cross entropy
    #hidden layer to final layer
    for i in range(len(output)):
        gradientWeight = ans[i] * (output[i] - 1)
        for j in range(len(input)):
            gradientWeight *= input[j] / len(ans)
            weightLayer[i][j] -= (learningRate * gradientWeight)
            print(weightLayer[i][j])
        sum = calculate(input,weightLayer,biasLayer,i)
        #gradientBias = ((ans[i] * (-1)) * (1 - (output[i] / np.exp(sum)))) / len(ans)
        gradientBias = gradientWeight / len(ans)
        biasLayer[i] -= (learningRate * gradientBias)


def backWardSigmoid(weightLayer1,weightLayer2,biasLayer1,input,output,finalOutput,ans): #cross entropy
    # input layer to hidden layer
    for i in range(len(output)):
        sum = calculate(input,weightLayer1,biasLayer1,i)
        W = 0
        for b in range(len(ans)):
            W += ans[b] * (1 - finalOutput[b]) * weightLayer2[b][i]
        for j in range(len(input)):
            #print(np.exp(-sum))
            gradientWeight = ((input[j] * (-1)) * output[i] * output[i] * np.exp(-sum) * W ) / len(ans)
            #print(((input[j] * (-1)) * output[i] * output[i] * np.exp(-sum) * L * W ))
            #gradientWeight = sigmoid(gradientWeight)
            #print(gradientWeight)
            weightLayer1[i][j] -= (learningRate * gradientWeight)
            #print(weightLayer1[i][j])
        gradientBias = (output[i] * output[i] * np.exp(-sum) * W ) / len(ans)
        biasLayer1[i] -= (learningRate * gradientBias)
    #print(weightLayer1[0][0])

def backWardSigmoid2(weightLayer1,weightLayer2,biasLayer1,input,output,finalOutput,ans): #mean square error
    #input layer to hidden layer
    for i in range(len(output)):
        sum = calculate(input,weightLayer1,biasLayer1,i)
        W = 0
        for b in range(len(ans)):
            W += (ans[b] - finalOutput[b]) * weightLayer2[b][i]
        for j in range(len(input)):
            #print(np.exp(-sum))
            gradientWeight = ((input[j] * (-1)) * output[i] * output[i] * np.exp(-sum) * W ) / len(ans)
            #print(((input[j] * (-1)) * output[i] * output[i] * np.exp(-sum) * L * W ))
            #gradientWeight = sigmoid(gradientWeight)
            #print(gradientWeight)
            weightLayer1[i][j] -= (learningRate * gradientWeight)
            #print(weightLayer1[i][j])
        gradientBias = (output[i] * output[i] * np.exp(-sum) * W ) / len(ans)
        biasLayer1[i] -= (learningRate * gradientBias)
    #print(weightLayer1[0][0])

def backWardSigmoid3(weightLayer1,weightLayer2,biasLayer1,input,output,finalOutput,ans): #cross entropy second version
    # input layer to hidden layer
    for i in range(len(output)):
        sum = calculate(input,weightLayer1,biasLayer1,i)
        W = 0
        for b in range(len(ans)):
            W += (finalOutput[b] - ans[b]) * weightLayer2[b][i]
        for j in range(len(input)):
            #print(np.exp(-sum))
            gradientWeight = ((input[j] * (-1)) * output[i] * output[i] * np.exp(-sum) * W ) / len(ans)
            #print(((input[j] * (-1)) * output[i] * output[i] * np.exp(-sum) * L * W ))
            #gradientWeight = sigmoid(gradientWeight)
            #print(gradientWeight)
            weightLayer1[i][j] -= (learningRate * gradientWeight)
            #print(weightLayer1[i][j])
        gradientBias = (output[i] * output[i] * np.exp(-sum) * W ) / len(ans)
        biasLayer1[i] -= (learningRate * gradientBias)
    #print(weightLayer1[0][0])

def backOut(weightLayer2,biasLayer2,learningRate,input,output,ans): #do-nothing mean square error
    # hidden layer to final layer
    for x in range(len(ans)):
        gradient = 2 * (output[x] - ans[x])
        for y in range(len(input)):
            gradient = gradient * input[y] / len(ans)
            weightLayer2[x][y] -= learningRate * gradient
        biasLayer2[x] -= learningRate * gradient / len(ans)

def backOut2(weightLayer2,biasLayer2,learningRate,input,output,ans): #do-nothing cross-entropy
    # hidden layer to final layer
    for x in range(len(ans)):
        for y in range(len(input)):
            gradient = (-1) * ans[x] / output[x]
            gradient = gradient * input[y] / len(ans)
            weightLayer2[x][y] -= learningRate * gradient
        biasLayer2[x] -= 0

def backOut3(weightLayer2,biasLayer2,learningRate,input,output,ans): #sigmoid mean square error
    # hidden layer to final layer
    for x in range(len(ans)):
        gradient = 2 * (output[x] - ans[x]) * output[x] * (1 - output[x])
        for y in range(len(input)):
            gradient = gradient * input[y] / len(ans)
            weightLayer2[x][y] -= learningRate * gradient
        biasLayer2[x] -= learningRate * gradient / len(ans)

def backRelu(weightLayer1,weightLayer2,biasLayer1,input,output,finalOutput,ans,lr):
    # input layer to hidden layer
    for x in  range(len(output)):
        we = 0
        for q in range(len(ans)):
            we += ans[q] * (1 - finalOutput[q]) * weightLayer2[q][x]
            #print(we)
        for y in range(len(input)):
            #print(we)
            gradientWeight = input[y] * we / len(ans)
            #print(gradientWeight)
            weightLayer1[x][y] -= lr * gradientWeight * 10
            #print(weightLayer1[x][y])
        gradientBias =  we / len(ans)
        biasLayer1[x] -= lr * gradientBias

answer = [[0]*10 for i in range(10)]
for ans in range(10):
    if(ans == 0):
        answer[ans][ans] = 1
    else:
        answer[ans][ans-1] = 0
        answer[ans][ans] = 1

# answer = [[0]*2 for i in range(2)]
# answer[0] = [1,0]
# answer[1] = [0,1]

# layer1Weight = createWeight(784,10)
# layer2Weight = createWeight(10,10)
# layer1Bias = createBias(10)
# layer2Bias = createBias(10)
layer1Weight = createWeight(100,15)
layer2Weight = createWeight(15,10)
layer1Bias = createBias(20)
layer2Bias = createBias(10)

#train
for cla in range(2000):
    for nu in range(1,10):
        for qr in range(1,4):
            img = Image.new('RGB', (10, 10), 'white')
            drawobj = ImageDraw.Draw(img)
            input = to_bin("img/{}/{}.png".format(nu,qr))
            HiddenLayer = [0] * 15
            HiddenLayer = fullyConnected(input,layer1Weight,layer1Bias,15)
            HiddenLayer = activationFunction(HiddenLayer,"sigmoid")
            finalLayer = fullyConnected(HiddenLayer,layer2Weight,layer2Bias,10)
            finalLayer = activationFunction(finalLayer,"sigmoid")
            backOut3(layer2Weight, layer2Bias, learningRate, HiddenLayer, finalLayer, answer[nu])
            backWardSigmoid2(layer1Weight, layer2Weight, layer1Bias, input, HiddenLayer, finalLayer, answer[nu])

correct = 0
count = 0

#test
for cla in range(100):
    for nu in range(1,10):
        for qr in range(1,3):
            img = Image.new('RGB', (10, 10), 'white')
            drawobj = ImageDraw.Draw(img)
            input = to_bin("img/{}/{}.png".format(nu,qr))
            HiddenLayer = [0] * 15
            HiddenLayer = fullyConnected(input,layer1Weight,layer1Bias,15)
            HiddenLayer = activationFunction(HiddenLayer,"sigmoid")
            finalLayer = fullyConnected(HiddenLayer,layer2Weight,layer2Bias,10)
            finalLayer = activationFunction(finalLayer,"sigmoid")
            max = -2
            for ma in range(10):
                if (finalLayer[ma] > max):
                    max = finalLayer[ma]
            for la in range(10):
                if (finalLayer[la] == max):
                    finalLayer[la] = 1
                else:
                    finalLayer[la] = 0
            #print(finalLayer)
            #print(answer[nu])
            if (finalLayer == answer[nu]):
                correct += 1
                count += 1
            else:
                count += 1
            print("{}: 數字{} 第{}個 目前正確率：{}".format(cla, nu, qr, correct / count))
