import numpy as np
import random
from PIL import Image

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

# answer = [[0]*10 for i in range(10)]
# for ans in range(10):
#     if(ans == 0):
#         answer[ans][ans] = 1
#     else:
#         answer[ans][ans-1] = 0
#         answer[ans][ans] = 1

answer = [[0]*2 for i in range(2)]
answer[0] = [1,0]
answer[1] = [0,1]

# layer1Weight = createWeight(784,10)
# layer2Weight = createWeight(10,10)
# layer1Bias = createBias(10)
# layer2Bias = createBias(10)
layer1Weight = createWeight(9,3)
layer2Weight = createWeight(3,2)
layer1Bias = createBias(3)
layer2Bias = createBias(2)

#train
# for cla in range(10):
#     for nu in range(101):
#         imagePath = "numbers/{}/{}.png".format(cla,nu)
#         image = Image.open(imagePath)
#         pixel = np.array(image.getdata())
#         input = [0] * 784
#         for pi in range(784):
#             if(pixel[pi][3] != 0):
#                 input[pi] = -1
#             else:
#                 input[pi] = 1
#         HiddenLayer = [0] * 10
#         HiddenLayer = fullyConnected(input,layer1Weight,layer1Bias,10)
#         HiddenLayer = activationFunction(HiddenLayer,"sigmoid")
#         finalLayer = fullyConnected(HiddenLayer,layer2Weight,layer2Bias,10)
#         finalLayer = activationFunction(finalLayer,"softmax")
#         backWardSoftmax(layer2Weight,layer2Bias,learningRate,HiddenLayer,finalLayer,answer[cla])
#         backWardSigmoid(layer1Weight,layer2Weight,layer1Bias,input,HiddenLayer,finalLayer,answer[cla])
        #print(layer1Weight[cla])

#train2
for i in range(120):
    for q in range(2):
        if (i % 2 ==0 and q == 1):
            input = [0] * 9
            for j in range(1, 9, 3):
                input[j] = 1
        elif(i % 2 != 0 and q == 1):
            input = [0] * 9
            for j in range(4,9,3):
                input[j] = 1
        elif(i % 2 != 0 and q == 0):
            input = [0] * 9
            for j in range(9):
                if (j == 0):
                    continue
                elif (j == 4):
                    continue
                else:
                    input[j] = 1
        else:
            input = [0] * 9
            for j in range(9):
                if (j == 4):
                    continue
                else:
                    input[j] = 1
        # if(i == 0 and q >= 10  and q %2==0):
        #     input[0] = 0
        #     input[7] = 0
        HiddenLayer = [0] * 3
        HiddenLayer = fullyConnected(input,layer1Weight,layer1Bias,3)
        #print(HiddenLayer)
        HiddenLayer = activationFunction(HiddenLayer,"sigmoid")
        finalLayer = fullyConnected(HiddenLayer,layer2Weight,layer2Bias,2)
        #print(finalLayer)
        finalLayer = activationFunction(finalLayer,"sigmoid")
        #print("{} {}".format(finalLayer[0], finalLayer[1]))
        #print(crossEntropy(finalLayer,answer[q]))
        #backWardSoftmax(layer2Weight,layer2Bias,learningRate,HiddenLayer,finalLayer,answer[q])
        #backOut(layer2Weight,layer2Bias,learningRate,HiddenLayer,finalLayer,answer[q])
        #backOut2(layer2Weight, layer2Bias, learningRate, HiddenLayer, finalLayer, answer[q])
        backOut3(layer2Weight, layer2Bias, learningRate, HiddenLayer, finalLayer, answer[q])
        #backRelu(layer1Weight,layer2Weight,layer1Bias,input,HiddenLayer,finalLayer,answer[q],learningRate)
        #backWardSigmoid(layer1Weight,layer2Weight,layer1Bias,input,HiddenLayer,finalLayer,answer[q])
        backWardSigmoid2(layer1Weight, layer2Weight, layer1Bias, input, HiddenLayer, finalLayer, answer[q])
        #backWardSigmoid3(layer1Weight, layer2Weight, layer1Bias, input, HiddenLayer, finalLayer, answer[q])
        #print(layer1Weight)


correct = 0
count = 0

#test
# for cla2 in range(1):
#     for nu2 in range(0,101):
#         imagePath2 = "numbers/{}/{}.png".format(cla2, nu2)
#         image2 = Image.open(imagePath2)
#         pixel2 = np.array(image2.getdata())
#         input2 = [0] * 784
#         for pi2 in range(784):
#             if (pixel2[pi2][3] != 0):
#                 input2[pi2] = -1
#             else:
#                 input2[pi2] = 1
#         HiddenLayer2 = [0] * 10
#         HiddenLayer2 = fullyConnected(input2, layer1Weight, layer1Bias, 10)
#         HiddenLayer2 = activationFunction(HiddenLayer2, "sigmoid")
#         finalLayer2 = fullyConnected(HiddenLayer2, layer2Weight, layer2Bias, 10)
#         finalLayer2 = activationFunction(finalLayer2, "softmax")
#         max = -2
#         for ma in range(10):
#             if(finalLayer2[ma] > max):
#                 max = finalLayer2[ma]
#         for la in range(10):
#             if(finalLayer2[la] == max):
#                 finalLayer2[la] = 1
#             else:
#                 finalLayer2[la] = 0
#         #print(finalLayer2)
#         #print(answer[cla2])
#         if(finalLayer2 == answer[cla2]):
#             correct += 1
#             count += 1
#         else:
#             count += 1
#         print("{}: 第{}個 目前正確率：{}".format(cla2,nu2,correct/count))

#test2
for i in range(200):
    for q in range(2):
        if (i % 2 == 0 and q == 1):
            input = [0] * 9
            for j in range(1, 9, 3):
                input[j] = 1
        elif (i % 2 != 0 and q == 1):
            input = [0] * 9
            for j in range(4, 9, 3):
                input[j] = 1
        elif (i % 2 == 0 and q == 0):
            input = [0] * 9
            for j in range(9):
                if (j == 0):
                    continue
                elif (j == 4):
                    continue
                else:
                    input[j] = 1
        elif (i % 3 == 0 and q == 0):
            input = [0] * 9
            for j in range(9):
                if (j == 0):
                    continue
                elif (j == 4):
                    continue
                elif (j == 8):
                    continue
                else:
                    input[j] = 1
        else:
            input = [0] * 9
            for j in range(9):
                if (j == 4):
                    continue
                else:
                    input[j] = 1
        HiddenLayer = [0] * 3
        HiddenLayer = fullyConnected(input,layer1Weight,layer1Bias,3)
        HiddenLayer = activationFunction(HiddenLayer,"sigmoid")

        finalLayer = fullyConnected(HiddenLayer,layer2Weight,layer2Bias,2)
        #finalLayer = activationFunction(finalLayer,"softmax")
        finalLayer = activationFunction(finalLayer, "softmax")
        max = -2
        for ma in range(2):
            if(finalLayer[ma] > max):
                max = finalLayer[ma]
        for la in range(2):
            if(finalLayer[la] == max):
                finalLayer[la] = 1
            else:
                finalLayer[la] = 0
        #print(finalLayer)
        #print(answer[cla2])
        if(finalLayer == answer[q]):
            correct += 1
            count += 1
        else:
            count += 1
        print("{}: 第{}個 目前正確率：{}".format(i,q,correct/count))

