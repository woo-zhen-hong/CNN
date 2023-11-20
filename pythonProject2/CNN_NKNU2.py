from PIL import Image
import numpy as np
import random
from math import ceil

width = 0
height = 0

def img_num(a):
    l = len(a)
    s = ''
    for i in range(0,4-l):
        s += '0'
    s += a
    return s

def importImgToData(path):
    global height, width
    orgImg = Image.open(path)
    pxMatrix = np.asarray(orgImg)

    height = orgImg.height
    width = orgImg.width
    data = [[[0 for rgb in range(3)] for x in range(width)] for y in range(height)]
    #print(height, width)

    for y in range(height):
        for x in range(width):
            r, g, b = pxMatrix[y][x]
            #print(r, g, b)
            data[y][x][0] = r
            data[y][x][1] = g
            data[y][x][2] = b

    return data

def importMNISTToData(path):
    orgImg = Image.open(path)
    pxMatrix = np.asarray(orgImg)
    # print("pic dim :", pxMatrix.ndim)

    height = orgImg.height
    width = orgImg.width
    data = [[[0 for rgb in range(3)] for x in range(width)] for y in range(height)]
    # print(height, width)

    for y in range(height):
        for x in range(width):
            r, g, b, a= pxMatrix[y][x]
            # print(r, g, b)
            data[y][x][0] = a
            # data[y][x][1] = g
            # data[y][x][2] = b
            #data[y][x] = a
    return data


#padding 0 surround the picture data
def padding(convdata, times):
    #print(convdata)
    pdata = [[[0 for i in range(len(convdata[0][0]))]for x in range(len(convdata[0])+(2*times))]for y in range(len(convdata)+(2*times))]
    for i in range(len(convdata[0][0])):
        for y in range(len(convdata)):
            for x in range(len(convdata[0])):
                pdata[y + times][x + times][i] = convdata[y][x][i]
    return pdata

def ranKernal():
     ker = [[0 for x in range(3)]for y in range(3)]
     for i in range(3):
         for j in range(3):
             ker[i][j] = random.randint(0, 1)
     return ker


def firstConv(data, times, strides):
    #psteps = ( n(s-1) - s + f ) / 2
    padding_steps = int((len(data)*(strides - 1) - strides + 3 ) / 2)
    #padding
    pdata = padding(data, padding_steps)
    #calculate the output data size : ((n + 2p - f) / s ) + 1
    sizeConvdataOut = int(len(data) + 2*padding_steps - 3 / strides ) + 1
    #print(sizeConvdataOut)
    #convdataOut[y][x][i = conv times(16)]
    convdata = [[[0 for rgb in range(3)] for x in range(sizeConvdataOut)] for y in range(sizeConvdataOut)]
    convdataOut = [[[0 for i in range(times)]for x in range(sizeConvdataOut)]for y in range(sizeConvdataOut)]
    kernalArr = []

    for i in range(times):
        kernal = ranKernal()
        kernalArr.append(kernal)
        for y in range(1, len(pdata) - 1):
            for x in range(1, len(pdata[y]) - 1):
                convdata[y - 1][x - 1][0] = pdata[y - 1][x - 1][0] * kernal[0][0] + pdata[y - 1][x][0] * kernal[0][
                    1] + pdata[y - 1][x + 1][0] * kernal[0][2] + \
                                            pdata[y][x - 1][0] * kernal[1][0] + pdata[y][x][0] * kernal[1][1] + \
                                            pdata[y][x + 1][0] * kernal[1][2] + \
                                            pdata[y + 1][x - 1][0] * kernal[2][0] + pdata[y + 1][x][0] * kernal[2][
                                                1] + pdata[y + 1][x + 1][0] * kernal[2][2]
                convdata[y - 1][x - 1][1] = pdata[y - 1][x - 1][1] * kernal[0][0] + pdata[y - 1][x][1] * kernal[0][
                    1] + \
                                            pdata[y - 1][x + 1][1] * kernal[0][2] + \
                                            pdata[y][x - 1][1] * kernal[1][0] + pdata[y][x][1] * kernal[1][1] + \
                                            pdata[y][
                                                x + 1][1] * kernal[1][2] + \
                                            pdata[y + 1][x - 1][1] * kernal[2][0] + pdata[y + 1][x][1] * kernal[2][
                                                1] + \
                                            pdata[y + 1][x + 1][1] * kernal[2][2]
                convdata[y - 1][x - 1][2] = pdata[y - 1][x - 1][2] * kernal[0][0] + pdata[y - 1][x][2] * kernal[0][
                    1] + \
                                            pdata[y - 1][x + 1][2] * kernal[0][2] + \
                                            pdata[y][x - 1][2] * kernal[1][0] + pdata[y][x][2] * kernal[1][1] + \
                                            pdata[y][
                                                x + 1][2] * kernal[1][2] + \
                                            pdata[y + 1][x - 1][2] * kernal[2][0] + pdata[y + 1][x][2] * kernal[2][
                                                1] + \
                                            pdata[y + 1][x + 1][2] * kernal[2][2]
                #merge RGB data : convdata[][][] to convdataOut[][][]
                convdataOut[y - 1][x - 1][i] = convdata[y - 1][x - 1][0] + convdata[y - 1][x - 1][1] + \
                                               convdata[y - 1][x - 1][2]
                x += strides - 1
            y += strides - 1
                #test trans to gray
                # gray = (convdata[y - 1][x - 1][0] + convdata[y - 1][x - 1][1] + convdata[y - 1][x - 1][2]) / 3
                # convdataGray[y - 1][x - 1][0] , convdataGray[y - 1][x - 1][1] , convdataGray[y - 1][x - 1][2] = gray, \
                #                                                                                             gray, gray

    #print(convdataOut)
    return convdataOut

def nextConv(data, times, strides):
    # psteps = ( n(s-1) - s + f ) / 2
    padding_steps = int(ceil((len(data) * (strides - 1) - strides + 3) / 2))
    #print(len(data))
    #print(padding_steps)
    # padding
    pdata = padding(data, padding_steps)
    # calculate the output data size : ((n + 2p - f) / s ) + 1
    sizeConvdataOut = int((len(data) + (2 * padding_steps) - 3 )/ strides) + 1
    #pdata = condataout (layer1)
    #pdata formate: [y][x][i = conv times(16)]
    convdataL2 = [[0 for x in range(len(pdata[1]))] for y in range(len(pdata))]
    convdataOut = [[[0 for i in range(times)] for x in range(sizeConvdataOut)] for y in range(sizeConvdataOut)]
    kernalArr = []
    #print(len(convdataL2))

    #compress pdata(y*x*16dim) to condataL2(y*x*1dim)
    for y in range(len(pdata)):
        for x in range(len(pdata[y])):
            convdataL2[y][x] = sum(pdata[y][x])


    #convolution
    for i in range(times):
        kernal = ranKernal()
        kernalArr.append(kernal)
        for y in range(1, len(convdataL2) - strides , strides):
            for x in range(1, len(convdataL2[y]) - strides , strides):
                #print(y,x)
                convdataOut[int((y - 1)/strides)][int((x - 1)/strides)][i] = convdataL2[y - 1][x - 1] * kernal[0][0] + \
                                               convdataL2[y - 1][x] * kernal[0][1] + \
                                               convdataL2[y - 1][x + 1] * kernal[0][2] + \
                                               convdataL2[y][x - 1] * kernal[1][0] + \
                                               convdataL2[y][x] * kernal[1][1] + \
                                               convdataL2[y][x + 1] * kernal[1][2] + \
                                               convdataL2[y + 1][x - 1] * kernal[2][0] + \
                                               convdataL2[y + 1][x] * kernal[2][1] + \
                                               convdataL2[y + 1][x + 1] * kernal[2][2]
    #print(convdataOut)
    return convdataOut

def lastmerge(convdata):
    #[y][x][] to [y][x]
    convedData = [[0 for x in range(len(convdata[0]))] for y in range(len(convdata))]
    for y in range(len(convdata)):
        for x in range(len(convdata[y])):
            convedData[y][x] = sum(convdata[y][x])
            #print(sum(convdata[y][x]))
    #print(convedData)
    return convedData


def pooling(conved):
    #print(len(conved),len(conved[0]))
    #print(conved)
    # conved format: [y][x]
    poolingOut = [[0 for x in range(len(conved[0])-2)] for y in range(len(conved)-2)]

    for y in range(1, len(conved) - 1):
        for x in range(1, len(conved[y]) - 1):
            poolingOut[y-1][x-1] = max(conved[y - 1][x - 1], conved[y - 1][x], conved[y - 1 ][x + 1],
                                       conved[y][x - 1], conved[y][x], conved[y][x + 1],
                                       conved[y + 1][x - 1], conved[y + 1][x], conved[y + 1][x + 1])
    return poolingOut

def flatten2DTo1D(poolingOut):
    flattenData = []
    for y in range(len(poolingOut)):
        for x in range(len(poolingOut[0])):
            flattenData.append(poolingOut[y][x])
    return flattenData

#定義relu
def relu(x):
    return max(0,x)

# 定義sigmoid
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
def flaten(dst):
    im = Image.open(dst, 'r')
    width, height = im.size
    pixel = list(im.getdata())
    arr = []
    for i in range(0, len(pixel)):
        arr.append(int(pixel[i][3]))
    return arr

#製作答案
def makeAnswer(ans_arr,num):
    if (num == 1):
        ans_arr.append([1, 0])
    elif (num == 2):
        ans_arr.append([0, 1])
    return ans_arr

#卷積層
def conv(trainStart,trainEnd,typeStart,typeEnd,path,times,strides):
    img_arr = []
    ans_arr = []
    for case in range(trainStart,trainEnd):
        for num in range(typeStart,typeEnd):
            data = importImgToData(path.format(num,img_num(str(case)))) # square picture only
            convdata = nextConv(data,times ,strides)
            nextconvdata = nextConv(convdata, times, strides)
            output = lastmerge(nextconvdata)
            pooling1 = pooling(output)
            flatten = flatten2DTo1D(pooling1)
            img_arr.append(flatten)
            if (num == 1):
                ans_arr.append([1, 0])
            elif (num == 2):
                ans_arr.append([0, 1])
    return img_arr,ans_arr

# 隨機初始化權重和bias
def createWeights(leftRange,rightRange,input_size,output_size):
    np.random.seed(0)
    weights = np.random.uniform(leftRange, rightRange, size=(input_size, output_size))
    return weights

#第一次建立權重
def firstMakeWeight(leftRange,rightRange,input_size,hidden_size,output_size):
    weights_input_hidden = createWeights(leftRange, rightRange, input_size, hidden_size)
    weights_hidden_output = createWeights(leftRange, rightRange, hidden_size, output_size)
    return weights_input_hidden, weights_hidden_output

#第一次建立bias
def firstMakeBias(leftRange,rightRange,total_size,hidden_size,output_size):
    #total_size是(trainEnd-trainStart)*(typeEnd-typeStart)
    bias_input_hidden = createWeights(leftRange, rightRange, total_size, hidden_size)
    bias_hidden_output = createWeights(leftRange, rightRange, total_size, output_size)
    return bias_input_hidden, bias_hidden_output

#計算正確率
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

#全連接層(1 hidden layer)
def fullyConnected(X,Y,weights_input_hidden,bias_input_hidden,weights_hidden_output,bias_hidden_output,learning_rate,activation):
    # 向前傳播1
    input_layer_hidden = np.dot(X, weights_input_hidden)
    input_layer_hidden += bias_input_hidden
    if(activation == "sigmoid"):
        input_layer_hidden = sigmoid(input_layer_hidden)
    hidden_layer_output = np.dot(input_layer_hidden, weights_hidden_output)
    hidden_layer_output += bias_hidden_output
    if (activation == "sigmoid"):
        hidden_layer_output = sigmoid(hidden_layer_output)
    # 計算損失（MSE）1
    error = Y - hidden_layer_output
    loss = np.mean(error ** 2)
    # 反向傳播1
    if (activation == "sigmoid"):
        d_output = error * sigmoid_derivative(hidden_layer_output)
    error_hidden = d_output.dot(weights_hidden_output.T)
    if (activation == "sigmoid"):
        d_hidden = error_hidden * sigmoid_derivative(input_layer_hidden)
    # 更新權重1
    weights_hidden_output += input_layer_hidden.T.dot(d_output) * learning_rate
    bias_hidden_output += d_output * learning_rate
    X = np.array(X)  # transfer list to array
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate
    bias_input_hidden += d_hidden * learning_rate
    return loss, hidden_layer_output

#測試
def testFullyConnected(X,weights_input_hidden,bias_input_hidden,weights_hidden_output,bias_hidden_output,activation):
    # 向前傳播1
    input_layer_hidden = np.dot(X, weights_input_hidden)
    input_layer_hidden += bias_input_hidden
    if (activation == "sigmoid"):
        input_layer_hidden = sigmoid(input_layer_hidden)
    hidden_layer_output = np.dot(input_layer_hidden, weights_hidden_output)
    hidden_layer_output += bias_hidden_output
    if (activation == "sigmoid"):
        hidden_layer_output = sigmoid(hidden_layer_output)
    return hidden_layer_output

#紀錄最好的權重
def maxWeight(max_weights_input_hidden, max_weights_hidden_output, max_bias_input_hidden, max_bias_hidden_output, epoch,hidden_layer_output,accuracy):
    # global max_weights_input_hidden, max_weights_hidden_output, max_bias_input_hidden, max_bias_hidden_output
    if (epoch == 0 or acc(hidden_layer_output) >= accuracy):
        max_weights_input_hidden, max_weights_hidden_output, max_bias_input_hidden, max_bias_hidden_output = weights_input_hidden, weights_hidden_output, bias_input_hidden, bias_hidden_output
        accuracy = acc(hidden_layer_output)
    return max_weights_input_hidden, max_weights_hidden_output, max_bias_input_hidden, max_bias_hidden_output, accuracy

if __name__ == '__main__':
    img_arr, ans_arr = conv(10,110,1,3,"CIFAR-10/valid/{}/{}.jpg",3,2)

    input_size = 900
    hidden_size = 200
    output_size = 2
    learning_rate = 0.005
    accuracy = 0

    weights_input_hidden, weights_hidden_output = firstMakeWeight(-0.1, 0.1, input_size, hidden_size, output_size)
    bias_input_hidden, bias_hidden_output = firstMakeBias(-0.1, 0.1, 200, hidden_size, output_size)

    max_weights_input_hidden, max_weights_hidden_output = firstMakeWeight(-0.1, 0.1, input_size, hidden_size, output_size)
    max_bias_input_hidden, max_bias_hidden_output = firstMakeBias(-0.1, 0.1, 200, hidden_size, output_size)
    
    for epoch in range(20001):
        loss, hidden_layer_output = fullyConnected(img_arr,ans_arr,weights_input_hidden,bias_input_hidden,weights_hidden_output,bias_hidden_output,learning_rate,"sigmoid")
        max_weights_input_hidden, max_weights_hidden_output, max_bias_input_hidden, max_bias_hidden_output, accuracy = maxWeight(max_weights_input_hidden, max_weights_hidden_output, max_bias_input_hidden, max_bias_hidden_output, epoch,hidden_layer_output,accuracy)
        if(loss <= 0.03):
            break
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    #=======================================================================================以下測試
    img_arr2, ans_arr2 = conv(140, 240, 1, 3, "CIFAR-10/valid/{}/{}.jpg", 3, 2)
    predicted_output = testFullyConnected(img_arr2, max_weights_input_hidden, max_bias_input_hidden, max_weights_hidden_output, max_bias_hidden_output, "sigmoid")
    print("測試結果：")
    print(str(acc(predicted_output)) + "%")