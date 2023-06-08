import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
from PIL import Image

# EPOCH = 10                #全部data訓練10次
# BATCH_SIZE = 50           #每次訓練隨機丟50張圖像進去
# LR =0.001                 #learning rate
# DOWNLOAD_MNIST = False    #第一次用要先下載data,所以是True
# if_use_gpu = 1            #使用gpu
#
# train_data = torchvision.datasets.MNIST(
#     root='./mnist',
#     train=True,
#     transform=torchvision.transforms.ToTensor(),
#     #把灰階從0~255壓縮到0~1
#     download=DOWNLOAD_MNIST
# )
#
# train_loader = Data.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle=True)
# #shuffle是隨機從data裡讀去資料.
#
# test_data = torchvision.datasets.MNIST(
#     root='./mnist/',
#     train=False,
#     transform=torchvision.transforms.ToTensor(),
#     download=DOWNLOAD_MNIST,
#     )
#
# test_x = Variable(torch.unsqueeze(test_data.data, dim=1).float(), requires_grad=False)
# #requires_grad = False 不參與反向傳播, test data 不用做
#
# test_y = test_data.targets
#
#
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         # 搭建網路的起手式,nn.module是所有網路的基類.
#         # 我們開始定義一系列網路如下：  #train data ＝ (1,28,28)
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(
#                 # convolution2D
#                 in_channels=1,
#                 # input channel(EX:RGB)
#                 out_channels=16,
#                 # output feature maps
#                 kernel_size=5,
#                 # filter大小
#                 stride=1,
#                 # 每次convolution移動多少
#                 padding=2,
#                 # 在圖片旁邊補0
#             ),
#             nn.ReLU(),  # activation function #(16,28,28)
#             nn.MaxPool2d (kernel_size = 2),  # (16,14,14)
#         )
#         # 以上為一層conv + ReLu + maxpool
#
#         # 快速寫法：
#         self.conv2 = nn.Sequential(
#         nn.Conv2d(16, 32, 5, 1, 2),  # (32,14,14)
#         nn.ReLU(),
#         nn.MaxPool2d(2)  # (32,7,7)
#         )
#         self.out = nn.Linear(32 * 7 * 7, 10)  # 10=0~9
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x.view(x.size(0), -1)
#         output = self.out(x)
#         return output
#
# # forward流程:
# # x = x.view(x.size(0), -1) 展平data
#
# cnn = CNN()
# # if if_use_gpu:
# #     cnn = cnn.cuda()
#
# optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
# loss_function = nn.CrossEntropyLoss()
#
# # 優化器使用Adam
# # loss_func 使用CrossEntropy（classification task）
#
# for epoch in range(EPOCH):
#     for step, (x, y) in enumerate(train_loader):
#         b_x = Variable(x, requires_grad=True)
#         b_y = Variable(y, requires_grad=False)
#         #print(x)
#         #print(y)
#         #print(b_x)
#         #print(b_y)
#         # 決定跑幾個epoch,enumerate把load進來的data列出來成（x,y）
#
#         # if if_use_gpu:
#         #     b_x = b_x.cuda()
#         #     b_y = b_y.cuda()
#         # 使用cuda加速
#         output = cnn(b_x)  # 把data丟進網路中
#         loss = loss_function(output, b_y)
#         optimizer.zero_grad()  # 計算loss,初始梯度
#         loss.backward()  # 反向傳播
#         optimizer.step()
#
#     if epoch % 100 == 0:
#         print('Epoch:', epoch, '|step:', step, '|train loss:%.4f' % loss.data)
#
#         # 每100steps輸出一次train loss

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def crossEntropy(x,y):
    length = len(x)
    lossSum = 0
    for i in range(length):
        lossSum += (y[i] * np.log(x[i]))
        if i == length - 1:
            lossSum *= (-1)
    L = lossSum / length
    return L

def sigmoid(r):
    return 1 / 1 + np.exp(-r)

def createWeight(inputlen, outputlen):
    weightLayer = [[0]*inputlen for i in range(outputlen)] #outputlen列inputlen行
    for i in range(outputlen):
        for j in range(inputlen):
            weightLayer[i][j] = random.random()
    return weightLayer

def createBias(inputlen):   #輸入神經元個數
    biasLayer = [0]*inputlen
    for i in range(inputlen):
        biasLayer[i] = random.random()
    return biasLayer

def calculate(input,weightLayer,biasLayer,outputlen): #outputlen 放第幾個神經元
    sum = 0
    for j in range(len(input)):
        sum += input[j] * weightLayer[outputlen][j]
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
            input[i] = sigmoid(input[i])
        elif activation == "softmax":
            input[i] = softmax(input[i])
    return input

movement = 0
learningRate = 0.000001
def backWardSoftmax(weightLayer,biasLayer,learningRate,input,output,ans):
    for i in range(len(output)):
        for j in range(len(input)):
            gradientWeight = ((-ans[i]) * (input[j] *(1 - output[i]))) / len(ans)
            weightLayer[i][j] -= (learningRate * gradientWeight)
        sum = calculate(input,weightLayer,biasLayer,i)
        gradientBias = ((-ans[i]) * (1 - (output[i] / np.exp(sum)))) / len(ans)
        biasLayer[i] -= (learningRate * gradientBias)

def backWardSigmoid(weightLayer1,weightLayer2,biasLayer1,input,output,finalOutput,ans):
    L = 0
    for a in range(len(ans)):
        L += ans[a] * (1 - finalOutput[a])
    for i in range(len(output)):
        sum = calculate(input,weightLayer1,biasLayer1,i)
        W = 0
        for b in range(len(ans)):
            W += weightLayer2[b][i]
        for j in range(len(input)):
            gradientWeight = ((-input[j]) * output[i] * output[i] * np.exp(-sum) * L * W ) / len(ans)
            weightLayer1[i][j] -= (learningRate * gradientWeight)
        gradientBias = (output[i] * output[i] * np.exp(-sum) * L * W ) / len(ans)
        biasLayer1[i] -= (learningRate * gradientBias)

# im = Image.open("numbers/1/1.png")
# width, height = im.size
# pixel = np.array(im.getdata())

# for i in range(0,784):
#     print(pixel[i][3])

answer = [[0]*10 for i in range(10)]
for ans in range(10):
    if(ans == 0):
        answer[ans][ans] = 1
    else:
        answer[ans][ans-1] = 0
        answer[ans][ans] = 1

layer1Weight = createWeight(784,10)
layer2Weight = createWeight(10,10)
layer1Bias = createBias(10)
layer2Bias = createBias(10)

#train
for cla in range(10):
    for nu in range(101):
        imagePath = "numbers/{}/{}.png".format(cla,nu)
        image = Image.open(imagePath)
        pixel = np.array(image.getdata())
        input = [0] * 784
        for pi in range(784):
            if(pixel[pi][3] != 0):
                input[pi] = 1
        HiddenLayer = [0] * 10
        HiddenLayer = fullyConnected(input,layer1Weight,layer1Bias,10)
        HiddenLayer = activationFunction(HiddenLayer,"sigmoid")
        finalLayer = fullyConnected(HiddenLayer,layer2Weight,layer2Bias,10)
        finalLayer = activationFunction(finalLayer,"softmax")
        backWardSoftmax(layer2Weight,layer2Bias,learningRate,HiddenLayer,finalLayer,answer[cla])
        backWardSigmoid(layer1Weight,layer2Weight,layer1Bias,input,HiddenLayer,finalLayer,answer[cla])

correct = 0
count = 0

#test
for cla2 in range(10):
    for nu2 in range(101,201):
        imagePath2 = "numbers/{}/{}.png".format(cla2, nu2)
        image2 = Image.open(imagePath2)
        pixel2 = np.array(image2.getdata())
        input2 = [0] * 784
        for pi2 in range(784):
            if (pixel2[pi2][3] != 0):
                input2[pi2] = 1
        HiddenLayer2 = [0] * 10
        HiddenLayer2 = fullyConnected(input2, layer1Weight, layer1Bias, 10)
        HiddenLayer2 = activationFunction(HiddenLayer2, "sigmoid")
        finalLayer2 = fullyConnected(HiddenLayer2, layer2Weight, layer2Bias, 10)
        finalLayer2 = activationFunction(finalLayer2, "softmax")
        max = -2
        for ma in range(10):
            if(finalLayer2[ma] > max):
                max = finalLayer2[ma]
        for la in range(10):
            if(finalLayer2[la] == max):
                finalLayer2[la] = 1
            else:
                finalLayer2[la] = 0
        if(finalLayer2 == answer[cla2]):
            correct += 1
            count += 1
        else:
            count += 1
        print("{}: 第{}個 目前正確率：{}".format(cla2,nu2,correct/count))