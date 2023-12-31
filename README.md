# doCNN

hi, this is our project for base-CNN.
The IDE is pycharm. Environment is mini-anaconda.

2023,05,16
At this moment, there is only convolution feature separate with neural network.
2023,11,27
At this moment, all of the unit are designed with modular design

The convolution layer is coded by asakeaoyama
The fullyConnected layer and backpropagation layer is coded by MingYao and woo-zhen-hong

以程式碼(PYTHON)實作全連接層和反向傳播。實作專題的過程中使用敏捷式開發法，從最一開始的神經網路概念，一步一步開發到最後實作出整個「CNN卷積神經網路」架構。利用影像處理概念中的灰階化原理將輸入的影像做處理，以提升運算的速度。運用軟體工程的概念將函式模組化以及依據物件導向程式設計概念使用物件及類別。使用矩陣儲存參數並運用矩陣乘法來加速整個計算過程。最後依照資料分析的角度分析出專題「CNN卷積神經網路」的正確率。

# 辨識率
* 執行速度會與執行電腦的執行效能有關，以下數據是使用 MacBook Air (M2 Silicon)
* Execution speed is related to the performance of the executing computer. The following data is using a MacBook Air (M2 Silicon).
# 資料集(Data Set)：Mnist
* 模型(Model): NKNU-CNN
* 訓練內容(Train Content): 手寫數字345(Hand-Written Number 345)
* 步長(Stride): 1
* 濾波器(Kernel/Filter): 1/0
* 隱藏層個數(Hidden Layer): 1
* 神經元個數(Neuron): 400
* 期(Epoch): 2000 
* 訓練資料個數(Train Data): 各100張(100 pics/per number)
* 學習率(Learning Rate): 0.003
* 激活函數(Activation Function): Sigmoid
* 訓練正確率(Train Accuracy): 100%
* 損失(Loss): 0.00149
* 測試資料個數(Test Data): 各100張(100 pics/per number)
* 測試正確率(Test Accuracy): 100%
* 備註(Remark): 有分訓練集與測試集

* 模型(Model): NKNU-CNN
* 訓練內容(Train Content): 手寫數字0-9(Hand-Written Number 0-9)
* 步長(Stride): 1
* 濾波器(Kernel/Filter): 1/0
* 隱藏層個數(Hidden Layer): 1
* 神經元個數(Neuron): 110
* 期(Epoch): 10000 
* 訓練資料個數(Train Data): 各500張(500 pics/per number)
* 學習率(Learning Rate): 0.0001
* 激活函數(Activation Function): Sigmoid
* 訓練正確率(Train Accuracy): 82.42%
* 損失(Loss): 0.03
* 測試資料個數(Test Data): 各500張(500 pics/per number)
* 測試正確率(Test Accuracy): 81.14%
* 時間(Time): 1小時39分鐘18秒
* 備註(Remark): 有分訓練集與測試集

# 資料集(Data Set): CIFAR-10
* 模型(Model): NKNU-CNN
* 訓練內容(Train Content): CIFAR-10 飛機、汽車、鳥(CIFAR-10 Airplane、Car、Bird)
* 步長(Stride): 2
* 濾波器(Kernel/Filter): 1/0
* 隱藏層個數(Hidden Layer): 1
* 神經元個數(Neuron): 110
* 期(Epoch): 2500 
* 訓練資料個數(Train Data): 各150張(150 pics/per number)
* 學習率(Learning Rate): 0.0015
* 激活函數(Activation Function): Sigmoid
* 訓練正確率(Train Accuracy): 91.11%
* 損失(Loss): 0.132
* 測試資料個數(Test Data): 各150張(150 pics/per number)
* 測試正確率(Test Accuracy): 83.33%
* 時間(Time): 2分鐘11秒
* 備註(Remark): 有分訓練集與測試集

* 模型(Model): Pytorch
* 訓練內容(Train Content): CIFAR-10 飛機、汽車、鳥(CIFAR-10 Airplane、Car、Bird)
* 步長(Stride): 2
* 濾波器(Kernel/Filter): 無法設定(Can't Assign)
* 隱藏層個數(Hidden Layer): 1
* 神經元個數(Neuron): 110
* 期(Epoch): 200 
* 訓練資料個數(Train Data): 各300張(300 pics/per number)
* 學習率(Learning Rate): 0.001
* 激活函數(Activation Function): Sigmoid
* 訓練正確率(Train Accuracy): 96.33%
* 損失(Loss): 0.117
* 測試資料個數(Test Data): 各300張(300 pics/per number)
* 測試正確率(Test Accuracy): 84.75%
* 時間(Time): 3分鐘49秒
* 備註(Remark): 有分訓練集與測試集

* 模型(Model): NKNU-CNN
* 訓練內容(Train Content): CIFAR-10 飛機、汽車、鳥、貓、鹿(CIFAR-10 Airplane、Car、Bird、Cat、Deer)
* 步長(Stride): 2
* 濾波器(Kernel/Filter): 1/-1
* 隱藏層個數(Hidden Layer): 1
* 神經元個數(Neuron): 110
* 期(Epoch): 20000 
* 訓練資料個數(Train Data): 各500張(500 pics/per number)
* 學習率(Learning Rate): 0.0003
* 激活函數(Activation Function): Sigmoid
* 訓練正確率(Train Accuracy): 97%
* 損失(Loss): 0.093
* 測試資料個數(Test Data): 各500張(500 pics/per number)
* 測試正確率(Test Accuracy): 87.2%
* 時間(Time): 15分鐘14秒
* 備註(Remark): 有分訓練集與測試集

* 模型(Model): NKNU-CNN
* 訓練內容(Train Content): CIFAR-10 飛機、汽車、鳥、貓、鹿、狗、青蛙、馬、船、卡車(CIFAR-10 Airplane、Car、Bird、Cat、Deer、Dog、Frog、Horse、Boat、Truck)
* 步長(Stride): 2
* 濾波器(Kernel/Filter): 1/0
* 隱藏層個數(Hidden Layer): 1
* 神經元個數(Neuron): 110
* 期(Epoch): 30000 
* 訓練資料個數(Train Data): 各500張(500 pics/per number)
* 學習率(Learning Rate): 0.0003
* 激活函數(Activation Function): Sigmoid
* 訓練正確率(Train Accuracy): 97.1%
* 損失(Loss): 0.057
* 測試資料個數(Test Data): 各500張(500 pics/per number)
* 測試正確率(Test Accuracy): 96.22%
* 時間(Time): 1小時33分鐘43秒
* 備註(Remark): 有分訓練集與測試集