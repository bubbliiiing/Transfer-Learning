## Transfer-Learning-迁移学习思想Keras当中的实现
---

## 目录
1. [所需环境 Environment](#所需环境)
2. [仓库内容 WhatsIn](#仓库内容)
3. [文件下载 Download](#文件下载)

## 所需环境
tensorflow-gpu==1.13.1  
keras==2.1.5  

## 仓库内容
仓库保存了三个例子，分别是：
- [x] VGG16训练分类网络-利用dense全连接完成分类-猫狗大战   
- [x] MobileNet训练分类网络-利用GlobalAveragePooling全局池化完成分类-猫狗大战   
- [x] Resnet50-Segnet训练语义分割网络-区分斑马线  
这三个例子均使用了迁移学习的思想，主干部分提取出来的特征是通用的，所以我们可以不训练主干部分先，因此训练部分分为两步，分别是冻结训练和解冻训练，冻结训练是不训练主干的，解冻训练是训练主干的。 由于训练的特征层变多，解冻后所需显存变大。

## 文件下载：
猫狗数据集：链接: [https://pan.baidu.com/s/1TqmdkJBY49ftg19tRK2Ngg](https://pan.baidu.com/s/1TqmdkJBY49ftg19tRK2Ngg) 提取码: htxf  
斑马线数据集：连接：链接：https://pan.baidu.com/s/1uzwqLaCXcWe06xEXk1ROWw  提取码：pp6w   
