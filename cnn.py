# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 20:17:59 2022

@author: 86178
"""

#基本包导入
import numpy as np
import pandas as pd
import time
import tensorflow as tf
import matplotlib.pyplot as plt

#实时数据增强功能 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#调用显卡内存分配指令需要的包
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
#显卡内存分配指令：按需分配 
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#数据提取
(x_img_train,y_label_train), (x_img_test, y_label_test)=tf.keras.datasets.cifar100.load_data() 
#z-score标准化
mean = np.mean(x_img_train, axis=(0, 1, 2, 3))#四个维度 批数 像素x像素 通道数
std = np.std(x_img_train, axis=(0, 1, 2, 3))

x_img_train = (x_img_train - mean) / (std + 1e-7)#trick 加小数点 避免出现整数 
x_img_test = (x_img_test - mean) / (std + 1e-7) 

#one-hot独热映射
y_label_train = tf.keras.utils.to_categorical(y_label_train, 100)
y_label_test = tf.keras.utils.to_categorical(y_label_test, 100)
model = tf.keras.Sequential()

#conv1
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], input_shape=(32, 32, 3), strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) #批标准化
model.add(tf.keras.layers.Dropout(0.3)) #随机丢弃神经元，防止过拟合
#conv2
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) 
#最大池化1
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))


#conv3
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) 
model.add(tf.keras.layers.Dropout(0.4))
#conv4
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) 
#最大池化2
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))



#conv5
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) 
model.add(tf.keras.layers.Dropout(0.4))
#conv6
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) 
model.add(tf.keras.layers.Dropout(0.4))
#conv7
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) 
#最大池化3
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))


#conv8
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) 
model.add(tf.keras.layers.Dropout(0.4))
#conv9
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) 
model.add(tf.keras.layers.Dropout(0.4))
#conv10
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) 
#最大池化4
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))


#conv11
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) 
model.add(tf.keras.layers.Dropout(0.4))
#conv12
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) 
model.add(tf.keras.layers.Dropout(0.4))
#conv13
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) 
#最大池化5
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

#全连接 MLP三层 
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(rate=0.5))


model.add(tf.keras.layers.Dense(units=512,activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization())


model.add(tf.keras.layers.Dense(units=100))
model.add(tf.keras.layers.Activation('softmax'))
#查看摘要
model.summary()
#超参数
training_epochs = 280
batch_size = 128
learning_rate = 0.1
momentum = 0.9 #SGD加速动量
lr_decay = 1e-6 #学习衰减
lr_drop = 20 #衰减倍数

tf.random.set_seed(777)#可复现
def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))

reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
datagen = ImageDataGenerator(
    featurewise_center=False,  # 布尔值。将输入数据的均值设置为 0，逐特征进行。
    samplewise_center=False,  # 布尔值。将每个样本的均值设置为 0。
    featurewise_std_normalization=False,  # 布尔值。将输入除以数据标准差，逐特征进行。
    samplewise_std_normalization=False,  # 布尔值。将每个输入除以其标准差。
    zca_whitening=False,  # 布尔值。是否应用 ZCA 白化。
    #zca_epsilon  ZCA 白化的 epsilon 值，默认为 1e-6。
    rotation_range=15,  # 整数。随机旋转的度数范围 (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # 布尔值。随机水平翻转。
    vertical_flip=False)  # 布尔值。随机垂直翻转。

datagen.fit(x_img_train)
#配置优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                    decay=1e-6, momentum=momentum, nesterov=True)
#交换熵、自定义优化器，评价标准。
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


t1=time.time()
history=model.fit(datagen.flow(x_img_train, y_label_train,
                                 batch_size=batch_size), epochs=training_epochs, verbose=2, callbacks=[reduce_lr],
                    steps_per_epoch=x_img_train.shape[0] // batch_size, validation_data=(x_img_test, y_label_test))   
t2=time.time()
CNNfit = float(t2-t1)
print("Time taken: {} seconds".format(CNNfit))
print(history)
print(history.history['loss'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['xunlian', 'test'], loc='upper left')
plt.show()
print('begin  test ')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model acc')
plt.ylabel('acc')
plt.xlabel('Epoch')
plt.legend(['xunlian', 'test'], loc='upper left')
plt.show()


scores = model.evaluate(x_img_test, 
                        y_label_test, verbose=0)
scores[1]
