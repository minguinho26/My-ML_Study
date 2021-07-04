import tensorflow as tf
import math
import numpy as np
from tqdm import tqdm
import os
import cv2
import copy
from glob import glob

# ResNet과 PlainNet(Skip Connection 적용 안한거) 둘다 구현

# 입력 이미지 크기 : 224*224
# 최적화 : weight decay of 0.0001 and a momentum of 0.9(tf.keras.optimizers.SGD)
# 가중치 : 미니배치 SGD(배치 크기 : 256)
# 학습률 : 로스가 정체되면 10%로 줄임
# 손실 함수 : cross_entropy(tf.nn.softmax_cross_entropy_with_logits)
class ResNet34(tf.keras.Model):
    def __init__(self):
        super(ResNet34, self).__init__(name='ResNet34')
        self.Optimizers = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

        # 레이어(논문에 나온 구조 그대로 사용)
        self.BN = tf.keras.layers.BatchNormalization() # 모든 컨볼루션 연산 이후 거쳐야한다.
        self.ReLU = tf.keras.layers.ReLU() # Conv -> BN -> ReLU

        regularizer = tf.keras.regularizers.l2(0.0005) # weight decay

        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev= math.sqrt(2/(64*7^2)) , seed=None) # sqrt(2/(레이어 필터 개수 * 필터 크기의 제곱))
        self.conv1 = tf.keras.layers.Conv2D(64, 7, strides=(2, 2),padding="same", input_shape=(1, 224, 224, 3)) # downsampling directly by convolutional layers that have a stride of 2
        self.maxPooling = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))


        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev= math.sqrt(2/(64*3^2)) , seed=None)
        self.conv2_1_1 = tf.keras.layers.Conv2D(64, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv2_1_2 = tf.keras.layers.Conv2D(64, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv2_2_1 = tf.keras.layers.Conv2D(64, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv2_2_2 = tf.keras.layers.Conv2D(64, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv2_3_1 = tf.keras.layers.Conv2D(64, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv2_3_2 = tf.keras.layers.Conv2D(64, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")

        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev= math.sqrt(2/(128*3^2)) , seed=None)
        self.conv3_1_1 = tf.keras.layers.Conv2D(128, 3, strides=(2, 2), kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv3_1_2 = tf.keras.layers.Conv2D(128, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv3_2_1 = tf.keras.layers.Conv2D(128, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv3_2_2 = tf.keras.layers.Conv2D(128, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv3_3_1 = tf.keras.layers.Conv2D(128, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv3_3_2 = tf.keras.layers.Conv2D(128, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv3_4_1 = tf.keras.layers.Conv2D(128, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv3_4_2 = tf.keras.layers.Conv2D(128, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")

        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev= math.sqrt(2/(256*3^2)) , seed=None)
        self.conv4_1_1 = tf.keras.layers.Conv2D(256, 3, strides=(2, 2), kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv4_1_2 = tf.keras.layers.Conv2D(256, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv4_2_1 = tf.keras.layers.Conv2D(256, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv4_2_2 = tf.keras.layers.Conv2D(256, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv4_3_1 = tf.keras.layers.Conv2D(256, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv4_3_2 = tf.keras.layers.Conv2D(256, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv4_4_1 = tf.keras.layers.Conv2D(256, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv4_4_2 = tf.keras.layers.Conv2D(256, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv4_5_1 = tf.keras.layers.Conv2D(256, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv4_5_2 = tf.keras.layers.Conv2D(256, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv4_6_1 = tf.keras.layers.Conv2D(256, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv4_6_2 = tf.keras.layers.Conv2D(256, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")

        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev= math.sqrt(2/(512*3^2)) , seed=None)
        self.conv5_1_1 = tf.keras.layers.Conv2D(512, 3, strides=(2, 2), kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv5_1_2 = tf.keras.layers.Conv2D(512, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv5_2_1 = tf.keras.layers.Conv2D(512, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv5_2_2 = tf.keras.layers.Conv2D(512, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv5_3_1 = tf.keras.layers.Conv2D(512, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv5_3_2 = tf.keras.layers.Conv2D(512, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")

        self.average_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.fcn = tf.keras.layers.Dense(275, activation='softmax')
    
    def calc_Conv(self, input, layer) :
        output = layer(input)
        output = self.BN(output)
        output = self.ReLU(output)

        return output

    # Skip Connection(element-wise addition)
    def Skip_connection(self, input, layer_group, pooling_size):
        conv_1 = layer_group[0]
        conv_2 = layer_group[1]

        ori_input = 0
    
        if pooling_size == 2 :
            # 크기도 2배로 줄이면서 채널 숫자도 바꿔야 한다
            ori_input = tf.identity(input)
            #ori_input을 Ws와 곱해 output과 같은 크기가 되도록 해야한다.
            
            size_toConvert = tf.shape(ori_input).numpy()
            new_channel = size_toConvert[-1] * 2
            
            ori_input = tf.keras.layers.Conv2D(new_channel, 1, strides=(2, 2), padding="same")(ori_input)
        else :
            ori_input = tf.identity(input)
        
        output = self.calc_Conv(input, conv_1)
        output = self.calc_Conv(output, conv_2)

        return tf.math.add(output, ori_input)

    def call(self, image): # (1, 224, 224, 3) tensor를 받는다.

        output = self.calc_Conv(image, self.conv1)
        output = self.maxPooling(output)

        # Skip Connection 시작
        output = self.Skip_connection(output, [self.conv2_1_1, self.conv2_1_2], 1)
        output = self.Skip_connection(output, [self.conv2_2_1, self.conv2_2_2], 1)
        output = self.Skip_connection(output, [self.conv2_3_1, self.conv2_3_2], 1)

        output = self.Skip_connection(output, [self.conv3_1_1, self.conv3_1_2], 2)
        output = self.Skip_connection(output, [self.conv3_2_1, self.conv3_2_2], 1)
        output = self.Skip_connection(output, [self.conv3_3_1, self.conv3_3_2], 1)
        output = self.Skip_connection(output, [self.conv3_4_1, self.conv3_4_2], 1)

        output = self.Skip_connection(output, [self.conv4_1_1, self.conv4_1_2], 2)
        output = self.Skip_connection(output, [self.conv4_2_1, self.conv4_2_2], 1)
        output = self.Skip_connection(output, [self.conv4_3_1, self.conv4_3_2], 1)
        output = self.Skip_connection(output, [self.conv4_4_1, self.conv4_4_2], 1)
        output = self.Skip_connection(output, [self.conv4_5_1, self.conv4_5_2], 1)
        output = self.Skip_connection(output, [self.conv4_6_1, self.conv4_6_2], 1)

        output = self.Skip_connection(output, [self.conv5_1_1, self.conv5_1_2], 2)
        output = self.Skip_connection(output, [self.conv5_2_1, self.conv5_2_2], 1)
        output = self.Skip_connection(output, [self.conv5_3_1, self.conv5_3_2], 1)

        output = self.average_pool(output)
        output = self.flatten(output)

        output = self.fcn(output)

        return output

    # 각 입력 데이터에 대한 loss를 구하고 grad를 구한다.
    def Get_Gradient(self, image, label) :
        with tf.GradientTape() as tape:
            tape.watch(self.conv1.variables)
            tape.watch(self.conv2_1_1.variables)
            tape.watch(self.conv2_1_2.variables)
            tape.watch(self.conv2_2_1.variables)
            tape.watch(self.conv2_2_2.variables)
            tape.watch(self.conv2_3_1.variables)
            tape.watch(self.conv2_3_2.variables)
            tape.watch(self.conv3_1_1.variables)
            tape.watch(self.conv3_1_2.variables)
            tape.watch(self.conv3_2_1.variables)
            tape.watch(self.conv3_2_2.variables)
            tape.watch(self.conv3_3_1.variables)
            tape.watch(self.conv3_3_2.variables)
            tape.watch(self.conv3_4_1.variables)
            tape.watch(self.conv3_4_2.variables)
            tape.watch(self.conv4_1_1.variables)
            tape.watch(self.conv4_1_2.variables)
            tape.watch(self.conv4_2_1.variables)
            tape.watch(self.conv4_2_2.variables)
            tape.watch(self.conv4_3_1.variables)
            tape.watch(self.conv4_3_2.variables)
            tape.watch(self.conv4_4_1.variables)
            tape.watch(self.conv4_4_2.variables)
            tape.watch(self.conv4_5_1.variables)
            tape.watch(self.conv4_5_2.variables)
            tape.watch(self.conv4_6_1.variables)
            tape.watch(self.conv4_6_2.variables)
            tape.watch(self.conv5_1_1.variables)
            tape.watch(self.conv5_1_2.variables)
            tape.watch(self.conv5_2_1.variables)
            tape.watch(self.conv5_2_2.variables)
            tape.watch(self.conv5_3_1.variables)
            tape.watch(self.conv5_3_2.variables)
            tape.watch(self.fcn.variables)

            output = self.call(image)
            
            loss = tf.keras.losses.categorical_crossentropy(label, output)

            g = tape.gradient(loss, 
                [self.conv1.variables[0], self.conv1.variables[1],
                self.conv2_1_1.variables[0], self.conv2_1_1.variables[1], 
                self.conv2_1_2.variables[0], self.conv2_1_2.variables[1],
                self.conv2_2_1.variables[0], self.conv2_2_1.variables[1], 
                self.conv2_2_2.variables[0], self.conv2_2_2.variables[1],
                self.conv2_3_1.variables[0], self.conv2_3_1.variables[1],
                self.conv2_3_2.variables[0], self.conv2_3_2.variables[1],
                self.conv3_1_1.variables[0], self.conv3_1_1.variables[1], 
                self.conv3_1_2.variables[0], self.conv3_1_2.variables[1],
                self.conv3_2_1.variables[0], self.conv3_2_1.variables[1], 
                self.conv3_2_2.variables[0], self.conv3_2_2.variables[1],
                self.conv3_3_1.variables[0], self.conv3_3_1.variables[1],
                self.conv3_3_2.variables[0], self.conv3_3_2.variables[1],
                self.conv3_4_1.variables[0], self.conv3_4_1.variables[1],
                self.conv3_4_2.variables[0], self.conv3_4_2.variables[1],
                self.conv4_1_1.variables[0], self.conv4_1_1.variables[1], 
                self.conv4_1_2.variables[0], self.conv4_1_2.variables[1],
                self.conv4_2_1.variables[0], self.conv4_2_1.variables[1], 
                self.conv4_2_2.variables[0], self.conv4_2_2.variables[1],
                self.conv4_3_1.variables[0], self.conv4_3_1.variables[1],
                self.conv4_3_2.variables[0], self.conv4_3_2.variables[1],
                self.conv4_4_1.variables[0], self.conv4_4_1.variables[1],
                self.conv4_4_2.variables[0], self.conv4_4_2.variables[1],
                self.conv4_5_1.variables[0], self.conv4_5_1.variables[1],
                self.conv4_5_2.variables[0], self.conv4_5_2.variables[1],
                self.conv4_6_1.variables[0], self.conv4_6_1.variables[1],
                self.conv4_6_2.variables[0], self.conv4_6_2.variables[1],
                self.conv5_1_1.variables[0], self.conv5_1_1.variables[1], 
                self.conv5_1_2.variables[0], self.conv5_1_2.variables[1],
                self.conv5_2_1.variables[0], self.conv5_2_1.variables[1], 
                self.conv5_2_2.variables[0], self.conv5_2_2.variables[1],
                self.conv5_3_1.variables[0], self.conv5_3_1.variables[1],
                self.conv5_3_2.variables[0], self.conv5_3_2.variables[1],
                self.fcn.variables[0], self.fcn.variables[1]]) # fcn에서 에러가 뜬다. 왜지?

            return g, loss
    def Plus_grad(self, g_list_1, g_list_2) :
        for i in range(0, len(g_list_1)) :
            g_list_1[i] = tf.math.add(g_list_1[i], g_list_2[i])
        return g_list_1
    def Div_grad(self, g_list, num) :
        for i in range(0, len(g_list)) :
            g_list[i] = tf.math.scalar_mul(1/num, g_list[i])
        return g_list
        
    def App_Gradient(self, image_minibatch, label_minibatch, lr = 0.1) :
        # 출력값을 구한다 -> 로스를 구한다 -> 그레디언트를 구한다 -> 적용 시킨다
        total_g = 0
        total_loss = 0
        for i in range(0, len(image_minibatch)) :
            image = np.expand_dims(image_minibatch[i], axis = 0)
            label = label_minibatch[i]
            g, loss = self.Get_Gradient(image, label)
            if i == 0 :
                total_g = copy.deepcopy(g) # 그래디언트 '리스트'다
                total_loss = tf.identity(loss)
            else :
                temp = copy.deepcopy(g)
                total_g = self.Plus_grad(total_g, g)
                total_loss = tf.math.add(total_loss, loss)
                
        avr_g = self.Div_grad(total_g, len(image_minibatch)) # 미니배치의 평균 gradient를 구한다
        avr_loss = total_loss/len(image_minibatch)

        self.Optimizers.apply_gradients(zip(avr_g, [
                self.conv1.variables[0], self.conv1.variables[1],
                self.conv2_1_1.variables[0], self.conv2_1_1.variables[1], 
                self.conv2_1_2.variables[0], self.conv2_1_2.variables[1],
                self.conv2_2_1.variables[0], self.conv2_2_1.variables[1], 
                self.conv2_2_2.variables[0], self.conv2_2_2.variables[1],
                self.conv2_3_1.variables[0], self.conv2_3_1.variables[1],
                self.conv2_3_2.variables[0], self.conv2_3_2.variables[1],
                self.conv3_1_1.variables[0], self.conv3_1_1.variables[1], 
                self.conv3_1_2.variables[0], self.conv3_1_2.variables[1],
                self.conv3_2_1.variables[0], self.conv3_2_1.variables[1], 
                self.conv3_2_2.variables[0], self.conv3_2_2.variables[1],
                self.conv3_3_1.variables[0], self.conv3_3_1.variables[1],
                self.conv3_3_2.variables[0], self.conv3_3_2.variables[1],
                self.conv3_4_1.variables[0], self.conv3_4_1.variables[1],
                self.conv3_4_2.variables[0], self.conv3_4_2.variables[1],
                self.conv4_1_1.variables[0], self.conv4_1_1.variables[1], 
                self.conv4_1_2.variables[0], self.conv4_1_2.variables[1],
                self.conv4_2_1.variables[0], self.conv4_2_1.variables[1], 
                self.conv4_2_2.variables[0], self.conv4_2_2.variables[1],
                self.conv4_3_1.variables[0], self.conv4_3_1.variables[1],
                self.conv4_3_2.variables[0], self.conv4_3_2.variables[1],
                self.conv4_4_1.variables[0], self.conv4_4_1.variables[1],
                self.conv4_4_2.variables[0], self.conv4_4_2.variables[1],
                self.conv4_5_1.variables[0], self.conv4_5_1.variables[1],
                self.conv4_5_2.variables[0], self.conv4_5_2.variables[1],
                self.conv4_6_1.variables[0], self.conv4_6_1.variables[1],
                self.conv4_6_2.variables[0], self.conv4_6_2.variables[1],
                self.conv5_1_1.variables[0], self.conv5_1_1.variables[1], 
                self.conv5_1_2.variables[0], self.conv5_1_2.variables[1],
                self.conv5_2_1.variables[0], self.conv5_2_1.variables[1], 
                self.conv5_2_2.variables[0], self.conv5_2_2.variables[1],
                self.conv5_3_1.variables[0], self.conv5_3_1.variables[1],
                self.conv5_3_2.variables[0], self.conv5_3_2.variables[1],
                self.fcn.variables[0], self.fcn.variables[1]]))

        return avr_g, avr_loss # 가중치, 로스 확인을 위해 반환
        

    def training(self, input_list, label_list) :
        # mini batch 생성
        input_minibatch_list = []
        label_minibatch_list = []
        count = 0
        temp_input_minibatch = []
        temp_label_minibatch = []

        bar = tqdm(range(0, len(input_list)), desc = "make minibatch" )
        for i in bar :
            temp_input_minibatch.append(input_list[i])
            temp_label_minibatch.append(label_list[i])
            count = count + 1
            if count % 256 == 0 or i == len(input_list) - 1:
                input_minibatch_list.append(temp_input_minibatch)
                label_minibatch_list.append(temp_label_minibatch)
                temp_input_minibatch = []
                temp_label_minibatch = []
        
        bar = tqdm(range(0, len(input_minibatch_list)), desc = "training ResNet")
        
        grad_one_epoch = 0
        for i in bar :
            avr_g, avr_loss = self.App_Gradient(input_minibatch_list[i], label_minibatch_list[i]) # 미니배치 단위로 그레디언트 적용
            if grad_one_epoch == 0 :
                grad_one_epoch = copy.deepcopy(avr_g)
            else :
                grad_one_epoch = self.Plus_grad(grad_one_epoch, avr_g)
            
            desc_str = "training ResNet, Loss = %f " % avr_loss
            bar.set_description(desc_str)
        
        grad_one_epoch = self.Div_grad(grad_one_epoch, len(input_minibatch_list))

        return grad_one_epoch

class PlainNet34(tf.keras.Model):
    def __init__(self):
        super(PlainNet34, self).__init__(name='PlainNet34')
        self.Optimizers = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

        # 레이어(논문에 나온 구조 그대로 사용)
        self.BN = tf.keras.layers.BatchNormalization() # 모든 컨볼루션 연산 이후 거쳐야한다.
        self.ReLU = tf.keras.layers.ReLU() # Conv -> BN -> ReLU

        regularizer = tf.keras.regularizers.l2(0.0005) # weight decay

        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev= math.sqrt(2/(64*7^2)) , seed=None) # sqrt(2/(레이어 필터 개수 * 필터 크기의 제곱))
        self.conv1 = tf.keras.layers.Conv2D(64, 7, strides=(2, 2),padding="same", input_shape=(1, 224, 224, 3)) # downsampling directly by convolutional layers that have a stride of 2
        self.maxPooling = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))


        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev= math.sqrt(2/(64*3^2)) , seed=None)
        self.conv2_1_1 = tf.keras.layers.Conv2D(64, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv2_1_2 = tf.keras.layers.Conv2D(64, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv2_2_1 = tf.keras.layers.Conv2D(64, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv2_2_2 = tf.keras.layers.Conv2D(64, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv2_3_1 = tf.keras.layers.Conv2D(64, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv2_3_2 = tf.keras.layers.Conv2D(64, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")

        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev= math.sqrt(2/(128*3^2)) , seed=None)
        self.conv3_1_1 = tf.keras.layers.Conv2D(128, 3, strides=(2, 2), kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv3_1_2 = tf.keras.layers.Conv2D(128, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv3_2_1 = tf.keras.layers.Conv2D(128, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv3_2_2 = tf.keras.layers.Conv2D(128, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv3_3_1 = tf.keras.layers.Conv2D(128, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv3_3_2 = tf.keras.layers.Conv2D(128, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv3_4_1 = tf.keras.layers.Conv2D(128, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv3_4_2 = tf.keras.layers.Conv2D(128, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")

        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev= math.sqrt(2/(256*3^2)) , seed=None)
        self.conv4_1_1 = tf.keras.layers.Conv2D(256, 3, strides=(2, 2), kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv4_1_2 = tf.keras.layers.Conv2D(256, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv4_2_1 = tf.keras.layers.Conv2D(256, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv4_2_2 = tf.keras.layers.Conv2D(256, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv4_3_1 = tf.keras.layers.Conv2D(256, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv4_3_2 = tf.keras.layers.Conv2D(256, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv4_4_1 = tf.keras.layers.Conv2D(256, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv4_4_2 = tf.keras.layers.Conv2D(256, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv4_5_1 = tf.keras.layers.Conv2D(256, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv4_5_2 = tf.keras.layers.Conv2D(256, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv4_6_1 = tf.keras.layers.Conv2D(256, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv4_6_2 = tf.keras.layers.Conv2D(256, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")

        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev= math.sqrt(2/(512*3^2)) , seed=None)
        self.conv5_1_1 = tf.keras.layers.Conv2D(512, 3, strides=(2, 2), kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv5_1_2 = tf.keras.layers.Conv2D(512, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv5_2_1 = tf.keras.layers.Conv2D(512, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv5_2_2 = tf.keras.layers.Conv2D(512, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv5_3_1 = tf.keras.layers.Conv2D(512, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")
        self.conv5_3_2 = tf.keras.layers.Conv2D(512, 3, kernel_initializer=initializer, kernel_regularizer = regularizer, padding="same")

        self.average_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.fcn = tf.keras.layers.Dense(275, activation='softmax')

    def calc_Conv(self, input, layer) :
        output = layer(input)
        output = self.BN(output)
        output = self.ReLU(output)

        return output

    def call(self, image): # (1, 224, 224, 3) tensor를 받는다.
        output = self.conv1(image)
        output = self.BN(output)
        output = self.ReLU(output)
        output = self.maxPooling(output)

        # Skip Connection없음
        output = self.calc_Conv(output, self.conv2_1_1)
        output = self.calc_Conv(output, self.conv2_1_2)
        output = self.calc_Conv(output, self.conv2_2_1)
        output = self.calc_Conv(output, self.conv2_2_2)
        output = self.calc_Conv(output, self.conv2_3_1)
        output = self.calc_Conv(output, self.conv2_3_2)

        output = self.calc_Conv(output, self.conv3_1_1)
        output = self.calc_Conv(output, self.conv3_1_2)
        output = self.calc_Conv(output, self.conv3_2_1)
        output = self.calc_Conv(output, self.conv3_2_2)
        output = self.calc_Conv(output, self.conv3_3_1)
        output = self.calc_Conv(output, self.conv3_3_2)
        output = self.calc_Conv(output, self.conv3_4_1)
        output = self.calc_Conv(output, self.conv3_4_2)

        output = self.calc_Conv(output, self.conv4_1_1)
        output = self.calc_Conv(output, self.conv4_1_2)
        output = self.calc_Conv(output, self.conv4_2_1)
        output = self.calc_Conv(output, self.conv4_2_2)
        output = self.calc_Conv(output, self.conv4_3_1)
        output = self.calc_Conv(output, self.conv4_3_2)
        output = self.calc_Conv(output, self.conv4_4_1)
        output = self.calc_Conv(output, self.conv4_4_2)
        output = self.calc_Conv(output, self.conv4_5_1)
        output = self.calc_Conv(output, self.conv4_5_2)
        output = self.calc_Conv(output, self.conv4_6_1)
        output = self.calc_Conv(output, self.conv4_6_2)

        output = self.calc_Conv(output, self.conv5_1_1)
        output = self.calc_Conv(output, self.conv5_1_2)
        output = self.calc_Conv(output, self.conv5_2_1)
        output = self.calc_Conv(output, self.conv5_2_2)
        output = self.calc_Conv(output, self.conv5_3_1)
        output = self.calc_Conv(output, self.conv5_3_2)

        output = self.average_pool(output)
        output = self.flatten(output)

        output = self.fcn(output)

        return output

    # 각 입력 데이터에 대한 loss를 구하고 grad를 구한다.
    def Get_Gradient(self, image, label) :
        with tf.GradientTape() as tape:
            tape.watch(self.conv1.variables)
            tape.watch(self.conv2_1_1.variables)
            tape.watch(self.conv2_1_2.variables)
            tape.watch(self.conv2_2_1.variables)
            tape.watch(self.conv2_2_2.variables)
            tape.watch(self.conv2_3_1.variables)
            tape.watch(self.conv2_3_2.variables)
            tape.watch(self.conv3_1_1.variables)
            tape.watch(self.conv3_1_2.variables)
            tape.watch(self.conv3_2_1.variables)
            tape.watch(self.conv3_2_2.variables)
            tape.watch(self.conv3_3_1.variables)
            tape.watch(self.conv3_3_2.variables)
            tape.watch(self.conv3_4_1.variables)
            tape.watch(self.conv3_4_2.variables)
            tape.watch(self.conv4_1_1.variables)
            tape.watch(self.conv4_1_2.variables)
            tape.watch(self.conv4_2_1.variables)
            tape.watch(self.conv4_2_2.variables)
            tape.watch(self.conv4_3_1.variables)
            tape.watch(self.conv4_3_2.variables)
            tape.watch(self.conv4_4_1.variables)
            tape.watch(self.conv4_4_2.variables)
            tape.watch(self.conv4_5_1.variables)
            tape.watch(self.conv4_5_2.variables)
            tape.watch(self.conv4_6_1.variables)
            tape.watch(self.conv4_6_2.variables)
            tape.watch(self.conv5_1_1.variables)
            tape.watch(self.conv5_1_2.variables)
            tape.watch(self.conv5_2_1.variables)
            tape.watch(self.conv5_2_2.variables)
            tape.watch(self.conv5_3_1.variables)
            tape.watch(self.conv5_3_2.variables)
            tape.watch(self.fcn.variables)

            output = self.call(image)
            
            loss = tf.keras.losses.categorical_crossentropy(label, output)

            g = tape.gradient(loss, 
                [self.conv1.variables[0], self.conv1.variables[1],
                self.conv2_1_1.variables[0], self.conv2_1_1.variables[1], 
                self.conv2_1_2.variables[0], self.conv2_1_2.variables[1],
                self.conv2_2_1.variables[0], self.conv2_2_1.variables[1], 
                self.conv2_2_2.variables[0], self.conv2_2_2.variables[1],
                self.conv2_3_1.variables[0], self.conv2_3_1.variables[1],
                self.conv2_3_2.variables[0], self.conv2_3_2.variables[1],
                self.conv3_1_1.variables[0], self.conv3_1_1.variables[1], 
                self.conv3_1_2.variables[0], self.conv3_1_2.variables[1],
                self.conv3_2_1.variables[0], self.conv3_2_1.variables[1], 
                self.conv3_2_2.variables[0], self.conv3_2_2.variables[1],
                self.conv3_3_1.variables[0], self.conv3_3_1.variables[1],
                self.conv3_3_2.variables[0], self.conv3_3_2.variables[1],
                self.conv3_4_1.variables[0], self.conv3_4_1.variables[1],
                self.conv3_4_2.variables[0], self.conv3_4_2.variables[1],
                self.conv4_1_1.variables[0], self.conv4_1_1.variables[1], 
                self.conv4_1_2.variables[0], self.conv4_1_2.variables[1],
                self.conv4_2_1.variables[0], self.conv4_2_1.variables[1], 
                self.conv4_2_2.variables[0], self.conv4_2_2.variables[1],
                self.conv4_3_1.variables[0], self.conv4_3_1.variables[1],
                self.conv4_3_2.variables[0], self.conv4_3_2.variables[1],
                self.conv4_4_1.variables[0], self.conv4_4_1.variables[1],
                self.conv4_4_2.variables[0], self.conv4_4_2.variables[1],
                self.conv4_5_1.variables[0], self.conv4_5_1.variables[1],
                self.conv4_5_2.variables[0], self.conv4_5_2.variables[1],
                self.conv4_6_1.variables[0], self.conv4_6_1.variables[1],
                self.conv4_6_2.variables[0], self.conv4_6_2.variables[1],
                self.conv5_1_1.variables[0], self.conv5_1_1.variables[1], 
                self.conv5_1_2.variables[0], self.conv5_1_2.variables[1],
                self.conv5_2_1.variables[0], self.conv5_2_1.variables[1], 
                self.conv5_2_2.variables[0], self.conv5_2_2.variables[1],
                self.conv5_3_1.variables[0], self.conv5_3_1.variables[1],
                self.conv5_3_2.variables[0], self.conv5_3_2.variables[1],
                self.fcn.variables[0], self.fcn.variables[1]]) # fcn에서 에러가 뜬다. 왜지?

            return g, loss
    def Plus_grad(self, g_list_1, g_list_2) :
        for i in range(0, len(g_list_1)) :
            g_list_1[i] = tf.math.add(g_list_1[i], g_list_2[i])
        return g_list_1
    def Div_grad(self, g_list, num) :
        for i in range(0, len(g_list)) :
            g_list[i] = tf.math.scalar_mul(1/num, g_list[i])
        return g_list
        
    def App_Gradient(self, image_minibatch, label_minibatch, lr = 0.1) :
        # 출력값을 구한다 -> 로스를 구한다 -> 그레디언트를 구한다 -> 적용 시킨다
        total_g = 0
        total_loss = 0
        for i in range(0, len(image_minibatch)) :
            image = np.expand_dims(image_minibatch[i], axis = 0)
            label = label_minibatch[i]
            g, loss = self.Get_Gradient(image, label)
            if i == 0 :
                total_g = copy.deepcopy(g) # 그래디언트 '리스트'다
                total_loss = tf.identity(loss)
            else :
                temp = copy.deepcopy(g)
                total_g = self.Plus_grad(total_g, g)
                total_loss = tf.math.add(total_loss, loss)
                
        avr_g = self.Div_grad(total_g, len(image_minibatch)) # 미니배치의 평균 gradient를 구한다
        avr_loss = total_loss/len(image_minibatch)

        self.Optimizers.apply_gradients(zip(avr_g, [
                self.conv1.variables[0], self.conv1.variables[1],
                self.conv2_1_1.variables[0], self.conv2_1_1.variables[1], 
                self.conv2_1_2.variables[0], self.conv2_1_2.variables[1],
                self.conv2_2_1.variables[0], self.conv2_2_1.variables[1], 
                self.conv2_2_2.variables[0], self.conv2_2_2.variables[1],
                self.conv2_3_1.variables[0], self.conv2_3_1.variables[1],
                self.conv2_3_2.variables[0], self.conv2_3_2.variables[1],
                self.conv3_1_1.variables[0], self.conv3_1_1.variables[1], 
                self.conv3_1_2.variables[0], self.conv3_1_2.variables[1],
                self.conv3_2_1.variables[0], self.conv3_2_1.variables[1], 
                self.conv3_2_2.variables[0], self.conv3_2_2.variables[1],
                self.conv3_3_1.variables[0], self.conv3_3_1.variables[1],
                self.conv3_3_2.variables[0], self.conv3_3_2.variables[1],
                self.conv3_4_1.variables[0], self.conv3_4_1.variables[1],
                self.conv3_4_2.variables[0], self.conv3_4_2.variables[1],
                self.conv4_1_1.variables[0], self.conv4_1_1.variables[1], 
                self.conv4_1_2.variables[0], self.conv4_1_2.variables[1],
                self.conv4_2_1.variables[0], self.conv4_2_1.variables[1], 
                self.conv4_2_2.variables[0], self.conv4_2_2.variables[1],
                self.conv4_3_1.variables[0], self.conv4_3_1.variables[1],
                self.conv4_3_2.variables[0], self.conv4_3_2.variables[1],
                self.conv4_4_1.variables[0], self.conv4_4_1.variables[1],
                self.conv4_4_2.variables[0], self.conv4_4_2.variables[1],
                self.conv4_5_1.variables[0], self.conv4_5_1.variables[1],
                self.conv4_5_2.variables[0], self.conv4_5_2.variables[1],
                self.conv4_6_1.variables[0], self.conv4_6_1.variables[1],
                self.conv4_6_2.variables[0], self.conv4_6_2.variables[1],
                self.conv5_1_1.variables[0], self.conv5_1_1.variables[1], 
                self.conv5_1_2.variables[0], self.conv5_1_2.variables[1],
                self.conv5_2_1.variables[0], self.conv5_2_1.variables[1], 
                self.conv5_2_2.variables[0], self.conv5_2_2.variables[1],
                self.conv5_3_1.variables[0], self.conv5_3_1.variables[1],
                self.conv5_3_2.variables[0], self.conv5_3_2.variables[1],
                self.fcn.variables[0], self.fcn.variables[1]]))

        return avr_g, avr_loss # 가중치, 로스 확인을 위해 반환
        

    def training(self, input_list, label_list) :
        # mini batch 생성
        input_minibatch_list = []
        label_minibatch_list = []
        count = 0
        temp_input_minibatch = []
        temp_label_minibatch = []

        bar = tqdm(range(0, len(input_list)), desc = "make minibatch" )
        for i in bar :
            temp_input_minibatch.append(input_list[i])
            temp_label_minibatch.append(label_list[i])
            count = count + 1
            if count % 256 == 0 or i == len(input_list) - 1:
                input_minibatch_list.append(temp_input_minibatch)
                label_minibatch_list.append(temp_label_minibatch)
                temp_input_minibatch = []
                temp_label_minibatch = []
        
        bar = tqdm(range(0, len(input_minibatch_list)), desc = "training ResNet")
        
        grad_one_epoch = 0
        for i in bar :
            avr_g, avr_loss = self.App_Gradient(input_minibatch_list[i], label_minibatch_list[i]) # 미니배치 단위로 그레디언트 적용
            if grad_one_epoch == 0 :
                grad_one_epoch = copy.deepcopy(avr_g)
            else :
                grad_one_epoch = self.Plus_grad(grad_one_epoch, avr_g)
            
            desc_str = "training ResNet, Loss = %f " % avr_loss
            bar.set_description(desc_str)
        
        grad_one_epoch = self.Div_grad(grad_one_epoch, len(input_minibatch_list))

        return grad_one_epoch 

                
def preprocessing_Dataset(root_path) : # train - label - data 구조로 이루어진 데이터셋인 경우 데이터셋 내에 있는 파일을 뽑아내고 라벨 데이터도 만들어내는 함수. '/home/ubuntu/CUAI_2021/Advanced_Minkyu_Kim/Bird_Dataset'같은 폴더 경로를 넣어준다.
    file_list = os.listdir(root_path) # 폴더 내에 있는 파일(폴더 포함)리스트 얻음

    class_idx = 0
    class_list = [] # 데이터셋 내부에 있는 클래스 종류를 얻는다.
    image_list = []
    label_list = []

    bar = tqdm(range(0, len(file_list)), desc = "preprocessing...")

    for i in bar :
        class_list.append(file_list[i])
        class_iamges_folder_path = root_path + '/' + file_list[i]
        class_image_list = sorted([x for x in glob(class_iamges_folder_path + '/**')])
        for j in range(0, 20) :
            # 이미지 전처리(사이즈 변경, RGB범위를 0~255 -> 0~1)
            image = cv2.imread(class_image_list[j])
            image = cv2.resize(image, (224, 224))/255
            # one-hot encoding
            label = np.zeros(len(file_list))
            label[class_idx] = 1

            image_list.append(image)
            label_list.append(label)
        class_idx = class_idx + 1

    image_list = np.asarray(image_list)
    label_list = np.asarray(label_list)
    
    arr_forShuffle = np.arange(image_list.shape[0])
    np.random.shuffle(arr_forShuffle)
    
    image_list = image_list[arr_forShuffle]
    label_list = label_list[arr_forShuffle]

    image_list = image_list.astype('float32')

    return image_list, label_list, np.asarray(class_list)









        
        




