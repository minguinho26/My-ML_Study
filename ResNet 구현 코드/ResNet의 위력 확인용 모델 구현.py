import tensorflow as tf
import math
import numpy as np
import tqdm

# ResNet과 PlainNet(Skip Connection 적용 안한거) 둘다 구현

# 입력 이미지 크기 : 224*224
# 최적화 : weight decay of 0.0001 and a momentum of 0.9(tf.keras.optimizers.SGD)
# 가중치 : 미니배치 SGD(배치 크기 : 256)
# 학습률 : 로스가 정체되면 10%로 줄임
# 손실 함수 : cross_entropy(tf.nn.softmax_cross_entropy_with_logits)
class ResNet34(tf.keras.Model):
    def __init__(self):
        
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
        self.fcn = tf.keras.layers.Dense(1000, activation='softmax') # 분류할 객체 종류가 1000가지
    
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
            #ori_input을 Ws와 곱해 output과 같은 크기가 되도록 해야한다. 나는 이를 reshape로 해결했다.
            size_toConvert = tf.shape(ori_input).numpy()
            size_toConvert[-1] = size_toConvert[-1] * 2
            ori_input = tf.reshape(ori_input, size_toConvert) # 이게 진짜 될까...?

            # 채널 숫자를 바꿨으니 이제 크기를 줄여보자
            ori_input = self.maxPooling(ori_input)
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

            loss = tf.keras.losses.CategoricalCrossentropy(label, output)

            g = tape.gradient(
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
                self.fcn.variables[0], self.fcn.variables[1])

            return g

    def App_Gradient(self, image_minibatch, label_minibatch, lr = 0.1) :
        # 출력값을 구한다 -> 로스를 구한다 -> 그레디언트를 구한다 -> 적용 시킨다
        total_g = 0
        for i in range(0, len(image_minibatch)) :
            image = image_minibatch[i]
            label = label_minibatch[i]
            g = self.Get_Gradient(image, label)
            if i == 0 :
                total_g = tf.identity(g)
            else :
                total_g = tf.math.add(total_g, g)
        avr_g = tf.math.divide(total_g, len(image_minibatch)) # 미니배치의 평균 gradient를 구한다

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

        return avr_g # 가중치 확인을 위해 반환
        

    def training(self, input_list, label_list) :
        # mini batch 생성
        input_minibatch_list = []
        label_minibatch_list = []
        count = 0
        temp_input_minibatch = []
        temp_label_minibatch = []

        bar = tqdm(range(0, len(input_list), desc = "make minibatch" ))

        for i in bar :
            temp_input_minibatch.append(input_list[i])
            temp_label_minibatch.append(label_list[i])
            count = count + 1
            if count == 256 or i == len(input_list):
                input_minibatch_list.append(temp_input_minibatch)
                label_minibatch_list.append(temp_label_minibatch)
                temp_input_minibatch = []
                temp_label_minibatch = []
        
        bar = tqdm(range(0, len(input_minibatch_list), desc = "apply grad of minibatch"))

        grad_one_epoch = 0
        for i in bar :
            avr_g = self.App_Gradient(input_minibatch_list[i], label_minibatch_list[i]) # 미니배치 단위로 그레디언트 적용
            if grad_one_epoch == 0 :
                grad_one_epoch = tf.identity(avr_g)
            else :
                grad_one_epoch = tf.math.add(grad_one_epoch, avr_g)
        grad_one_epoch = tf.math.divide(grad_one_epoch, len(input_minibatch_list))

        return grad_one_epoch 

class PlainNet34(tf.keras.Model):
    def __init__(self):
        
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
        self.fcn = tf.keras.layers.Dense(1000, activation='softmax') # 분류할 객체 종류가 1000가지

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

            loss = tf.keras.losses.CategoricalCrossentropy(label, output)

            g = tape.gradient(
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
                self.fcn.variables[0], self.fcn.variables[1])

            return g

    def App_Gradient(self, image_minibatch, label_minibatch, lr = 0.1) :
        # 출력값을 구한다 -> 로스를 구한다 -> 그레디언트를 구한다 -> 적용 시킨다
        total_g = 0
        for i in range(0, len(image_minibatch)) :
            image = image_minibatch[i]
            label = label_minibatch[i]
            g = self.Get_Gradient(image, label)
            if i == 0 :
                total_g = tf.identity(g)
            else :
                total_g = tf.math.add(total_g, g)
        avr_g = tf.math.divide(total_g, len(image_minibatch)) # 미니배치의 평균 gradient를 구한다

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

        return avr_g # 가중치 확인을 위해 반환
        

    def training(self, input_list, label_list) :
        # mini batch 생성
        input_minibatch_list = []
        label_minibatch_list = []
        count = 0
        temp_input_minibatch = []
        temp_label_minibatch = []

        bar = tqdm(range(0, len(input_list), desc = "make minibatch" ))

        for i in bar :
            temp_input_minibatch.append(input_list[i])
            temp_label_minibatch.append(label_list[i])
            count = count + 1
            if count == 256 or i == len(input_list):
                input_minibatch_list.append(temp_input_minibatch)
                label_minibatch_list.append(temp_label_minibatch)
                temp_input_minibatch = []
                temp_label_minibatch = []
        
        bar = tqdm(range(0, len(input_minibatch_list), desc = "apply grad of minibatch"))

        grad_one_epoch = 0
        for i in bar :
            avr_g = self.App_Gradient(input_minibatch_list[i], label_minibatch_list[i]) # 미니배치 단위로 그레디언트 적용
            if grad_one_epoch == 0 :
                grad_one_epoch = tf.identity(avr_g)
            else :
                grad_one_epoch = tf.math.add(grad_one_epoch, avr_g)
        grad_one_epoch = tf.math.divide(grad_one_epoch, len(input_minibatch_list))

        return grad_one_epoch 

                






        
        




