import numpy as np
import copy
import tensorflow as tf
import random
from tqdm import tqdm

# RPN(Region Proposal Networks)
# 원본 이미지(224*224크기에 r,g,b채널을 갖고 있는 이미지)를 입력받아 (1764,2), (1764,4) 크기의 텐서를 출력으로 받는다. 
# 224*224이미지를 VGG16에 넣으면 14*14*512 텐서가 나오는데 이 때 14*14의 한 픽셀에서 9개의 앵커를 갖다대며 물체가 있는지 없는지 추측하고 그 추측 영역도 내놓는다.
# 그래서 총 앵커는 14*14*9 = 1764개이며 각 앵커에 대한 object score를 출력해야하기 때문에 한 앵커당 2개의 output => (1764,2) 텐서를 출력값으로 내놓는다. 
# 그리고 추측 영역을 (x,y,w,h)로 나타내기 때문에 각 앵커당 4개의 출력값이 필요해 (1764,4) 텐서를 출력값으로 내놓는 것. 출력값은 각 앵커를 기반으로 추측하는 것이기 때문에 (x + dx, y + dy, w + dw, h + dh)다.
# 논문에서는 이를 두고 '각 특성맵의 픽셀당 k개의 앵커가 있고 k개의 앵커를 통해 2k개의 score output, 4k개의 coordinates of boxes을 내놓는다'고 말한다. 
class RPN(tf.keras.Model):
    def __init__(self, initializer, regularizer, SharedConvNet):
        super(RPN, self).__init__(name='rpn')
        
        self.Optimizers = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9)
        self.training_count = 0

        # 공용 레이어
        self.conv1_1 = SharedConvNet.layers[0]
        self.conv1_2 = SharedConvNet.layers[1]
        self.pooling_1 = SharedConvNet.layers[2]

        self.conv2_1 = SharedConvNet.layers[3]
        self.conv2_2 = SharedConvNet.layers[4]
        self.pooling_2 = SharedConvNet.layers[5]

        self.conv3_1 = SharedConvNet.layers[6]
        self.conv3_2 = SharedConvNet.layers[7]
        self.conv3_3 = SharedConvNet.layers[8]
        self.pooling_3 = SharedConvNet.layers[9]

        self.conv4_1 = SharedConvNet.layers[10]
        self.conv4_2 = SharedConvNet.layers[11]
        self.conv4_3 = SharedConvNet.layers[12]
        self.pooling_4 = SharedConvNet.layers[13]

        self.conv5_1 = SharedConvNet.layers[14]
        self.conv5_2 = SharedConvNet.layers[15]
        self.conv5_3 = SharedConvNet.layers[16]
    
        # RPN만의 레이어
        self.intermediate_layer = tf.keras.layers.Conv2D(512, (3, 3), padding = 'SAME' , activation = 'relu', name = "intermediate_layer", dtype='float32')
        self.cls_layer = tf.keras.layers.Conv2D(18, (1, 1), kernel_initializer=initializer, padding = 'SAME' ,kernel_regularizer = regularizer, name = "output_1", dtype='float32')
        self.reg_layer = tf.keras.layers.Conv2D(36, (1, 1), kernel_initializer=initializer, padding = 'SAME' ,kernel_regularizer = regularizer, name = "output_2", dtype='float32')
    
    def call(self, inputs, anchor_optimize_list):
        # 정방향 연산
        # inputs = np.array(inputs)
        output = self.conv1_1(inputs)
        output = self.conv1_2(output)
        output = self.pooling_1(output)

        output = self.conv2_1(output)
        output = self.conv2_2(output)
        output = self.pooling_2(output)

        output = self.conv3_1(output)
        output = self.conv3_2(output)
        output = self.conv3_3(output)
        output = self.pooling_3(output)

        output = self.conv4_1(output)
        output = self.conv4_2(output)
        output = self.conv4_3(output)
        output = self.pooling_4(output)

        output = self.conv5_1(output)
        output = self.conv5_2(output)
        shared_output = self.conv5_3(output)
        # RPN
        shared_output = self.intermediate_layer(shared_output)
        cls_layer_output = self.cls_layer(shared_output) # (1,14,14,18)
        reg_layer_output = self.reg_layer(shared_output) # (1,14,14,36)
        
        # [1764, 4], [1764, 2]로 가공
        cls_layer_output = tf.reshape(cls_layer_output, [1764,2])
        cls_layer_output = tf.nn.softmax(cls_layer_output) 
        
        reg_layer_output = tf.reshape(reg_layer_output, [1764,4])
        
        anchor_optimize_list_Use = copy.deepcopy(anchor_optimize_list)
        
        anchor_tensor = tf.convert_to_tensor(anchor_optimize_list_Use, dtype=tf.float32)
        reg_layer_output = tf.math.add(reg_layer_output, anchor_tensor)
        
        return cls_layer_output, reg_layer_output

    def get_minibatch_index(self, cls_layer_label): # 라벨이니까 (1764,2) 넘파이 온다

        index_list = np.zeros(1764) # 각 앵커가 미니배치 뽑혔나 안뽑혔나
        index_pos = np.array([])
        index_neg = np.array([])
        # cls_layer_label을 보고 긍정, 부정 앵커 분류. 그렇게 데이터셋을 구성함
        for i in range(0, 1764):
            if cls_layer_label[i][0] == 1.0 : # positive anchor
                index_pos = np.append(index_pos, i)
            elif cls_layer_label[i][0] == 0.0 : # negative anchor
                index_neg = np.append(index_neg, i)

        max_for = min([128, len(index_pos)])
        ran_list = random.sample(range(0, len(index_pos)), max_for)

        for i in range(0, len(ran_list)) :
            index = int(index_pos[ran_list[i]])
            index_list[index] = 1

        ran_list = random.sample(range(0, len(index_neg)), 256 - max_for) # 랜덤성 증가?를 위해 또다시 난수 생성
        for i in range(0, len(ran_list)) :
            index = int(index_neg[ran_list[i]])
            index_list[index] = 1

        return index_list # (1764,1) <- 1,0으로 이루어진 boolean 넘파이 배열

    # multi task loss
    def multi_task_loss(self, cls_layer_output, reg_layer_output ,cls_layer_label, reg_layer_label, minibatch_index_list, anchor_optimize_list):
        
        # label은 (1764,2)와 (1764,4)임
        tensor_cls_label = tf.convert_to_tensor(cls_layer_label, dtype=tf.float32)
        tensor_reg_label = tf.convert_to_tensor(reg_layer_label, dtype=tf.float32)

        # loss 계산(Loss 텐서에서 미니배치에 해당되는 애들만 걸러내야함)
        Cls_Loss = tf.nn.softmax_cross_entropy_with_logits(labels=tensor_cls_label, logits = cls_layer_output) # (1764,1) 텐서
        Cls_Loss = tf.reshape(Cls_Loss, [1764, 1])

        filter_x = tf.Variable([[1.0],[0.0],[0.0], [0.0]])
        filter_y = tf.Variable([[0.0],[1.0],[0.0], [0.0]])
        filter_w = tf.Variable([[0.0],[0.0],[1.0], [0.0]])
        filter_h = tf.Variable([[0.0],[0.0],[0.0], [1.0]])

        x = tf.matmul(reg_layer_output,filter_x)
        y = tf.matmul(reg_layer_output,filter_y)
        w = tf.matmul(reg_layer_output,filter_w)
        h = tf.matmul(reg_layer_output,filter_h)
        
        
        anchor_Use = copy.deepcopy(anchor_optimize_list)
            
        anchor_tensor = tf.convert_to_tensor(anchor_Use, dtype=tf.float32)
        
        x_a = tf.matmul(anchor_tensor,filter_x)
        y_a = tf.matmul(anchor_tensor,filter_y)
        w_a = tf.matmul(anchor_tensor,filter_w)
        h_a = tf.matmul(anchor_tensor,filter_h)

        x_star = tf.matmul(tensor_reg_label,filter_x)
        y_star = tf.matmul(tensor_reg_label,filter_y)
        w_star = tf.matmul(tensor_reg_label,filter_w)
        h_star = tf.matmul(tensor_reg_label,filter_h)

        denominator = tf.math.log(tf.constant(10, dtype=tf.float32)) # 텐서 로그는 ln밖에 없어서 ln10을 구한 뒤 나누는 방식으로 log10을 구한다(로그의 밑변환 공식)

        # 4개만 떼서 계산하니까 잘됨
        t_x = tf.math.divide(tf.subtract(x, x_a), w_a)
        t_y = tf.math.divide(tf.subtract(y, y_a), h_a)
        t_w = tf.math.divide(tf.math.log(tf.math.divide(w, w_a)), denominator)
        t_h = tf.math.divide(tf.math.log(tf.math.divide(h, h_a)), denominator)

        t_x_star = tf.math.divide(tf.math.subtract(x_star, x_a), w_a)
        t_y_star = tf.math.divide(tf.math.subtract(y_star, y_a), h_a)
        t_w_star = tf.math.divide(tf.math.log(tf.math.divide(w_star, w_a)), denominator)
        t_h_star = tf.math.divide(tf.math.log(tf.math.divide(h_star, h_a)), denominator)
        # -inf -> 0으로 만들어준다.
        t_w_star = tf.where(tf.math.is_inf(t_w_star), tf.zeros_like(t_w_star), t_w_star)
        t_h_star = tf.where(tf.math.is_inf(t_h_star), tf.zeros_like(t_h_star), t_h_star)
        
        # (1764,1)에 해당하는 t_x, t_y...을 구했다. 여기서 미니배치에 해당되는 애들만 걸러낸다. 

        # 미니배치에 해당되는 애들만 0이 아닌 값으로 만들기. 미니배치 리스트는 미니배치에 해당되는 인덱스는 1이고 나머지는 다 0이니까 tf.math.multiply를 사용해 원소별 곱을 해주면 미니배치에 해당되는 값들만 얻을 수 있다. 
        minibatch_index_tensor = tf.convert_to_tensor(minibatch_index_list, dtype=tf.float32) # 텐서로 변환
        minibatch_index_tensor = tf.reshape(minibatch_index_tensor, [1764, 1])

        # 다 곱해서 미니배치 성분만 남기기
        t_x_minibatch = tf.math.multiply(t_x, minibatch_index_tensor)
        t_y_minibatch = tf.math.multiply(t_y, minibatch_index_tensor)
        t_w_minibatch = tf.math.multiply(t_w, minibatch_index_tensor)
        t_h_minibatch = tf.math.multiply(t_h, minibatch_index_tensor)        

        t_x_star_minibatch = tf.math.multiply(t_x_star, minibatch_index_tensor)
        t_y_star_minibatch = tf.math.multiply(t_y_star, minibatch_index_tensor)
        t_w_star_minibatch = tf.math.multiply(t_w_star, minibatch_index_tensor)
        t_h_star_minibatch = tf.math.multiply(t_h_star, minibatch_index_tensor)
        
        
        
        t_w_minibatch = tf.where(tf.math.is_nan(t_w_minibatch), tf.zeros_like(t_w_minibatch), t_w_minibatch)
        t_h_minibatch = tf.where(tf.math.is_nan(t_h_minibatch), tf.zeros_like(t_h_minibatch), t_h_minibatch)

        t_w_star_minibatch = tf.where(tf.math.is_nan(t_w_star_minibatch), tf.zeros_like(t_w_star_minibatch), t_w_star_minibatch)
        t_h_star_minibatch = tf.where(tf.math.is_nan(t_h_star_minibatch), tf.zeros_like(t_h_star_minibatch), t_h_star_minibatch)

        Cls_Loss_minibatch = tf.math.multiply(Cls_Loss, minibatch_index_tensor)

        # 각 성분별로 1764개분 Loss를 다 합친 4개의 값이 나왔다
        # X성질에 대한 Smooth L1. huber_loss에서 delta = 1로 하면 smooth L1과 같다.
        # 미니배치 성분만 뽑아내서 미니배치가 아닌 인덱스의 값은 0인데 Smooth L1에서 |x| < 1이면 0.5*x^2니까 0이 나오며 이는 loss에 어떠한 영향을 미치지 않는다. 
        x_huber_loss = tf.compat.v1.losses.huber_loss(t_x_star_minibatch, t_x_minibatch) 
        y_huber_loss = tf.compat.v1.losses.huber_loss(t_y_star_minibatch, t_y_minibatch)
        w_huber_loss = tf.compat.v1.losses.huber_loss(t_w_star_minibatch, t_w_minibatch) 
        h_huber_loss = tf.compat.v1.losses.huber_loss(t_h_star_minibatch, t_h_minibatch)

        # print(x_huber_loss, y_huber_loss, w_huber_loss, h_huber_loss)

        # 한 번에 더하니까 에러가 발생해 tf.math.add로 두개씩 더한다.
        Reg_Loss = tf.math.add(x_huber_loss, y_huber_loss)   
        Reg_Loss = tf.math.add(Reg_Loss, w_huber_loss) # (x_huber_loss + y_huber_loss) + w_huber_loss
        Reg_Loss = tf.math.add(Reg_Loss, h_huber_loss) # (x_huber_loss + y_huber_loss + w_huber_loss) + h_huber_loss

        N_reg = tf.constant([1.0/1764.0])
        N_cls = tf.constant([10.0/256.0]) # lambda도 곱한 값

        loss_cls = tf.multiply(N_reg, tf.reduce_sum(Cls_Loss_minibatch))
        loss_reg = tf.multiply(N_cls, Reg_Loss)
        
        loss = tf.add(loss_cls, loss_reg)
        
        if self.training_count >= 60000 and self.training_count <= 80000:
            div_constant = tf.constant([1.0/10.0])
            loss = tf.multiply(loss, div_constant)
        elif self.training_count > 80000 :
            self.training_count = 0
        
        return loss

    def get_grad(self, image, cls_layer_label, reg_layer_label, anchor_optimize_list, training_step): 
        g = 0
        
        minibatch_index_list = self.get_minibatch_index(cls_layer_label) # 미니배치 인덱스 휙득
        
        with tf.GradientTape() as tape:
            
            if training_step == 1 :
                tape.watch(self.conv3_1.variables)
                tape.watch(self.conv3_2.variables)
                tape.watch(self.conv3_3.variables)
                tape.watch(self.conv4_1.variables)
                tape.watch(self.conv4_2.variables)
                tape.watch(self.conv4_3.variables)
                tape.watch(self.conv5_1.variables)
                tape.watch(self.conv5_2.variables)
                tape.watch(self.conv5_3.variables)
                tape.watch(self.intermediate_layer.variables)
                tape.watch(self.cls_layer.variables)
                tape.watch(self.reg_layer.variables)
                
                cls_layer_output, reg_layer_output = self.call(image, anchor_optimize_list)
                
                Loss = self.multi_task_loss(cls_layer_output, reg_layer_output ,cls_layer_label, reg_layer_label, minibatch_index_list, anchor_optimize_list)
                g = tape.gradient(Loss, [self.conv3_1.variables[0], self.conv3_1.variables[1], 
                                        self.conv3_2.variables[0],self.conv3_2.variables[1], 
                                        self.conv3_3.variables[0],self.conv3_3.variables[1], 
                                        self.conv4_1.variables[0],self.conv4_1.variables[1], 
                                        self.conv4_2.variables[0],self.conv4_2.variables[1], 
                                        self.conv4_3.variables[0],self.conv4_3.variables[1], 
                                        self.conv5_1.variables[0],self.conv5_1.variables[1], 
                                        self.conv5_2.variables[0],self.conv5_2.variables[1], 
                                        self.conv5_3.variables[0],self.conv5_3.variables[1], 
                                        self.intermediate_layer.variables[0],self.intermediate_layer.variables[1], 
                                        self.cls_layer.variables[0],self.cls_layer.variables[1], 
                                        self.reg_layer.variables[0],self.reg_layer.variables[1]])
            
            else :
                tape.watch(self.intermediate_layer.variables)
                tape.watch(self.cls_layer.variables)
                tape.watch(self.reg_layer.variables)
                
                cls_layer_output, reg_layer_output = self.call(image)
                
                Loss = self.multi_task_loss(cls_layer_output, reg_layer_output ,cls_layer_label, reg_layer_label, minibatch_index_list)
                g = tape.gradient(Loss, [self.intermediate_layer.variables[0],self.intermediate_layer.variables[1], 
                                        self.cls_layer.variables[0],self.cls_layer.variables[1], 
                                        self.reg_layer.variables[0],self.reg_layer.variables[1]])

        return g, Loss
    
    
    def App_Gradient(self, training_step, image, cls_layer_label, reg_layer_label, anchor_optimize_list) :
        g, Loss = self.get_grad(image, cls_layer_label, reg_layer_label, anchor_optimize_list, training_step)
        
        if training_step == 1:
            self.Optimizers.apply_gradients(zip(g, [self.conv3_1.variables[0], self.conv3_1.variables[1], 
                                                    self.conv3_2.variables[0],self.conv3_2.variables[1], 
                                                    self.conv3_3.variables[0],self.conv3_3.variables[1], 
                                                    self.conv4_1.variables[0],self.conv4_1.variables[1], 
                                                    self.conv4_2.variables[0],self.conv4_2.variables[1], 
                                                    self.conv4_3.variables[0],self.conv4_3.variables[1], 
                                                    self.conv5_1.variables[0],self.conv5_1.variables[1],
                                                    self.conv5_2.variables[0],self.conv5_2.variables[1], 
                                                    self.conv5_3.variables[0],self.conv5_3.variables[1], 
                                                    self.intermediate_layer.variables[0],self.intermediate_layer.variables[1], 
                                                    self.cls_layer.variables[0],self.cls_layer.variables[1], 
                                                    self.reg_layer.variables[0],self.reg_layer.variables[1]]))
        elif training_step == 3:
            self.Optimizers.apply_gradients(zip(g, [self.intermediate_layer.variables[0],self.intermediate_layer.variables[1], 
                                                    self.cls_layer.variables[0],self.cls_layer.variables[1], 
                                                    self.reg_layer.variables[0],self.reg_layer.variables[1]]))
        return Loss
    # 모델 훈련
    def Training_model(self, image_list, cls_layer_label_list, reg_layer_label_list, anchor_optimize_list_forAllImage, training_step):
        
        self.training_count = 0
        
        bar = tqdm(range(0, len(image_list) ))
        loss_acc = 0.0
        for i in bar :
            image = np.expand_dims(image_list[i], axis = 0) # (1,224,224,3)으로 제작
            Loss = self.App_Gradient(training_step, image, cls_layer_label_list[i], reg_layer_label_list[i], anchor_optimize_list_forAllImage[i])    
            
            self.training_count = self.training_count + 1
            
            loss_acc = loss_acc + (Loss.numpy()).item()
            avr_loss = loss_acc / (float)(i + 1)
            desc_str = "training RPN, Loss = %f " % avr_loss
            bar.set_description(desc_str)
            if i == len(image_list) - 1 :
                loss_acc = 0.0
        
