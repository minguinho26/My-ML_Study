import numpy as np
import tensorflow as tf
import random
from tqdm import tqdm

# RoI Pooling Layer + Fast RCNN

# Pooling 작업을 RoI 지역에 대해 한다. 14*14*512를 7*7*512로
class RoiPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, pool_size):
        super(RoiPoolingLayer, self).__init__(name='RoI_Pooling_Layer')
        self.pool_size = pool_size # VGG16에서는 7*7
        
    def build(self, input_shape): # input shape로 (1,14,14,512)와 같이 받으니까 3번 원소 자리의 값인 512를 가져간다. 
        self.nb_channels = input_shape[3] # 채널 조정
        # 맨처음 입력받을 때 채널 숫자를 받는다. 풀링이라 채널 개수를 유지해야하기 때문

    def compute_output_shape(self, input_shape): # If the layer has not been built, this method will call build on the layer. 
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, image, RoI_inImage): # 정방향 연산. shared FeatureMap (1,14,14,512)와 입력 이미지에 있는 RoI 리스트를 받는다. RoI_inImage는 리스트다
        # 해야될거
        # 1. RoiPooling을 위해 RoI 영역을 (14,14)에 맞게 변형하기

        RoiPooling_outputs_List = [] # RoI 부근을 잘라낸 뒤 7*7로 만들어낸 것들을 여기에 모은다. 그러면 (n,1,7,7,512)가 되겠지

        for i in range(0, len(RoI_inImage)): # 이미지 당 RoI 갯수만큼 for문 반복 -> RoI 갯수만큼 특성맵 얻으려고
            # 224 -> 14로 16배 줄어들었으니 이에 맞춰 RoI도 줄인다. 
            # RoI 양식이(x,y,w,h)였는데 이를 (r,c,w,h)로 바꿔준다. r,c는 왼쪽 위 좌표(min x, min y)고 w,h,는 RoI의 너비, 높이다. 

            r = RoI_inImage[i][0] - RoI_inImage[i][2]/2
            c = RoI_inImage[i][1] - RoI_inImage[i][3]/2
            w = RoI_inImage[i][2]
            h = RoI_inImage[i][3]
            
            # 1/16배로 만들기
            r = round(r / 16.0)
            c = round(c / 16.0)
            w = round(w / 16.0)
            h = round(h / 16.0)

            # 제일 큰 앵커 사이즈가 128*128인데 이는 (14,14)에서 (8,8)이 된다. 
            # 제일 작은 앵커는 (2,2)이다. 그래서 나는 'by copying 7 times each cell and then max-pooling back to 7x7', 즉 이미지의 각 셀을 7*7로 복사한 뒤 (7*7)로 max Pooling하는 방식을 사용하고자 했다.
            # 아이디어 출처 : https://stackoverflow.com/questions/48163961/how-do-you-do-roi-pooling-on-areas-smaller-than-the-target-size
            # 그러나 이 방식은 너무 까다로워 max pooling의 대체지인 resize를 사용하기로 했다. 픽셀간 경계부분 말고는 7*7 출력값이 달라지는게 없어서 큰 차이는 없을것으로 예상된다.
            image_inRoI = image[:, c:c+h, r:r+w, :] # RoI에 해당되는 부분을 추출한다.
            image_resize = tf.image.resize(image_inRoI, (self.pool_size, self.pool_size)) # 7*7로 resize
            RoiPooling_outputs_List.append(image_resize)

        # RoiPooling_outputs_List는 (1,7,7,512) 텐서들로 이루어진 리스트다
        final_Pooling_output = tf.concat(RoiPooling_outputs_List, axis=0) # [resize_RoI, resize_RoI, resize_RoI]...리스트를 하나의 텐서로 통합. 밑에 붙히는 방식으로 쫘라락 붙힌다

        final_Pooling_output = tf.reshape(final_Pooling_output, (1, len(RoI_inImage), self.pool_size, self.pool_size, self.nb_channels)) # 통합한걸 (1,RoI 개수,7,7,512)로 reshape

        return final_Pooling_output # (1,RoI 개수,7,7,512) 텐서를 반환
    
    def get_config(self): # 구성요소 반환
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Detector(tf.keras.Model):
    def __init__(self, initializer, regularizer, SharedConvNet):
        super(Detector, self).__init__(name='Detector')
        # 레이어 
        # 공용 레이어(14*14*512)를 받아서 RoI Pooling Layer에 넣어 7*7*512를 만들고 FCs를 거쳐 두가지 Output을 생성한다. 

        # 클래스 분류 레이어와 박스 위치 회귀 레이어의 초기화를 다르게 해야한다. 
        Classify_layer_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        Box_regression_layer_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001, seed=None)

        # 공용 컨볼루전 레이어들 추가
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

        # RoI Pooling : H*W(7*7)에 맞게 입력 특성맵을 pooling. RoI에 해당하는 영역을 7*7로 Pooling한다. 
        self.RoI_Pooling_Layer = RoiPoolingLayer(7) # Pooling 이후 크기를 7*7*512로 만든다. -> (1,num_roi,7,7,512)
        self.Flatten_layer = tf.keras.layers.Flatten() # num_roi*7*7*512개의 텐서가 일렬로 나열됨
        self.Fully_Connected = tf.keras.layers.Dense(4096, activation='relu') # 별 말이 없으니 기본적으로 지정된 kernel_initializer를 사용하자. 여기선 RoI별 [1, 7*7*512] 텐서를 넣는다.
        self.Classify_layer = tf.keras.layers.Dense(21, activation='softmax', kernel_initializer = Classify_layer_initializer, name = "output_1")
        self.Reg_layer = tf.keras.layers.Dense(84, activation= None, kernel_initializer = Box_regression_layer_initializer, name = "output_2")
    
    def call(self, Image, RoI_list): # input으로 (224,224,3)과 RoI를 받는다. RoI는 앞서 RPN에서 뽑아낸걸 쓴다. 

        output = self.conv1_1(Image)
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
        
        final_Pooling_output = self.RoI_Pooling_Layer(shared_output, RoI_list) # 공용 레이어와 RoI를 넣어 (1, RoI 개수, 7,7,512) 휙득

        flatten_ouput_forAllRoI = self.Flatten_layer(final_Pooling_output) # Flatten으로 한줄 세우기 = (7*7*512) * RoI 개수

        flatten_perRoI = tf.split(flatten_ouput_forAllRoI, num_or_size_splits = len(RoI_list)) # (1, 7*7*512)텐서가 모인 리스트로 만들었다.

        Classify_layer_output = []
        Reg_layer_output = []

        for i in range(0, len(flatten_perRoI)):
            flatten_output = flatten_perRoI[i] # flatten된걸 하나씩 꺼냄
            Fully_Connected_output = self.Fully_Connected(flatten_output) # FCs로 만들기
            # 객체 분류 레이어, 박스 회귀 레이어
            cls_output = self.Classify_layer(Fully_Connected_output) 
            reg_output = self.Reg_layer(Fully_Connected_output)

            Classify_layer_output.append(cls_output)
            Reg_layer_output.append(reg_output)

        return Classify_layer_output, Reg_layer_output # Classify_layer_output : [1,21] 텐서가 len(RoI_list)개 모인 리스트, Reg_layer_output : [1, 84] 텐서가 len(RoI_list)개 모인 리스트

    # 필요한거 : multi task loss, gradient 계산, 적용
    def multi_task_loss(self, image, RoI_list, Ground_Truth_Box_list, Cls_label_list):
        # image는 (1,224,224,3), RoI_list는 (x,y,w,h)양식인 박스들 64개, Ground_Truth_Box_list는 (x_min, y_min, x_max, y_max)인 애들이 64개, Cls_label_list는 (,21)인 원-핫 인코딩된 애들이 64개
        # 여기서는 64개의 RoI에 해당하는 Loss 64개를 구한다. 
        Classify_layer_output, Reg_layer_output = self.call(image, RoI_list) # tensor들이 64개씩 모인 리스트 2개 휙득. Reg_layer_output은 (x,y,w,h)인 텐서들이 21개 일렬로 있다 -> (1, 84) 텐서

        loss_list = []

        for i in range(0, 64) : # index 0~15는 IoU가 0.5이상인 RoI들, 16~63은 IoU가 0.1~0.49999...인 RoI들
            # 각 RoI별 리스트 하나씩 꺼냄
            cls_output = Classify_layer_output[i]
            reg_output = Reg_layer_output[i]
            # 라벨값도 하나씩 꺼냄
            ground_truth_box = Ground_Truth_Box_list[i]
            cls_label = Cls_label_list[i]

            # loss 계산
            cls_loss = tf.nn.softmax_cross_entropy_with_logits(labels=cls_label, logits=cls_output)

            reg_loss = 0
            if i < 16:
                # (1,84)에서 해당 클래스에 해당하는 값을 얻어야한다(예 : '자동차'객체에 대한 박스 위치 추측값)
                # 논문에서 (x,y,w,h)에 대한 smooth l1을 구하라길래 ground_truth_box를 (x,y,w,h)로 바꿔주고자 한다
                gtb = tf.constant([ground_truth_box[0] + ground_truth_box[2]/2, ground_truth_box[1] + ground_truth_box[3]/2, ground_truth_box[2] - ground_truth_box[0], ground_truth_box[3] - ground_truth_box[1]])
                class_index = tf.argmax(cls_label) # 라벨값의 원-핫 인코딩에서 가장 큰 값의 인덱스 = 클래스의 인덱스에 해당. 
                pred_box = reg_output[4*class_index:4*class_index + 4] # 예측값에서 해당 클래스에 해당되는 박스 좌표를 불러온다. 
                reg_loss = tf.compat.v1.losses.huber_loss(gtb, pred_box) # (x,y,w,h) 각 성분에 대해 smoothL1(ti −vi)한 값을 다 더한게 나온다. 
        
            loss = tf.add(cls_loss, reg_loss)
            loss_list.append(loss) # loss list에 loss를 넣는다.

        return loss_list # 64개의 loss로 이루어진 리스트를 반환

    def get_grad(self, Loss, cls_reg_boolean): # Loss 하나씩 그래디언트 구하기
        g = 0
        with tf.GradientTape() as tape:
            tape.watch(self.conv1_1.variables)
            tape.watch(self.conv1_2.variables)
            tape.watch(self.conv2_1.variables)
            tape.watch(self.conv2_2.variables)
            tape.watch(self.conv3_1.variables)
            tape.watch(self.conv3_2.variables)
            tape.watch(self.conv3_3.variables)
            tape.watch(self.conv4_1.variables)
            tape.watch(self.conv4_2.variables)
            tape.watch(self.conv4_3.variables)
            tape.watch(self.conv5_1.variables)
            tape.watch(self.conv5_2.variables)
            tape.watch(self.conv5_3.variables)
            tape.watch(self.Fully_Connected.variables)

            if cls_reg_boolean == 0:
                tape.watch(self.Classify_layer.variables)
            else:
                tape.watch(self.Reg_layer.variables)

            
            if cls_reg_boolean == 0:
                g = tape.gradient(Loss, [self.conv1_1.variables[0], self.conv1_1.variables[1],self.conv1_2.variables[0], self.conv1_2.variables[1],self.conv2_1.variables[0], self.conv2_1.variables[1],self.conv2_2.variables[0], self.conv2_2.variables[1], self.conv3_1.variables[0], self.conv3_1.variables[1], self.conv3_2.variables[0],self.conv3_2.variables[1], self.conv3_3.variables[0],self.conv3_3.variables[1], self.conv4_1.variables[0],self.conv4_1.variables[1], self.conv4_2.variables[0],self.conv4_2.variables[1], self.conv4_3.variables[0],self.conv4_3.variables[1], self.conv5_1.variables[0],self.conv5_2.variables[1], self.conv5_3.variables[0],self.conv5_3.variables[1], self.Fully_Connected.variables[0],self.Fully_Connected.variables[1], self.Classify_layer.variables[0],self.Classify_layer.variables[1]])
            else:
                g = tape.gradient(Loss, [self.conv1_1.variables[0], self.conv1_1.variables[1],self.conv1_2.variables[0], self.conv1_2.variables[1],self.conv2_1.variables[0], self.conv2_1.variables[1],self.conv2_2.variables[0], self.conv2_2.variables[1], self.conv3_1.variables[0], self.conv3_1.variables[1], self.conv3_2.variables[0],self.conv3_2.variables[1], self.conv3_3.variables[0],self.conv3_3.variables[1], self.conv4_1.variables[0],self.conv4_1.variables[1], self.conv4_2.variables[0],self.conv4_2.variables[1], self.conv4_3.variables[0],self.conv4_3.variables[1], self.conv5_1.variables[0],self.conv5_2.variables[1], self.conv5_3.variables[0],self.conv5_3.variables[1], self.Fully_Connected.variables[0],self.Fully_Connected.variables[1], self.Reg_layer.variables[0],self.Reg_layer.variables[1]])

        return g

    def App_Gradient(self, Loss_list, training_step) : # 64개의 loss로 이루어진 리스트를 받는다. training_step은 2 아니면 4다. 
        if training_step == 2:
            grad_acc_cls = tf.Variable(0.0)
            grad_acc_reg = tf.Variable(0.0)
            for i in range (0, 64) :
                Loss = Loss_list[i]
                # Detector는 로스 하나하나 적용함
                g_cls = self.get_grad(Loss, 0)
                self.Optimizers.apply_gradients(zip(g_cls, [self.Fully_Connected.variables[0],self.Fully_Connected.variables[1], self.Classify_layer.variables[0],self.Classify_layer.variables[1]]))
                g_reg = self.get_grad(Loss, training_step, 1)
                self.Optimizers.apply_gradients(zip(g_reg, [self.Fully_Connected.variables[0],self.Fully_Connected.variables[1], self.Classify_layer.variables[0],self.Reg_layer.variables[1]]))
                
                # grad를 다 더해서 VGG16을 훈련시킬거다.
                grad_acc_cls = tf.add(grad_acc_cls, g_cls)
                grad_acc_reg = tf.add(grad_acc_cls, g_reg)

            # 한 이미지에 대한 전체 grad를 얻는다. 
            total_grad = tf.add(grad_acc_cls, grad_acc_reg)
            # vgg16 훈련(conv3_1부터 conv5_3까지)
            self.Optimizers.apply_gradients(zip(total_grad, [self.conv3_1.variables[0], self.conv3_1.variables[1], self.conv3_2.variables[0],self.conv3_2.variables[1], self.conv3_3.variables[0],self.conv3_3.variables[1], self.conv4_1.variables[0],self.conv4_1.variables[1], self.conv4_2.variables[0],self.conv4_2.variables[1], self.conv4_3.variables[0],self.conv4_3.variables[1], self.conv5_1.variables[0],self.conv5_2.variables[1], self.conv5_3.variables[0],self.conv5_3.variables[1]]))
            
        # 4번 째 단계에선 Detector만 훈련시키면 된다.
        if training_step == 4:
            for i in range (0, 64) :
                Loss = Loss_list[i]
                # Detector는 로스 하나하나 적용함
                g_cls = self.get_grad(Loss, 0)
                self.Optimizers.apply_gradients(zip(g_cls, [self.Fully_Connected.variables[0],self.Fully_Connected.variables[1], self.Classify_layer.variables[0],self.Classify_layer.variables[1]]))
                g_reg = self.get_grad(Loss, training_step, 1)
                self.Optimizers.apply_gradients(zip(g_reg, [self.Fully_Connected.variables[0],self.Fully_Connected.variables[1], self.Classify_layer.variables[0],self.Reg_layer.variables[1]]))

    def get_minibatch(self, RoI_list, Ground_Truth_Box_list, Cls_label_list): # 64개의 RoI 추출 + RoI에 맞는 라벨들 추출
        # Ground_Truth_Box_list : (x_min, y_min, x_max, y_max)
        RoI_object_presume_group = []
        Ground_Truth_Box_object_presume_group = []
        Cls_label_object_presume_group = []

        RoI_background_presume_group = []
        Ground_Truth_Box_background_presume_group = []
        Cls_label_background_presume_group = []

        for i in range(0, len(RoI_list)):
            # IoU가 0.5 이상인 그룹, 0.1~0.499999...인 그룹 두개로 나눔
            RoI_x_min = RoI_list[i][0] - RoI_list[i][2]/2
            RoI_y_min = RoI_list[i][1] - RoI_list[i][3]/2
            RoI_x_max = RoI_list[i][0] + RoI_list[i][2]/2
            RoI_y_max = RoI_list[i][1] + RoI_list[i][3]/2

            max_IoU = 0 # RoI가 가진 최대 IoU
            ground_truth_box_Highest_IoU = 0 # 어떤 ground truth box와 가장 높은 IoU를 이뤄냈나
            cls_Higest_IoU = 0 # ground_truth_box_Highest_IoU는 어떤 클래스의 ground truth box와 IoU가 가장 높았나. 원핫 인코딩 양식임

            # RoI의 IoU를 구한다.
            for j in range(0, len(Ground_Truth_Box_list)):

                ground_truth_box = Ground_Truth_Box_list[j]
                cls_label = Cls_label_list[j]

                InterSection_min_x = max(RoI_x_min, ground_truth_box[0])
                InterSection_min_y = max(RoI_y_min, ground_truth_box[1])

                InterSection_max_x = min(RoI_x_max, ground_truth_box[2])
                InterSection_max_y = min(RoI_y_max, ground_truth_box[3])

                InterSection_Area = 0

                if (InterSection_max_x - InterSection_min_x + 1) >= 0 and (InterSection_max_y - InterSection_min_y + 1) >= 0 :
                    InterSection_Area = (InterSection_max_x - InterSection_min_x + 1) * (InterSection_max_y - InterSection_min_y + 1)

                box1_area = RoI_list[i][2] * RoI_list[i][3]
                box2_area = (ground_truth_box[2] - ground_truth_box[0]) * (ground_truth_box[3] - ground_truth_box[1])
                Union_Area = box1_area + box2_area - InterSection_Area

                IoU = (InterSection_Area/Union_Area)
                if IoU > max_IoU :
                    max_IoU = IoU
                    ground_truth_box_Highest_IoU = ground_truth_box
                    cls_Higest_IoU = cls_label

            # 두 그룹(IoU가 0.5이상인 애들, IoU가 0.1~0.499...인 애들)으로 나눔
            if max_IoU >= 0.5 :
                RoI_object_presume_group.append(RoI_list[i])
                Ground_Truth_Box_object_presume_group = ground_truth_box_Highest_IoU
                Cls_label_object_presume_group = cls_Higest_IoU

            elif max_IoU >= 0.1 and max_IoU < 0.5 :
                RoI_background_presume_group.append(RoI_list[i])
                Ground_Truth_Box_background_presume_group.append(ground_truth_box_Highest_IoU)
                Cls_label_background_presume_group.append(cls_Higest_IoU)

        # 나눠진 애들 중 각각 16, 32개를 선별
        # 인덱스를 랜덤으로 각각 16, 32개 선발
        RoI_minibatch = random.choice(RoI_object_presume_group, 16)
        Ground_Truth_Box_minibatch = random.choice(Ground_Truth_Box_object_presume_group, 16)
        Cls_label_minibatch = random.choice(Cls_label_object_presume_group, 16)

        RoI_minibatch.extend(random.choice(RoI_background_presume_group, 32))
        Ground_Truth_Box_minibatch.extend(random.choice(Ground_Truth_Box_background_presume_group, 32))
        Cls_label_minibatch.extend(random.choice(Cls_label_background_presume_group, 32))

        return RoI_minibatch, Ground_Truth_Box_minibatch, Cls_label_minibatch


    def Training_model(self, image_list, RoI_list_forAllImage, Reg_labels_for_FastRCNN, Cls_labels_for_FastRCNN, training_step):
        for i in tqdm(range(0, len(image_list)), desc = "training"):
            image = np.expand_dims(image_list[i], axis = 0)
            RoI_minibatch, Ground_Truth_Box_minibatch, Cls_label_minibatch = self.get_minibatch(self, RoI_list_forAllImage, Reg_labels_for_FastRCNN, Cls_labels_for_FastRCNN) # 이미지 당 64개의 미니배치 선별 (128/2 = 64)
            
            Loss_list = self.multi_task_loss(image, RoI_minibatch, Ground_Truth_Box_minibatch, Cls_label_minibatch) # RoI 64개에 대한 64개의 loss로 이루어진 리스트 반환
            self.App_Gradient(Loss_list, training_step)