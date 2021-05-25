import numpy as np
import tensorflow as tf
import random
from tqdm import tqdm

# RoI Pooling Layer + Fast RCNN

# Pooling 작업을 RoI 지역에 대해 한다. 14*14*512를 7*7*512로 만든다
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

            RoI_inImage_np = RoI_inImage[i].numpy()
            
            r = RoI_inImage_np[0] - RoI_inImage_np[2]/2
            c = RoI_inImage_np[1] - RoI_inImage_np[3]/2
            w = RoI_inImage_np[2]
            h = RoI_inImage_np[3]
            
            # 1/16배로 만들기
            r = int(round(r / 16.0))
            c = int(round(c / 16.0))
            w = int(round(w / 16.0))
            h = int(round(h / 16.0))

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
    def __init__(self, SharedConvNet):
        super(Detector, self).__init__(name='Detector')
        
        self.Optimizers = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9)
        
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
    
    def call(self, Image, RoI_list): 

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
        
        final_Pooling_output_forAllRoI = self.Flatten_layer(final_Pooling_output) # Flatten으로 한줄 세우기. [1, (7*7*512) * RoI 개수] 텐서가 나옴
        
        flatten_perRoI = tf.split(final_Pooling_output_forAllRoI, num_or_size_splits = len(RoI_list), axis = 1) # (1, 7*7*512)텐서가 모인 리스트로 만들었다.
        
        Classify_layer_output_list = []
        Reg_layer_output_list = []

        for i in range(0, len(flatten_perRoI)):
            flatten_output = flatten_perRoI[i] # flatten된걸 하나씩 꺼냄
            
            Fully_Connected_output = self.Fully_Connected(flatten_output) # FCs로 만들기
            # 객체 분류 레이어, 박스 회귀 레이어
            cls_output = self.Classify_layer(Fully_Connected_output) 
            reg_output = self.Reg_layer(Fully_Connected_output)

            Classify_layer_output_list.append(cls_output)
            Reg_layer_output_list.append(reg_output)
        
           
        return Classify_layer_output_list, Reg_layer_output_list

    def get_minibatch(self, nonCrossBoundary_RoI, Reg_label, Cls_label): 
        # 로스를 계산하기 전에 입력받은 데이터에서 64개씩 추출. Fast R-CNN 에서 했듯이 한다
        # Ground_Truth_Box_list : (x_min, y_min, x_max, y_max)
        RoI_object_presume_group = []
        Ground_Truth_Box_object_presume_group = []
        Cls_label_object_presume_group = []

        RoI_background_presume_group = []
        Ground_Truth_Box_background_presume_group = []
        Cls_label_background_presume_group = []

        for i in range(0, len(nonCrossBoundary_RoI)):
            # IoU가 0.5 이상인 그룹, 0.1~0.499999...인 그룹 두개로 나눔
            RoI_x_min = nonCrossBoundary_RoI[i][0] - nonCrossBoundary_RoI[i][2]/2
            RoI_y_min = nonCrossBoundary_RoI[i][1] - nonCrossBoundary_RoI[i][3]/2
            RoI_x_max = nonCrossBoundary_RoI[i][0] + nonCrossBoundary_RoI[i][2]/2
            RoI_y_max = nonCrossBoundary_RoI[i][1] + nonCrossBoundary_RoI[i][3]/2

            max_IoU = 0 # RoI가 가진 최대 IoU
            ground_truth_box_Highest_IoU = 0 # 어떤 ground truth box와 가장 높은 IoU를 이뤄냈나
            cls_Higest_IoU = 0 # ground_truth_box_Highest_IoU는 어떤 클래스의 ground truth box와 IoU가 가장 높았나. 원핫 인코딩 양식임

            # RoI의 IoU를 구한다.
            for j in range(0, len(Reg_label)): # Reg_labels은 각 RoI가 어떤 Ground Truth Box와 연관 있는지 나타낸다.

                ground_truth_box = Reg_label[j]
                cls_label = Cls_label[j]

                InterSection_min_x = max(RoI_x_min, ground_truth_box[0])
                InterSection_min_y = max(RoI_y_min, ground_truth_box[1])

                InterSection_max_x = min(RoI_x_max, ground_truth_box[2])
                InterSection_max_y = min(RoI_y_max, ground_truth_box[3])

                InterSection_Area = 0

                if (InterSection_max_x - InterSection_min_x + 1) >= 0 and (InterSection_max_y - InterSection_min_y + 1) >= 0 :
                    InterSection_Area = (InterSection_max_x - InterSection_min_x + 1) * (InterSection_max_y - InterSection_min_y + 1)

                box1_area = nonCrossBoundary_RoI[i][2] * nonCrossBoundary_RoI[i][3]
                box2_area = (ground_truth_box[2] - ground_truth_box[0]) * (ground_truth_box[3] - ground_truth_box[1])
                Union_Area = box1_area + box2_area - InterSection_Area

                IoU = (InterSection_Area/Union_Area)
                if IoU > max_IoU :
                    max_IoU = IoU
                    ground_truth_box_Highest_IoU = ground_truth_box
                    cls_Higest_IoU = cls_label

            # 두 그룹(IoU가 0.5이상인 애들, IoU가 0.1~0.499...인 애들)으로 나눔
            if max_IoU >= 0.5 :
                RoI_object_presume_group.append(nonCrossBoundary_RoI[i])
                Ground_Truth_Box_object_presume_group.append(ground_truth_box_Highest_IoU)
                Cls_label_object_presume_group.append(cls_Higest_IoU)

            elif max_IoU >= 0.1 and max_IoU < 0.5 :
                RoI_background_presume_group.append(nonCrossBoundary_RoI[i])
                Ground_Truth_Box_background_presume_group.append(ground_truth_box_Highest_IoU)
                Cls_label_background_presume_group.append(cls_Higest_IoU)
                
        # 나눠진 애들 중 각각 16, 48개를 선별
        # 인덱스를 랜덤으로 각각 16, 48개 선발. 만약 IoU가 0.5 이상인게 16개보다 작으면 부족한 부분을 다른 그룹에서 가져오기
        
        obj_RoI_num = min([16, len(RoI_object_presume_group)])
        
        if obj_RoI_num != 0:
            RoI_minibatch = random.sample(RoI_object_presume_group, obj_RoI_num)
            Reg_label_minibatch = random.sample(Ground_Truth_Box_object_presume_group, obj_RoI_num)
            Cls_label_minibatch = random.sample(Cls_label_object_presume_group, obj_RoI_num)
            
            obj_Bgr_num = min([64 - obj_RoI_num, len(RoI_background_presume_group)])

            RoI_minibatch.extend(random.sample(RoI_background_presume_group, obj_Bgr_num))
            Reg_label_minibatch.extend(random.sample(Ground_Truth_Box_background_presume_group, obj_Bgr_num))
            Cls_label_minibatch.extend(random.sample(Cls_label_background_presume_group, obj_Bgr_num))
        else :
            
            obj_Bgr_num = min([64, len(RoI_background_presume_group)])
            
            RoI_minibatch = random.sample(RoI_background_presume_group, obj_Bgr_num)
            Reg_label_minibatch = random.sample(Ground_Truth_Box_background_presume_group, obj_Bgr_num)
            Cls_label_minibatch = random.sample(Cls_label_background_presume_group, obj_Bgr_num)

        return RoI_minibatch, Reg_label_minibatch, Cls_label_minibatch, obj_RoI_num # 어느 구간부터 RoI 종류가 갈리는지
    
    def multi_task_loss(self, Classify_layer_output_list, Reg_layer_output_list, Cls_label_minibatch, Reg_label_minibatch, obj_RoI_num):
        
        Loss_final = 0
        
        for i in range(0, len(Reg_label_minibatch)) :
            cls_loss = tf.nn.softmax_cross_entropy_with_logits(labels = Cls_label_minibatch[i], logits=Classify_layer_output_list[i])
            loss = cls_loss # cls_loss가 loss에 들어간다는 걸 나타내기 위해 이 코드를 추가함

            if i < obj_RoI_num: # IoU > 0.5인 RoI들
                # (1,84)에서 해당 클래스에 해당하는 값을 얻어야한다(예 : '자동차'객체에 대한 박스 위치 추측값)
                # 논문에서 (x,y,w,h)에 대한 smooth l1을 구하라길래 ground_truth_box를 (x,y,w,h)로 바꿔주고자 한다
                Reg_label = Reg_label_minibatch[i]
                
                class_index = tf.argmax(Cls_label_minibatch[i]) # 라벨값의 원-핫 인코딩에서 가장 큰 값의 인덱스 = 클래스의 인덱스에 해당. 
                
                label_forCalc = np.zeros((1, 84))
                
                label_forCalc[0][4*class_index]     = Reg_label[0] + Reg_label[2]/2
                label_forCalc[0][4*class_index + 1] = Reg_label[1] + Reg_label[3]/2
                label_forCalc[0][4*class_index + 2] = Reg_label[2] - Reg_label[0]
                label_forCalc[0][4*class_index + 3] = Reg_label[3] - Reg_label[1]
                
                label_forCalc_tf = tf.convert_to_tensor(label_forCalc, dtype=tf.float32)
                
                filter_for_predbox = np.zeros((1, 84))
                
                filter_for_predbox[0][4*class_index]     = 1.0
                filter_for_predbox[0][4*class_index + 1] = 1.0
                filter_for_predbox[0][4*class_index + 2] = 1.0
                filter_for_predbox[0][4*class_index + 3] = 1.0
                
                filter_for_predbox_tf = tf.convert_to_tensor(filter_for_predbox, dtype=tf.float32)
                
                Reg_layer_output_list[i] = tf.math.multiply(Reg_layer_output_list[i], filter_for_predbox_tf)
                
                reg_loss = tf.compat.v1.losses.huber_loss(label_forCalc_tf, Reg_layer_output_list[i]) # (x,y,w,h) 각 성분에 대해 smoothL1(ti −vi)한 값을 다 더한게 나온다. 

                loss = tf.add(cls_loss, reg_loss)
                
            if i == 0:
                Loss_final = loss
            else :
                Loss_final = tf.add(Loss_final, loss)
        
        N_batch = tf.constant([1.0/len(Reg_label_minibatch)])
        Loss_mean = tf.multiply(Loss_final, N_batch)
        
        return Loss_mean # 64개 미니배치로 얻은 로스의 평균값
                
    
    def get_grad(self, image, RoI_minibatch, Reg_label_minibatch, Cls_label_minibatch, obj_RoI_num, training_step): 
        
        with tf.GradientTape() as tape:
            Loss_mean = 0
            g = 0
            # 매번 g를 구할 때마다 tape.watch~ 과정을 거쳐야 한다
            if training_step == 2 : # conv3_1 ~ 끝까지 훈련
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
                
                tape.watch(self.Classify_layer.variables)
                tape.watch(self.Reg_layer.variables)
                    
                Classify_layer_output_list, Reg_layer_output_list = self.call(image, RoI_minibatch) # 출력값을 얻어보자
                
                Loss_mean = self.multi_task_loss(Classify_layer_output_list, Reg_layer_output_list, Cls_label_minibatch, Reg_label_minibatch, obj_RoI_num)

                g = tape.gradient(Loss_mean, [self.conv3_1.variables[0], self.conv3_1.variables[1], 
                                              self.conv3_2.variables[0], self.conv3_2.variables[1], 
                                              self.conv3_3.variables[0], self.conv3_3.variables[1], 
                                              self.conv4_1.variables[0], self.conv4_1.variables[1], 
                                              self.conv4_2.variables[0], self.conv4_2.variables[1], 
                                              self.conv4_3.variables[0], self.conv4_3.variables[1], 
                                              self.conv5_1.variables[0], self.conv5_1.variables[1], 
                                              self.conv5_2.variables[0], self.conv5_2.variables[1], 
                                              self.conv5_3.variables[0], self.conv5_3.variables[1], 
                                              self.Fully_Connected.variables[0], self.Fully_Connected.variables[1], 
                                              self.Classify_layer.variables[0], self.Classify_layer.variables[1], 
                                              self.Reg_layer.variables[0], self.Reg_layer.variables[1]])
                

            elif training_step == 4 : # Detector만 훈련
                tape.watch(self.Fully_Connected.variables)
                tape.watch(self.Classify_layer.variables)
                tape.watch(self.Reg_layer.variables)
                    
                Classify_layer_output_list, Reg_layer_output_list = self.call(image, RoI_minibatch) # 출력값을 얻어보자
                    
                Loss_mean = self.multi_task_loss(Classify_layer_output_list, Reg_layer_output_list, Cls_label_minibatch, Reg_label_minibatch)
                
                g = tape.gradient(Loss_mean, [self.Fully_Connected.variables[0],self.Fully_Connected.variables[1], 
                                              self.Classify_layer.variables[0],self.Classify_layer.variables[1], 
                                              self.Reg_layer.variables[0],self.Reg_layer.variables[1]])
            
            minibatch_num_np = np.asarray([len(Cls_label_minibatch)])
            
            minibatch_num_tensor = tf.convert_to_tensor(minibatch_num_np, dtype=tf.float32)
        
            return g, Loss_mean, minibatch_num_tensor
    
    def App_Gradient(self, training_step, image, nonCrossBoundary_RoI, Reg_label, Cls_label) :
        
        #미니배치 구하기
        RoI_minibatch, Reg_label_minibatch, Cls_label_minibatch, obj_RoI_num = self.get_minibatch(nonCrossBoundary_RoI, Reg_label, Cls_label) # 이미지 당 64개의 미니배치 선별 (128/2 = 64) 
        
        if len(RoI_minibatch) == 0:
            return tf.constant([0.0])
        
        g, Loss_mean, minibatch_num = self.get_grad(image, RoI_minibatch, Reg_label_minibatch, Cls_label_minibatch, obj_RoI_num, training_step)
        
        if training_step == 2:
            # Detector 훈련
            
            self.Optimizers.apply_gradients(zip(g[-6:], [self.Fully_Connected.variables[0],self.Fully_Connected.variables[1], 
                                                        self.Classify_layer.variables[0],self.Classify_layer.variables[1], 
                                                        self.Reg_layer.variables[0],self.Reg_layer.variables[1]]))
                
            # 모인 그래디언트로 VGG16의 conv3_1부터 5_3까지 훈련시킨다.
            # 평균 * 미니배치 개수 = 전체 loss
            for i in range(0, len(g) - 6) :
                g[i] = tf.multiply(g[i], minibatch_num)
            
            self.Optimizers.apply_gradients(zip(g[:-6], [self.conv3_1.variables[0], self.conv3_1.variables[1], 
                                                        self.conv3_2.variables[0],self.conv3_2.variables[1], 
                                                        self.conv3_3.variables[0],self.conv3_3.variables[1], 
                                                        self.conv4_1.variables[0],self.conv4_1.variables[1], 
                                                        self.conv4_2.variables[0],self.conv4_2.variables[1], 
                                                        self.conv4_3.variables[0],self.conv4_3.variables[1], 
                                                        self.conv5_1.variables[0],self.conv5_1.variables[1], 
                                                        self.conv5_2.variables[0],self.conv5_2.variables[1],
                                                        self.conv5_3.variables[0],self.conv5_3.variables[1]]))

        elif training_step == 4:
            self.Optimizers.apply_gradients(zip(g[-6:], [self.Fully_Connected.variables[0],self.Fully_Connected.variables[1], 
                                                        self.Classify_layer.variables[0],self.Classify_layer.variables[1], 
                                                        self.Reg_layer.variables[0],self.Reg_layer.variables[1]]))
        
        return Loss_mean
        
    def Training_model(self, image_list, nonCrossBoundary_RoI_forAll_Image, Reg_labels_for_FastRCNN, Cls_labels_for_FastRCNN, training_step):
        bar = tqdm(range(0, len(image_list) ))
        loss_acc = 0.0
        for i in bar :
            image = np.expand_dims(image_list[i], axis = 0) # (1,224,224,3)으로 제작
            
            if len(nonCrossBoundary_RoI_forAll_Image[i]) == 0:
                continue
            
            Loss_mean = self.App_Gradient(training_step, image, nonCrossBoundary_RoI_forAll_Image[i], Reg_labels_for_FastRCNN[i], Cls_labels_for_FastRCNN[i])    
            
            if len(nonCrossBoundary_RoI_forAll_Image[i]) != 0:
                loss_acc = loss_acc + (Loss_mean.numpy()).item()
                avr_loss = loss_acc / (float)(i + 1)
                desc_str = "training Fast RCNN, Loss = %f " % avr_loss
                bar.set_description(desc_str)