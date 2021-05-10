# 모듈 import. 모듈 모아놓은 폴더에 넣거나 sys.path.append()로 모듈을 모아놓을 폴더 경로를 추가하고 거기에 모듈을 넣으면 된다. 
from RPN import *
from RoI_Pooling_Layer_and_Fast_RCNN import *
from util import *

import tensorflow as tf
from glob import glob

# 데이터셋 생성 
train_x_path = '/Users/minguinho/Documents/AI_Datasets/PASCAL_VOC_2007/train/VOCdevkit/VOC2007/JPEGImages'
train_y_path = '/Users/minguinho/Documents/AI_Datasets/PASCAL_VOC_2007/train/VOCdevkit/VOC2007/Annotations'

image_file_list = sorted([x for x in glob.glob(train_x_path + '/**')])
xml_file_list = sorted([x for x in glob.glob(train_y_path + '/**')])

anchor_size = [32, 64, 128] # 이미지 크기가 224*224라 32, 64, 128로 지정
anchor_aspect_ratio = [[1,1],[1,0.5], [0.5,1]] # W*L기준 
anchors, anchors_state = util.make_anchor(anchor_size, anchor_aspect_ratio) # 앵커 생성 + 유효한 앵커 인덱스 휙득

image_list, cls_layer_label_list, reg_layer_label_list = util.make_dataset_forRPN([image_file_list, xml_file_list, anchors, anchors_state])
Classes_inDataSet = util.get_Classes_inImage(xml_file_list)

max_num = len(tf.keras.applications.VGG16(weights='imagenet', include_top=False,  input_shape=(224, 224, 3)).layers) # 레이어 최대 개수

SharedConvNet = tf.keras.models.Sequential()
for i in range(0, max_num-1):
    SharedConvNet.add(tf.keras.applications.VGG16(weights='imagenet', include_top=False,  input_shape=(224, 224, 3)).layers[i])

initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
regularizer = tf.keras.regularizers.l2(0.0005)

for layer in SharedConvNet.layers:
    # 'kernel_regularizer' 속성이 있는 인스턴스를 찾아 regularizer를 추가
    if hasattr(layer, 'kernel_regularizer'):
        setattr(layer, 'kernel_regularizer', regularizer)

RPN_Model = RPN.RPN(initializer, regularizer, SharedConvNet, anchors, anchor_size, anchor_aspect_ratio)
Detector_Model = RoI_Pooling_Layer_and_Fast_RCNN.Detector(initializer, regularizer, SharedConvNet)

# 훈련
RPN_Model, Detector_Model = util.four_Step_Alternating_Training(RPN_Model, Detector_Model, image_list, cls_layer_label_list, reg_layer_label_list, Classes_inDataSet, EPOCH = 100)
# 출력
util.get_FasterRCNN_output("이미지 경로")




