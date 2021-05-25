# 모델 데이터셋 생성, 훈련에 필요한 함수들을 여기다 모아놨다. 

# 임포트
import tensorflow as tf
import numpy as np
import copy
import cv2
import xmltodict
from tqdm import tqdm

# 입력용 이미지 생성. 224, 224로 변환시키고 채널 값(0~255)를 0~1 사이의 값으로 정규화 시켜줌
def make_input(image_file_list): 
    images_list = []
    
    for i in tqdm(range(0, len(image_file_list)), desc="get image") :
        
        image = cv2.imread(image_file_list[i])
        image = cv2.resize(image, (224, 224))/255
        
        images_list.append(image)
    
    return np.asarray(images_list)

def make_anchor(anchor_size, anchor_aspect_ratio) :
    
    anchors = [] # [x,y,w,h]로 이루어진 리스트 
    anchors_state = [] # 이 앵커를 훈련에 쓸건가? 각 앵커별로 사용 여부를 나타낸다. 

    # 앵커 중심좌표 간격
    interval_x = 16
    interval_y = 16

    # 2단 while문 생성
    x = 8
    y = 8
    while(y <= 224): # 8~208 = 14개 
        while(x <= 224): # 8~208 = 14개 
            # k개의 앵커 생성. 여기서 k = len(anchor_size) * len(anchor_aspect_ratio)다
            for i in range(0, len(anchor_size)) : 
                for j in range(0, len(anchor_aspect_ratio)) :
                    anchor_width = anchor_aspect_ratio[j][0] * anchor_size[i]
                    anchor_height = anchor_aspect_ratio[j][1] * anchor_size[i]

                    anchor = [x, y, anchor_width, anchor_height]
                    anchors.append(anchor)
                    anchors_state.append(1)
                    
            x = x + interval_x 
        y = y + interval_y
        x = 8
    return anchors, anchors_state

# 이미지에 어떤 Ground Truth Box가 있는지 + 이미지 크기에 맞춰 앵커 크기도 변형
def get_Ground_Truth_Box_fromImage(xml_file_path, anchors): # xml_file_path은 파일 하나의 경로를 나타낸다

    f = open(xml_file_path)
    xml_file = xmltodict.parse(f.read()) 

    # 우선 원래 이미지 크기를 얻는다. 왜냐하면 앵커는 224*224 기준으로 만들었는데 원본 이미지는 224*224가 아니기 때문.
    # 224*224에 맞게 줄일려고 하는거다
    Image_Height = float(xml_file['annotation']['size']['height'])
    Image_Width  = float(xml_file['annotation']['size']['width'])

    Ground_Truth_Box_list = [] 
    
    # 원래 이미지에 128, 256, 512 앵커를 갖다대며 비교하는데 이게 224*224로 변형 -> 줄어든 비율만큼 앵커도 변형
    anchors_forImage = copy.deepcopy(anchors)
    for i in range(0, len(anchors)):
        # 크기만 변형. 좌표는 이미 특성맵(14*14)에 최적화 되어있음
        anchors_forImage[i][2] = anchors_forImage[i][2] * (224/Image_Width)
        anchors_forImage[i][3] = anchors_forImage[i][3] * (224/Image_Height)
        
    # multi-objects in image
    try:
        for obj in xml_file['annotation']['object']:
            
            # 박스 좌표(왼쪽 위, 오른쪽 아래) 얻기
            x_min = float(obj['bndbox']['xmin']) 
            y_min = float(obj['bndbox']['ymin'])
            x_max = float(obj['bndbox']['xmax']) 
            y_max = float(obj['bndbox']['ymax'])

            # 224*224에 맞게 변형시켜줌
            x_min = float((224/Image_Width)*x_min)
            y_min = float((224/Image_Height)*y_min)
            x_max = float((224/Image_Width)*x_max)
            y_max = float((224/Image_Height)*y_max)

            Ground_Truth_Box = [x_min, y_min, x_max, y_max]
            Ground_Truth_Box_list.append(Ground_Truth_Box)

    # single-object in image
    except TypeError as e : 
        # 박스 좌표(왼쪽 위, 오른쪽 아래) 얻기
        x_min = float(xml_file['annotation']['object']['bndbox']['xmin']) 
        y_min = float(xml_file['annotation']['object']['bndbox']['ymin']) 
        x_max = float(xml_file['annotation']['object']['bndbox']['xmax']) 
        y_max = float(xml_file['annotation']['object']['bndbox']['ymax']) 

        # 224*224에 맞게 변형시켜줌
        x_min = float((224/Image_Width)*x_min)
        y_min = float((224/Image_Height)*y_min)
        x_max = float((224/Image_Width)*x_max)
        y_max = float((224/Image_Height)*y_max)

        Ground_Truth_Box = [x_min, y_min, x_max, y_max]  
        Ground_Truth_Box_list.append(Ground_Truth_Box)

    
    Ground_Truth_Box_list = np.asarray(Ground_Truth_Box_list)
    Ground_Truth_Box_list = np.reshape(Ground_Truth_Box_list, (-1, 4))

    return Ground_Truth_Box_list, anchors_forImage # 이미지에 있는 Ground Truth Box 리스트, WH_ratio_list 받기(numpy)

# 앵커들을 Positive, Negative 앵커로 나누고 각 앵커가 참고한 Ground Truth Box와 Class를 반환하자
# RPN에는 '어떤 클래스인가?'는 알 필요가 없다. '객체인가 아닌가'이거 하나만 필요할 뿐. 
# 각 이미지에 맞게 최적화된 앵커를 입력값으로 받는다.
def align_anchor(anchors_for_Image, anchors_state, Ground_Truth_Box_list):

    # 각 앵커는 해당 위치에서 구한 여러가지 Ground truth Box와의 ioU 중 제일 높은거만 가져온다. 
    IoU_List = []
    Ground_truth_box_Highest_IoU_List = [] # 각 앵커가 어떤 Ground Truth Box를 보고 가장 높은 IoU를 계산했는가?

    for i in range(0, len(anchors_for_Image)):
        
        IoU_max = 0
        ground_truth_box_Highest_IoU = [0,0,0,0]
        
        anchor_for_thisImage = anchors_for_Image[i]
        
        anchor_minX = anchor_for_thisImage[0] - (anchor_for_thisImage[2]/2)
        anchor_minY = anchor_for_thisImage[1] - (anchor_for_thisImage[3]/2)
        anchor_maxX = anchor_for_thisImage[0] + (anchor_for_thisImage[2]/2)
        anchor_maxY = anchor_for_thisImage[1] + (anchor_for_thisImage[3]/2)
            
        anchor = [anchor_minX, anchor_minY, anchor_maxX, anchor_maxY]

        for j in range(0, len(Ground_Truth_Box_list)):
            
            ground_truth_box = Ground_Truth_Box_list[j]

            InterSection_min_x = max(anchor[0], ground_truth_box[0])
            InterSection_min_y = max(anchor[1], ground_truth_box[1])

            InterSection_max_x = min(anchor[2], ground_truth_box[2])
            InterSection_max_y = min(anchor[3], ground_truth_box[3])

            InterSection_Area = 0

            if (InterSection_max_x - InterSection_min_x + 1) >= 0 and (InterSection_max_y - InterSection_min_y + 1) >= 0 :
                InterSection_Area = (InterSection_max_x - InterSection_min_x + 1) * (InterSection_max_y - InterSection_min_y + 1)

            box1_area = (anchor[2] - anchor[0]) * (anchor[3] - anchor[1])
            box2_area = (ground_truth_box[2] - ground_truth_box[0]) * (ground_truth_box[3] - ground_truth_box[1])
            Union_Area = box1_area + box2_area - InterSection_Area

            IoU = (InterSection_Area/Union_Area)
            if IoU > IoU_max :
                IoU_max = IoU
                ground_truth_box_Highest_IoU = ground_truth_box

        IoU_List.append(IoU_max)
        Ground_truth_box_Highest_IoU_List.append(ground_truth_box_Highest_IoU)

        # 한 위치에 9개의 앵커 존재 -> 9개 앵커에 대한 IoU를 계산할 때마다 모아서 Positive, Negative 앵커 분류
        if i % 9 == 8 :
            IoU_List_inOneSpot = IoU_List[i-8:i+1]
            for num in list(range(i-8, i + 1)):
                if IoU_List[num] > 0.7 or (max(IoU_List_inOneSpot) == IoU_List[num] and IoU_List[num] >= 0.3): # positive anchor
                    anchors_state[num] = 2
                elif IoU_List[num] < 0.3 : # negative anchor
                    anchors_state[num] = 1
                else: # 애매한 앵커들
                    anchors_state[num] = 0     

    Ground_truth_box_Highest_IoU_List = np.asarray(Ground_truth_box_Highest_IoU_List)
    Ground_truth_box_Highest_IoU_List = np.reshape(Ground_truth_box_Highest_IoU_List, (-1, 4))
            
    return anchors_state, Ground_truth_box_Highest_IoU_List # 각 앵커의 상태, (모든)앵커가 IoU 계산에 참조한 Ground Truth Box


# RPN훈련을 위한 데이터셋. 
# RPN과 Detector는 별개의 모델이다. 즉, 두 모델을 훈련시킬 때 필요한 데이터셋은 따로따로 만들어야한다. 
# 여기선 RPN 훈련에 필요한 데이터만 만들거다. 
def make_dataset_forRPN(input_list) :
    image_file_list = input_list[0]
    xml_file_list = input_list[1]
    anchors = input_list[2]
    anchors_state = input_list[3]

    image_list = make_input(image_file_list) # 입력
    # 출력
    cls_layer_label_list = []
    reg_layer_label_list = []
    anchor_optimize_list_forAllImage = []

    # 값 계속 생성하는거 막기위한 변수
    cls_label_forPositive = [1.0, 0.0]
    cls_label_forNegative = [0.0, 1.0]
    cls_label_forUseless  = [0.5, 0.5]

    for i in tqdm(range(0, len(xml_file_list)), desc="get label"): # 각 이미지별로 데이터셋 생성(5011개)

        anchors_state_for = copy.deepcopy(anchors_state) # anchors_state는 매 사진마다 다르니까 원본값(?)을 복사해서 쓴다. 
        Ground_Truth_Box_list, anchors_forImage = get_Ground_Truth_Box_fromImage(xml_file_list[i], anchors) # 여기서는 Ground Truth Box에 대한 정보만 필요하다
        anchors_state_for, Ground_truth_box_Highest_IoU_List = align_anchor(anchors_forImage, anchors_state, Ground_Truth_Box_list)
        # 어떤 앵커가 Pos, neg 앵커인지, (모든)앵커가 참조한 ground truth box는 뭔지
    
        anchor_optimize_list_forAllImage.append(anchors_forImage) # 각 이미지에서 224*224로 줄어들 때 변하는 앵커 값을 반환한다
    
        for j in range(0, len(anchors_state_for)) :
            
            gtb_x = Ground_truth_box_Highest_IoU_List[j][0] + Ground_truth_box_Highest_IoU_List[j][2]/2
            gtb_y = Ground_truth_box_Highest_IoU_List[j][1] + Ground_truth_box_Highest_IoU_List[j][3]/2
            gtb_w = Ground_truth_box_Highest_IoU_List[j][2] - Ground_truth_box_Highest_IoU_List[j][0]
            gtb_h = Ground_truth_box_Highest_IoU_List[j][3] - Ground_truth_box_Highest_IoU_List[j][1]
            Ground_truth_box = [gtb_x, gtb_y, gtb_w, gtb_h]
            
            if anchors_state_for[j] == 2 : # positive
                # 라벨링을 위해 (min, max)를 (x,y,w,h)로
                cls_layer_label_list.append(cls_label_forPositive)
                reg_layer_label_list.append(Ground_truth_box) # IoU계산에 참조한(pos, neg 분류에 기여한) Ground Truth Box의 정보 휙득
            elif anchors_state_for[j] == 1 : # negative는 Ground Truth Box 정보가 필요없으니 [0,0,0,0]을 넣는다. 
                cls_layer_label_list.append(cls_label_forNegative) # 해당 앵커 output이 [0,1] -> negative
                reg_layer_label_list.append([0,0,0,0])
            else : 
                cls_layer_label_list.append(cls_label_forUseless) # 해당 앵커 output이 [0.5, 0.5] -> 무의미한 값
                reg_layer_label_list.append([0,0,0,0])

    # 논문에서 말한 출력값 크기에 맞게 reshape
    cls_layer_label_list = np.asarray(cls_layer_label_list)
    reg_layer_label_list = np.asarray(reg_layer_label_list)
    anchor_optimize_list_forAllImage = np.asarray(anchor_optimize_list_forAllImage)
    
    
    cls_layer_label_list = np.reshape(cls_layer_label_list, (-1, 1764, 2)) 
    reg_layer_label_list = np.reshape(reg_layer_label_list, (-1, 1764, 4))
    
    anchor_optimize_list_forAllImage = np.reshape(anchor_optimize_list_forAllImage, (-1, 1764, 4))

    image_list = image_list.astype('float32')

    return image_list, cls_layer_label_list, reg_layer_label_list, anchor_optimize_list_forAllImage # 훈련 데이터들(입, 출력)

# 데이터셋에 존재하는 클래스가 얼마나 있는지 알아낸다
def get_Classes_inImage(xml_file_list):
    Classes_inDataSet = []

    for xml_file_path in xml_file_list: 

        f = open(xml_file_path)
        xml_file = xmltodict.parse(f.read())
        # 사진에 객체가 여러개 있을 경우
        try: 
            for obj in xml_file['annotation']['object']:
                Classes_inDataSet.append(obj['name'].lower()) # 들어있는 객체 종류를 알아낸다
        # 사진에 객체가 하나만 있을 경우
        except TypeError as e: 
            Classes_inDataSet.append(xml_file['annotation']['object']['name'].lower()) 
        f.close()

    Classes_inDataSet = list(set(Classes_inDataSet)) # set은 중복된걸 다 제거하고 유니크한? 아무튼 하나만 가져온다. 그걸 리스트로 만든다
    Classes_inDataSet.sort() # 정렬

    return Classes_inDataSet

def get_mns_RoI(cls_output, reg_output):
    
    cls_output_np = cls_output.numpy()

    mns_RoI_list = []
    for i in range(0, len(cls_output_np)):
        if cls_output_np[i][0] > 0.7:
            mns_RoI_list.append(reg_output[i])
    
    return mns_RoI_list


def get_nonCrossBoundary_RoI(RPN_Model,image_list, anchor_optimize_list_forAllImage) :
    
    nonCrossBoundary_RoI_forAll_Image = []
    
    for i in tqdm(range(0, len(image_list)), desc = "get_nonCrossBoundary_RoI") :
        
        image = np.expand_dims(image_list[i], axis = 0)
        _, reg_output = RPN_Model(image, anchor_optimize_list_forAllImage[i])
        
        nonCrossBoundary_RoI_inImage = []
        
        for j in range(0, len(reg_output)) :
            x = reg_output[j][0]
            y = reg_output[j][1]
            w = reg_output[j][2]
            h = reg_output[j][3]
            
            if((x - (w/2) >= 0) and (y - (h/2) >= 0) and
            (x + (w/2) <= 224) and (y + (h/2) <= 224)):
                nonCrossBoundary_RoI_inImage.append(reg_output[j])
            
        nonCrossBoundary_RoI_forAll_Image.append(nonCrossBoundary_RoI_inImage)
        
    return nonCrossBoundary_RoI_forAll_Image

def make_DataSet_forFastRCNN_Train(xml_file_list, Classes_inDataSet, anchors):
    # 이미지별 GroundTruthBox도 반환한다    

    Cls_labels_for_FastRCNN = []
    Reg_labels_for_FastRCNN = []
    
    # 각 이미지에 해당하는 Ground Truth Box이에 맞는  Class 인덱스 추가
    for i in tqdm(range(0, len(xml_file_list)), desc = "get_dataset_forFAST_RCNN"):
        GroundTruthBoxes_inImage,_ = get_Ground_Truth_Box_fromImage(xml_file_list[i], anchors) # 이미지별 Ground Truth Box 리스트. (n, 4)크기의 리스트 받음

        classes = []
        f = open(xml_file_list[i])
        xml_file = xmltodict.parse(f.read())
        # 사진에 객체가 여러개 있을 경우
        try: 
            for obj in xml_file['annotation']['object']:
                classes.append(obj['name'].lower()) # 들어있는 객체 종류를 알아낸다
        # 사진에 객체가 하나만 있을 경우
        except TypeError as e: 
            classes.append(xml_file['annotation']['object']['name'].lower()) 

        # 한 이미지에서 얻은 클래스 리스트에서 각 클래스가 Classes_List 내에서 어떤 인덱스 번호를 갖고 있는지 알아낸다.
        # 그 인덱스 번호로 원-핫 인코딩 수행
        cls_index_list = []
        for class_val in classes :
            cls_index = Classes_inDataSet.index(class_val) # 클래스가 Classes_inDataSet 내에서 어떤 인덱스 번호를 갖고 있는가?
            cls_index_list.append(cls_index)# 한 이미지 내에 있는 Ground Truth Box별로 갖고 있는 클래스 인덱스를 저장
        cls_onehot_inImage = np.eye(len(Classes_inDataSet) + 1)[cls_index_list] # (n,21) 크기의 리스트 받음. 여기서 n은 한 이미지 내에 있는 객체 숫자

        # 저장
        Reg_labels_for_FastRCNN.append(GroundTruthBoxes_inImage)
        Cls_labels_for_FastRCNN.append(cls_onehot_inImage)

    return Reg_labels_for_FastRCNN, Cls_labels_for_FastRCNN # 이미지별 Ground Truth Box와 Classes 리스트


# 훈련
def four_Step_Alternating_Training(RPN_Model, Detector_Model, image_list, cls_layer_label_list, reg_layer_label_list, anchor_optimize_list_forAllImage, Reg_labels_for_FastRCNN, Cls_labels_for_FastRCNN, EPOCH): # 두 모델을 받아 훈련시킴
    # 각자 독립된 상태에서 훈련

    for i in range(0, EPOCH) : # RPN 훈련
        RPN_Model.Training_model(image_list, cls_layer_label_list, reg_layer_label_list, anchor_optimize_list_forAllImage,  1)

    # 훈련시킨 RPN에서 Detector훈련에 필요한 데이터 휙득
    nonCrossBoundary_RoI_forAll_Image = get_nonCrossBoundary_RoI(RPN_Model, image_list)# RoI는 경계선 넘지 않는 것들만
    
    for i in range(0, EPOCH) : # Detector 훈련
        Detector_Model.Training_model(image_list, nonCrossBoundary_RoI_forAll_Image, Reg_labels_for_FastRCNN, Cls_labels_for_FastRCNN, 2)

    # Detector_Model의 VGG를 RPN에 이식(레이어 공유 시작)
    RPN_Model.conv1_1 = Detector_Model.conv1_1
    RPN_Model.conv1_2 = Detector_Model.conv1_2
    RPN_Model.conv2_1 = Detector_Model.conv2_1
    RPN_Model.conv2_2 = Detector_Model.conv2_2
    RPN_Model.conv3_1 = Detector_Model.conv3_1
    RPN_Model.conv3_2 = Detector_Model.conv3_2
    RPN_Model.conv3_3 = Detector_Model.conv3_3
    RPN_Model.conv4_1 = Detector_Model.conv4_1
    RPN_Model.conv4_2 = Detector_Model.conv4_2
    RPN_Model.conv4_3 = Detector_Model.conv4_3
    RPN_Model.conv5_1 = Detector_Model.conv5_1
    RPN_Model.conv5_2 = Detector_Model.conv5_2
    RPN_Model.conv5_3 = Detector_Model.conv5_3

    for i in range(0, EPOCH) : # RPN 훈련
        RPN_Model.Training_model(image_list, cls_layer_label_list, reg_layer_label_list, anchor_optimize_list_forAllImage,  3)

    # 훈련시킨 RPN에서 Detector훈련에 필요한 데이터 휙득
    nonCrossBoundary_RoI_forAll_Image = get_nonCrossBoundary_RoI(RPN_Model, image_list)# RoI는 경계선 넘지 않는 것들만

    # RPN의 VGG16을 Detector의 VGG16 부분에 이식
    Detector_Model.conv1_1 = RPN_Model.conv1_1
    Detector_Model.conv1_2 = RPN_Model.conv1_2
    Detector_Model.conv2_1 = RPN_Model.conv2_1
    Detector_Model.conv2_2 = RPN_Model.conv2_2
    Detector_Model.conv3_1 = RPN_Model.conv3_1
    Detector_Model.conv3_2 = RPN_Model.conv3_2
    Detector_Model.conv3_3 = RPN_Model.conv3_3
    Detector_Model.conv4_1 = RPN_Model.conv4_1
    Detector_Model.conv4_2 = RPN_Model.conv4_2
    Detector_Model.conv4_3 = RPN_Model.conv4_3
    Detector_Model.conv5_1 = RPN_Model.conv5_1
    Detector_Model.conv5_2 = RPN_Model.conv5_2
    Detector_Model.conv5_3 = RPN_Model.conv5_3

    for i in range(0, EPOCH) : # Detector 훈련
        Detector_Model.Training_model(image_list, nonCrossBoundary_RoI_forAll_Image, Reg_labels_for_FastRCNN, Cls_labels_for_FastRCNN, 4)

    return RPN_Model, Detector_Model

# 값 출력
def get_FasterRCNN_output(RPN_Model, Detector_Model, Image, Classes_inDataSet, anchors) : # Image : 이미지 경로

    image_cv = cv2.imread(Image)
    height, width,_ = image_cv.shape # 이미지 원래 사이즈를 얻는다. [w, h]
    image_size = [width, height]

    image_cv = cv2.resize(image_cv, (224, 224))/255
    image_cv = np.expand_dims(image_cv, axis = 0)

    image_cv = image_cv.astype('float32')


    anchors_forImage = copy.deepcopy(anchors)
    for i in range(0, len(anchors)):
        # 크기만 변형. 좌표는 이미 특성맵(14*14)에 최적화 되어있음
        anchors_forImage[i][2] = anchors_forImage[i][2] * (224/image_size[0])
        anchors_forImage[i][3] = anchors_forImage[i][3] * (224/image_size[1])
            
    cls_output, reg_output = RPN_Model(image_cv, anchors_forImage) # (1764, 2), (1764, 4) 휙득

    mns_RoI_list = get_mns_RoI(cls_output, reg_output)

    Classify_layer_output_list, Reg_layer_output_list = Detector_Model(image_cv, mns_RoI_list)
    
    # 넘파이로 변환
    Classify_layer_output_list = Classify_layer_output_list.numpy()
    Reg_layer_output_list = Classify_layer_output_list.numpy()

    Classify_layer_output_list = np.reshape(Classify_layer_output_list, (-1, len(Classes_inDataSet)))
    Reg_layer_output_list = np.reshape(Reg_layer_output_list, (-1, 4*len(Classes_inDataSet)))

    im_read = cv2.imread(Image)

    for i in range(0, len(Reg_layer_output_list)) :
        
        class_index = np.argmax(Classify_layer_output_list[i]) # 라벨값의 원-핫 인코딩에서 가장 큰 값의 인덱스 = 클래스의 인덱스에 해당. 

        pred_box = [ 
        Reg_layer_output_list[i][4*class_index],
        Reg_layer_output_list[i][4*class_index + 1],
        Reg_layer_output_list[i][4*class_index + 2],
        Reg_layer_output_list[i][4*class_index + 3]]   

        
        # rectangle함수를 위해 필요한 '박스의 최소 x,y 좌표'와 '박스의 최대 x,y좌표'리스트를 생성한다. 
        min_box = (int(round(pred_box[2] - pred_box[0]/2)), int(round(pred_box[3] - pred_box[1]/2)))
        max_box = (int(round(pred_box[2] + pred_box[0]/2)), int(round(pred_box[3] + pred_box[1]/2)))
        # 출력하기
        cv2.rectangle(im_read, min_box, max_box, (0, 255, 0), 1) # 박스 그리기
        show_str = Classes_inDataSet + " : " + str(Classify_layer_output_list[i][class_index])
        # 글자 넣어주기
        cv2.putText(im_read, show_str, (min_box[0], min_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imwrite('output.jpg', im_read)