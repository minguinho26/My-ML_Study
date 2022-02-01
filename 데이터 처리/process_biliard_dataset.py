import json
import os
from PIL import Image
from tqdm import tqdm
import random

def make_new_dataset(dataset_root_folder) : 
    
    ori_image_folder = dataset_root_folder + 'ori_images'
    new_image_folder = dataset_root_folder + 'images'
    
    ori_annotation_folder = dataset_root_folder + 'ori_labels'
    new_annotation_folder = dataset_root_folder + 'labels'
    
    # 원래 이미지는 640 x 360
    # 416 x 416이미지로 처리해야함
    name_classes = ['biliard_stick', 'hand', 'two_balls', 'three_balls', 'red_ball', 'white_ball', 'yellow_ball', 'moving_red_ball', 'moving_white_ball', 'moving_yellow_ball']

    ori_annotation_list = sorted(os.listdir(ori_annotation_folder))
    random.shuffle(ori_annotation_list)
    
    if '.DS_Store' in ori_annotation_list : 
        ori_annotation_list.remove('.DS_Store')

    for i in tqdm(range(0, len(ori_annotation_list)), desc = "make new annotation") :

        with open(ori_annotation_folder + '/' + ori_annotation_list[i], 'r') as f:
            json_data = json.load(f)
            
        img = Image.open(ori_image_folder + '/' + ori_annotation_list[i][:-5] + ".jpg").convert('RGB')
        
        img_resize = img.resize((416, 416))
        img_resize.save(new_image_folder + '/' + ori_annotation_list[i][:-5] + ".jpg")
        
        new_txt_file = ori_annotation_list[i][:-5] + ".txt"
        new_label_file = open(new_annotation_folder + "/" + new_txt_file, 'w')

        for annotation in json_data['shapes'] :
            label_idx = name_classes.index(annotation['label'])
            
            if label_idx >= 7 : # movind ball을 그냥 ball이라고 라벨링
                label_idx -=3
                
            x1 = annotation['points'][0][0] * (416.0/640.0)
            y1 = annotation['points'][0][1] * (416.0/360.0)

            x2 = annotation['points'][1][0] * (416.0/640.0)
            y2 = annotation['points'][1][1] * (416.0/360.0)

            x_center = ((x1 + x2) / 2) / 416.0
            y_center = ((y1 + y2) / 2) / 416.0
            width = (x2 - x1) / 416.0
            height = (y2 - y1) / 416.0

            new_annotation = str(label_idx) + " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height) + "\n"
            
            new_label_file.write(new_annotation)

        new_label_file.close()

def create_img_path_txt_file(datset_root_folder) :    
    img_path_list = sorted(os.listdir(datset_root_folder + 'images'))
    random.shuffle(img_path_list)
    
    if '.DS_Store' in img_path_list : 
        img_path_list.remove('.DS_Store')
    
    count = 1
    valid_dataset_start_index = int(len(img_path_list) * 0.75) # 몇번째 이미지부터 validation dataset으로 사용할 것인가(801이면 800번째 사진까지 학습용 데이터셋으로 사용)
    
    train_img_path_list_file = open(datset_root_folder + 'train.txt', 'w')
    valid_img_path_list_file = open(datset_root_folder + 'valid.txt', 'w')
    
    for img_name in img_path_list :
        img_path = datset_root_folder + 'images' + '/' + img_name
        
        if count < valid_dataset_start_index :
            train_img_path_list_file.write(img_path + "\n")
        else : 
            valid_img_path_list_file.write(img_path + "\n")
        
        count+=1
    
    train_img_path_list_file.close()
    valid_img_path_list_file.close()
    