import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm

class ResNet(tf.keras.Model):
    def __init__(self, initializer, regularizer):
        
        super(ResNet, self).__init__(name='ResNet')
        self.layer_1 = tf.keras.layers.Dense(84, activation= None, kernel_initializer = Box_regression_layer_initializer, name = "output_2")

    
    def call(self, inputs):
        output = 0

        return output
    

    # multi task loss
    def loss(self, image):

        loss = 0
        
        return loss

    def get_grad(self, Loss, cls_reg_boolean): # cls_reg_boolean = 0이면 cls, cls_reg_boolean = 1 이면 reg
        g = 0

        return g
    
    def App_Gradient(self, Loss, training_step) :
        g = 0
        
        
    def Training_model(self, image_list, cls_layer_ouptut_label_list, reg_layer_ouptut_label_list, training_step):
        output = 0

        return output
