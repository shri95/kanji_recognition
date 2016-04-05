# -*- coding: utf-8 -*-
"""
Created on Mon Apr 04 14:54:02 2016

@author: shri
This file is used to perform image recognition based on trained model
"""
#############################################
import cv2
import tensorflow as tf
import numpy as np
from kanji_rec_common_params import *

from kanji_rec_model import *
from kanji_rec_map import getKanjiMap
import sys
import os
#Model delivered at the end of training process
MODEL_NAME = "./model_kanj_irec64.ckpt"

def prepareImage(im):
    '''Image conditioning to bring the image in a standard 64x64 format.
    At first 5% background is added to the image to ensure that kanji strokes touching the edges are captured;
    the image is then converted to square by adding extra background and 
    it is finally resized to bring it to 64x64 size'''
    (H,W) = im.shape
    #5% background 
    offset = int(round(float(max(H,W))*5/100))

    (l,r)= (int(np.ceil(float(H-W)/2)), int(np.floor(float(H-W)/2))) if H >W else (0,0)
    (t,b)= (int(np.ceil(float(W-H)/2)), int(np.floor(float(W-H)/2))) if W >H else (0,0)
    outImg=cv2.copyMakeBorder(im,t+offset,b+offset,l+offset,r+offset,cv2.BORDER_CONSTANT,value=[255,255,255])
    outImg = cv2.resize(outImg,(IMAGE_SIZE,IMAGE_SIZE))
    return outImg


eval_data_node = tf.placeholder(
    tf.float32,
    shape=(1, 64, 64, 1))
logits2 = model(eval_data_node, False)
eval_prediction = tf.nn.softmax(logits2)

# The model should ideally have saved variable names. However, this fix is 
#required to ensure that variables are retrieved with mapping consistent with that if training process
loadParam = {
"Variable":conv1_weights,
"Variable_1":conv1_biases,
"Variable_2":conv2_weights,
"Variable_3":conv2_biases ,
"Variable_4":fc1_weights,
"Variable_5":fc1_biases,
"Variable_6":fc2_weights,
"Variable_7":fc2_biases
}

  
def main(argv):
    mapDict = getKanjiMap()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        #restore variables from training process
        saver = tf.train.Saver(loadParam)
        saver.restore(sess, MODEL_NAME)
        for argc in range(1,len(sys.argv)):
            fName = sys.argv[argc]
            if os.path.isfile(fName):

               img = cv2.imread(fName,0)
               img=prepareImage(img)
               # to ensure that image has 0 mean and [-1:1]
               img = (img - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
               img = img.reshape([1,IMAGE_SIZE,IMAGE_SIZE,1])
    
               predictions = sess.run(
                    eval_prediction,
                    feed_dict={eval_data_node: img})

               labelID = (np.argmax(predictions))
               print("labelID: %d; Recognized Kanji:%s" %(labelID, mapDict[str(labelID)]))

            else:
                print("%s does not exist\n" %(fName))
                continue


if __name__ == "__main__":
   main(sys.argv[1:])