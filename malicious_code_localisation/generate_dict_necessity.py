import numpy as np 
import tensorflow as tf 
import random as python_random 
import tensorflow_addons as tfa 
import tensorflow.keras as keras
from PIL import Image

import numpy as np
random_seed = 123456
np.random.seed(random_seed)
python_random.seed(random_seed)
tf.random.set_seed(random_seed)
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Flatten, Dense
import argparse
from copy import copy, deepcopy




def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="The path to the directory that contains malware and goodware image folders", type=str, required=True)
    parser.add_argument("-tf", "--tfile", help="The name of the file that contains the test hashes", type=str, required=True) #we used test1.txt in our experiments
    parser.add_argument("-pm", "--pmodel", help="The path to the trained model", type=str, required=True) #we used model1 in our experiments, which is the model trained on the train1.txt
    parser.add_argument("-mg", "--malgood", help="'malware' or 'goodware' keyword", type=str, required=True)
    args = parser.parse_args()
    return args



def file_to_list(file_path):
    my_list = []
    file = open(file_path, "r")
    for line in file:
        my_list.append(line.strip("\n"))
    file.close()
    return my_list

import pickle
def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
    
def save_obj(obj, path ):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    if parts[-2] == 'goodware':
        return [0]
    else:
        return [1]


def get_image(path_img):
    image = np.asarray(Image.open(path_img))
    image = tf.convert_to_tensor(image, dtype_hint=None, name=None)
    return image

def get_shape(image):
    return image.shape[0]

def decode_img(path_img):
    image = tf.numpy_function(get_image, [path_img], tf.uint8)
    shape = tf.numpy_function(get_shape, [image], tf.int64)
    image = tf.reshape(image, [shape, 1, 1])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMG_SIZE*IMG_SIZE, 1])
    return tf.reshape(image, [IMG_SIZE*IMG_SIZE, 1])

def process_path(file_path):
    label = get_label(file_path)
    img = decode_img(file_path)
    return img, label




def get_tf_vect(sha):
    sha_dataset = tf.data.Dataset.from_tensor_slices([os.path.join(path_images, sha)])
    sha_dataset = sha_dataset.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    batch_test = 1
    sha_dataset = sha_dataset.cache()
    sha_dataset = sha_dataset.batch(batch_test)
    sha_dataset = sha_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return sha_dataset



def get_pred(num):
    if num>0.5:
        return 1
    else:
        return 0



def get_predictions_for_small_segments_necessity(my_dict, sha, index):
    all_steps = [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16]
    for my_array in get_tf_vect(sha).as_numpy_iterator():
        for step in all_steps:
            for j in range(step, 16385, step):
                my_array_copy = deepcopy(my_array)
                my_array_copy[0][0][(j-step):j] = [[0]]*int(step)
                predict = model.predict(tf.data.Dataset.from_tensors(my_array_copy))
                predict = get_pred(predict)
                my_dict[step][str(j-step)+":"+str(j)].append(predict)
    my_dict["sha"].append(sha)
    my_dict["index"].append(index)
    return my_dict


def main(path_images, model, list_sha, IMG_SIZE, mal_good):
    dict_necessity = {}
    all_steps = [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16]

    dict_necessity["sha"] = []
    dict_necessity["index"] = []
    for step in all_steps:
        dict_necessity[step] = dict()
        for j in range(step, 16385, step):
            if j==step:
                dict_necessity[step]["0:"+str(step)] = []
            elif j==16384:
                dict_necessity[step][str(16384-step)+":"+str(16384)] = []
            else:
                dict_necessity[step][str(j-step)+":"+str(j)] = []
    
    list_of_indexes = [i for i in range(len(list_sha))]
    
    for index in list_of_indexes:
        dict_necessity = get_predictions_for_small_segments_necessity(dict_necessity, list_sha[index], index)
        print("sha with index: %s is processed" % (index))
        save_obj(dict_necessity, "dict_necessity_%s.pkl" % mal_good)

if __name__ == "__main__":
    args = parseargs()
    path_images = args.path
    test_file = args.tfile
    path_model = args.pmodel
    mal_good = args.malgood
    
    IMG_SIZE = 128
    PATH_FILES = "../data_splits"
    
    list_sha = file_to_list(os.path.join(PATH_FILES, test_file))
    list_sha = [i for i in list_sha if i.startswith(mal_good)]
    model = tf.keras.models.load_model(path_model)
    
    main(path_images, model, list_sha, IMG_SIZE, mal_good)
