import numpy as np
import tensorflow as tf
import random as python_random
import tensorflow_addons as tfa
import tensorflow.keras as keras
from PIL import Image
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

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

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="The path to the directory that contains malware image folders", type=str, required=True)
    parser.add_argument("-d", "--dir", help="The name of the directory where to save the model", type=str, required=True)
    parser.add_argument("-pfam", "--pathfam", help="The path to the file that contains the hashes and the family labels", type=str, required=True) 
    parser.add_argument("-f", "--file", help="The name of the file where to save the results of the evaluation", type=str, required=True) 
    args = parser.parse_args()
    return args




#This table should contain family labels according to the dataset. 
#We represent in this table the families that contain more than 1 sample
#Families that have only one sample have a default label of -1

table_classes = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=[
            'jiagu', 'secneo', 'smsreg', 'tencentprotect', 'dnotua', 'utilcode', 'datacollector', 
            'wroba', 'hiddad', 'smspay', 'hypay', 'kuguo', 'wapron', 'syringe', 'ewind', 
            'revmob', 'triada', 'fakeapp', 'autoins', 'airpush', 'styricka', 'baiduprotect', 
            'leadbolt', 'joker', 'hiddapp', 'dingxprotect', 'johnnie', 'ramnit', 'smsspy', 'umpay',
            'adflex', 'mobidash', 'kyview', 'kyvu', 'fakedep', 'youmi', 'sandr', 'adpush', 'opfake', 
            'remotecode', 'shuame', 'inmobi', 'cynos', 'oivim', 'gexin', 'rogueware', 'zepfod', 
            'pornvideo', 'zbot', 'xiny', 'feejar', 'magpay', 'necro', 'cyfin', 'autoinst', 
            'presenoker', 'gpspy', 'anubis', 'xhelper', 'skeeyah', 'virtualapp', 'dataeye', 
            'hiddenads', 'qlist', 'boogr', 'aesads', 'moplus', 'hqwar', 'gappusin', 'smforw', 
            'senrec', 'fakengry', 'dianjin', 'inazigram', 'hifrm', 'cnzz', 'bankbot', 'hiddenad', 
            'igexin', 'piom', 'rootnik', 'jocker'
        ],
        values=tf.constant([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 
            23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 
            44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 
            65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81
        ]),
    ),
    default_value=tf.constant(-1)
)

def get_family_index(file_path):
    hash_malware = tf.strings.split(file_path, os.path.sep)[-1]
    hash_malware = tf.strings.split(hash_malware, ".")[0]
    family_name = table_families.lookup(hash_malware)
    family_index = table_classes.lookup(family_name)
    return family_index

def get_label_vector(x, y):
    return (x, tf.one_hot(y, 82))


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
    label = get_family_index(file_path)
    img = decode_img(file_path)
    return img, label

    
def main(path_images, dir_name, file_name, families, CHANNELS, EPOCHS, BATCH_SIZE, IMG_SIZE, PATH_FILES):
    recall_list, precision_list, accuracy_list, f1_list = [], [], [], []

    model_architecture = Sequential()           
    model_architecture.add(Conv1D(filters=64, kernel_size=12, activation='relu', input_shape=(IMG_SIZE*IMG_SIZE, 1)))
    model_architecture.add(MaxPooling1D(pool_size=12))           
    model_architecture.add(Conv1D(filters=128, kernel_size=12, activation='relu')) 
    model_architecture.add(MaxPooling1D(pool_size=12))                     
    model_architecture.add(Flatten())
    model_architecture.add(Dense(256, activation='sigmoid'))
    model_architecture.add(Dense(82, activation='sigmoid'))


    file_results = open(file_name, "w")
    file_results.write("Scores of the performance evaluation are: Accuracy, Precision, Recall, F1-score\n")
    for i in range(1, 11):
        file_results.write("Run: %d \n" % i)

        with open(os.path.join(PATH_FILES, "train_family_"+str(i)+".txt")) as f:
            train_hashes = f.read().splitlines()
            train_imgs = [os.path.join(path_images, image_hash) for image_hash in train_hashes]
        f.close()

        with open(os.path.join(PATH_FILES, "valid_family_"+str(i)+".txt")) as f:
            valid_hashes = f.read().splitlines()
            valid_imgs = [os.path.join(path_images, image_hash) for image_hash in valid_hashes]
        f.close()

        with open(os.path.join(PATH_FILES, "test_family_"+str(i)+".txt")) as f:
            test_hashes = f.read().splitlines()
            test_imgs = [os.path.join(path_images, image_hash) for image_hash in test_hashes]
        f.close()
    
        train_dataset = tf.data.Dataset.from_tensor_slices(train_imgs)
        train_dataset = train_dataset.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.filter(lambda x, y: y>-1) #to keep only the samples that belong to families with more than one sample. 
        train_dataset = train_dataset.map(get_label_vector, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        length_train = len(train_imgs)
        batch_train = length_train//BATCH_SIZE
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(buffer_size=length_train, seed = random_seed, reshuffle_each_iteration=False)
        train_dataset = train_dataset.batch(batch_train)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        valid_dataset = tf.data.Dataset.from_tensor_slices(valid_imgs)
        valid_dataset = valid_dataset.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        valid_dataset = valid_dataset.filter(lambda x, y: y>-1)
        valid_dataset = valid_dataset.map(get_label_vector, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        length_valid = len(valid_imgs)
        batch_valid = length_valid//BATCH_SIZE
        valid_dataset = valid_dataset.cache()
        valid_dataset = valid_dataset.shuffle(buffer_size=length_valid, seed = random_seed, reshuffle_each_iteration=False)
        valid_dataset = valid_dataset.batch(batch_valid)
        valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices(test_imgs)
        test_dataset = test_dataset.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test_dataset = test_dataset.filter(lambda x, y: y>-1)
        test_dataset = test_dataset.map(get_label_vector, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        length_test = len(test_imgs)
        batch_test = length_test//BATCH_SIZE
        test_dataset = test_dataset.cache()
        test_dataset = test_dataset.shuffle(buffer_size=length_test, seed = random_seed, reshuffle_each_iteration=False)
        test_dataset = test_dataset.batch(batch_test)
        test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        model = keras.models.clone_model(model_architecture)
        adam = keras.optimizers.Adam()

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.categorical_accuracy])

        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=50, restore_best_weights=True)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(dir_name, 'cp'+str(i)), 
                                                         save_weights_only=True,
                                                         monitor='val_categorical_accuracy', 
                                                         mode='max',
                                                         save_best_only=True)
        path_save_model = os.path.join(dir_name, 'model'+str(i))

        model.fit(train_dataset, shuffle=True, validation_data = valid_dataset, epochs=EPOCHS, callbacks=[es_callback, cp_callback], verbose=2)
        model.save(path_save_model)
        
        test_labels = []
        for data_label in test_dataset.as_numpy_iterator():
            for my_label in data_label[1]: #take only the label and discard the data from a batch
                test_labels.append(my_label)
        test_labels = [np.argmax(test_labels[i]) for i in range(len(test_labels))]
        test_predictions = model.predict(test_dataset, verbose=2)
        test_predictions = [np.argmax(test_predictions[i]) for i in range(len(test_predictions))]

        accuracy_test = accuracy_score(test_labels, test_predictions)
        precision_test = precision_score(test_labels, test_predictions, average='weighted')
        recall_test = recall_score(test_labels, test_predictions, average='weighted')
        f1_test = f1_score(test_labels, test_predictions, average='weighted')


        accuracy_list.append(accuracy_test)
        precision_list.append(precision_test)
        recall_list.append(recall_test)
        f1_list.append(f1_test)

        file_results.write("%f %f %f %f  \n" % (accuracy_test, precision_test, recall_test, f1_test))
        file_results.write("#"*50+"\n")

    file_results.write("Average scores: %f %f %f %f" % (np.mean(accuracy_list), 
                                                        np.mean(precision_list), 
                                                        np.mean(recall_list), 
                                                        np.mean(f1_list)))

    file_results.close()



if __name__ == "__main__":
    
    args = parseargs()
    path_images = args.path
    dir_name = args.dir
    families_path = args.pathfam
    file_name = args.file   

    CHANNELS = 1
    EPOCHS = 200
    BATCH_SIZE = 500
    IMG_SIZE = 128
    PATH_FILES = "data_splits_families"


    families = open(families_path, "r")
    hash_family = dict()
    for line in families:
        hash_family[line.strip("\n").split("\t")[0].upper()] = line.strip("\n").split("\t")[1]
    families.close()
    
    table_families = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=[k for k in hash_family.keys()],
        values=[v for v in hash_family.values()]),
    default_value="None")

    main(path_images, dir_name, file_name, families, CHANNELS, EPOCHS, BATCH_SIZE, IMG_SIZE, PATH_FILES)