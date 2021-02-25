# DexRay:

## To generate the images, use ``apktoimage.py`` script:
This script generates an image from the given APK based on the Dalvik bytecode.

### INPUTs are: 
    - The APK to convert into image
    - The path in which the resulting image will be
### OUTPUTs are:
    - A greyscale image representing the Dalvik bytecode

Example: 

```python3 apktoimage.py APK DESTINATION ```

## To generate an obfuscated APK, use ``launch_obfuscation.sh`` script in Obfuscation/ folder:
This script generates an obfuscated APK from the given APK based on options given in the script.

### INPUTs are: 
    - The APK to obfuscate
    - The path in which the resulting APK will be
### OUTPUTs are:
    - An obfuscated APK based one the input APK

Example: 

```sh launch_obfuscation.sh PATH_TO_APK PATH_OF_NEW_APK```

## To train and test the model, use ``DexRay.py`` script:
This script trains the Neural Network using the training images, and evaluates its learning using the test dataset.
The evaluation is repeated 10 times using the holdout technique.
The training, validation and test hashes are provided in `data_splits` directory.
To use this script, you need to extract the images for goodware and malware applications in `goodware_hashes.txt` and `malware_hashes.txt` using the `apktoimage.py` script.


### INPUTs are: 

    - The path to the directory that contains the extracted images. 
      In this directory, you need to have two folders: malware and goodware.
    - The name of directory where to save your model.
    - The name of the file where to save the evaluation results.

### OUTPUTs are:
    - The file that contains Accuracy, Precision, Recall, and F1-score of the ten trained models
      and their average scores.
    - The ten trained models

Example: 

```python3 DexRay.py -p "dataset_images" -d "results_dir" -f "results_dir/scores.txt"```


## To train and test the model on the obfuscated apps, use ``DexRay_obfuscation.py`` script:
This script trains the Neural Network using the training images, and evaluates its learning using the test dataset as described in Section4.2 of the paper.
The evaluation is repeated 10 times using the holdout technique.
The training, validation and test hashes are provided in `data_splits/obfuscation` directory.
To use this script, you need to extract images for the obfuscated and the non_obfuscated goodware and malware applications in `goodware_hashes.txt` and `malware_hashes.txt` using the `apktoimage.py` and `launch_obfuscation.sh` scripts.

### INPUTs are: 

    - The path to the directory that contains the extracted images. 
      In this directory, you need to have three folders: malware, goodware, and obf. 
      "malware" and "goodware" folders contain the images of the non_obfuscated apps.
      The "obf" contain also "malware" and "goodware" folders but for the obfuscated apps
    - The name of directory where to save your model.
    - The name of the file where to save the evaluation results.
    - The key-word about the obfuscated experiment to conduct. 
      - obf1 to evaluate DexRay on obfuscated apps that it has seen their non-obfuscated
        version in the training datase; 
      - obf2 to evaluate DexRay on obfuscated apps that it has NOT seen their non-obfuscated
        version in the training dataset; 
      - obf3 to augment the training dataset with 25% of obf images;  
      - obf4 to augment the training dataset with 50% of obf images; 
      - obf5 to augment the training dataset with 75% of obf images; 
      - obf6 to augment the training dataset with 100% of obf images.

    
### OUTPUTs are:
    - The file that contains Accuracy, Precision, Recall, and F1-score of the ten trained models 
      and their average scores.
    - The ten trained models
    - The checkpoint files of the training process

Example: 

```python3 DexRay_obfuscation.py -p "dataset_images" -d "results_dir_obf" -f "results_dir/scores_obf.txt" -obf "obf1"```
