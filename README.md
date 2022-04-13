# DexRay
In this repository, we host the artefacts of our approach DexRay, which is an image-based Android malware detector.
This work has been published at MLHat 2021.
### abstract
Computer vision has witnessed several advances in recent years, with unprecedented performance provided by deep representation learning research. Image formats thus appear attractive to other fields such as malware detection, where deep learning on images alleviates the need for comprehensively hand-crafted features generalising to different malware variants. We postulate that this research direction could become the next frontier in Android malware detection, and therefore requires a clear roadmap to ensure that new approaches indeed bring novel contributions. We contribute with a first building block by developing and assessing a baseline pipeline for image-based malware detection with straightforward steps. We propose DexRay, which converts the bytecode of the app DEX files into grey-scale "vector" images and feeds them to a 1-dimensional Convolutional Neural Network model. We view DexRay as foundational due to the exceedingly basic nature of the design choices, allowing to infer what could be a minimal performance that can be obtained with image-based learning in malware detection. The performance of DexRay evaluated on over 158k apps demonstrates that, while simple, our approach is effective with a high detection rate(F1-score= 0.96). Finally, we investigate the impact of time decay and image-resizing on the performance of DexRay and assess its resilience to obfuscation. This work-in-progress paper contributes to the domain of Deep Learning based Malware detection by providing a sound, simple, yet effective approach (with available artefacts) that can be the basis to scope the many profound questions that will need to be investigated to fully develop this domain.

# Getting started:
## To generate the images, use ``apktoimage.py`` script:
This script generates an image from the given APK based on the Dalvik bytecode.

### INPUTs are: 
    - The APK to convert into image
    - The path in which the resulting image will be
### OUTPUTs are:
    - A greyscale image representing the Dalvik bytecode

Example: 

```python3 apktoimage.py APK DESTINATION ```

## Images availability

Due to the large size of the images dataset, we share it upon request.

## To generate an obfuscated APK, use ``launch_obfuscation.sh`` script in Obfuscation/ folder:
This script generates an obfuscated APK from the given APK based on options given in the script.

### INPUTs are: 
    - The APK to obfuscate
    - The path for saving the resulting APK
### OUTPUTs are:
    - An obfuscated APK based on the input APK

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
This script trains the Neural Network using the training images, and evaluates its learning using the test dataset as described in Section4.4 of the paper.
The evaluation is repeated 10 times using the holdout technique.
The training, validation and test hashes are provided in `data_splits/obfuscation` directory.
To use this script, you need to extract images for the obfuscated and the non_obfuscated goodware and malware applications in `goodware_hashes.txt` and `malware_hashes.txt` using the `apktoimage.py` and `launch_obfuscation.sh` scripts.

### INPUTs are: 

    - The path to the directory that contains the extracted images. 
      In this directory, you need to have three folders: malware, goodware, and obf. 
      "malware" and "goodware" folders contain the images of the non_obfuscated apps.
      The "obf" contain also "malware" and "goodware" folders but for the obfuscated apps
    - The name of the directory where to save your model.
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
