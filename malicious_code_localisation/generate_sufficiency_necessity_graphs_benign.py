import os
import sys
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
import numpy as np

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pds", "--pathdictsuff", help="The path to the dict that contains the sufficiency predictions based on sub-images", type=str, required=True)
    parser.add_argument("-pdn", "--pathdictnece", help="The path to the dict that contains the necessity predictions based on sub-images", type=str, required=True)
    args = parser.parse_args()
    return args





#Necessity

def necessity(path_dict_necessity):
    dict_predictions_with_sum = load_obj(path_dict_necessity)
    all_pas = [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16]
    number_of_samples = len(dict_predictions_with_sum["sha"])
    for pas in all_pas:
        for j in range(pas, 16385, pas):
            if j==pas:
                dict_predictions_with_sum[pas]["0:"+str(pas)].append((sum([(1-m) for m in dict_predictions_with_sum[pas]["0:"+str(pas)]])*100)/number_of_samples)
            elif j==16384:
                dict_predictions_with_sum[pas][str(16384-pas)+":"+str(16384)].append((sum([(1-m) for m in dict_predictions_with_sum[pas][str(16384-pas)+":"+str(16384)]])*100)/number_of_samples)
            else:
                dict_predictions_with_sum[pas][str(j-pas)+":"+str(j)].append((sum([(1-m) for m in dict_predictions_with_sum[pas][str(j-pas)+":"+str(j)]])*100)/number_of_samples)

    split = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    count = 0

    fig, ax = plt.subplots(figsize=(10, 10), nrows=10, ncols=1)
    for k, v in dict_predictions_with_sum.items():
        if k!="sha" and k!="index":
            list_for_imshow = []
            len_of_keys = len(list(v.keys()))
            extend_with = int(1024/len_of_keys)
            percents = []
            for sub_v in list(v.values()):
                percents.append(sub_v[-1])
            for perc in percents:
                for j in range(extend_with):
                    list_for_imshow.append(perc)

            im = ax[count].imshow(np.array(list_for_imshow).reshape(1, 1024), cmap='gray', vmin=0, vmax=100,  aspect='auto')

            ax[count].tick_params(axis="x", colors="red")
            ax[count].set_xticks([h for h in range(0, int(16384/16), int(k/16))])
            ax[count].set_xticklabels([])
            ax[count].set_yticks([])
            ax[count].set_title("images are split into "+str(split[count])+ " parts")
            count+=1

    im=cm.ScalarMappable(cmap='gray')

    cbaxes = fig.add_axes([0.02, -0.08, 0.968, 0.02]) 
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), orientation="horizontal", ticklocation="bottom", cax = cbaxes)
    cbar.ax.tick_params(axis="x", colors="red", width=2)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([])

    necessities = ['The necessity is high', 'The necessity is low']
    my_labels = ['$0$% of goodware are detected','$100$% of goodware are detected']
    
    cbar.ax.text(0, -0.7, my_labels[0], va='center', fontsize=14)
    cbar.ax.text(0, -1.7, necessities[0], va='center', fontsize=14)
    cbar.ax.text(1, -0.7, my_labels[1], ha="right", va='center', fontsize=14)
    cbar.ax.text(1, -1.7, necessities[1], ha="right", va='center', fontsize=14)

    fig.tight_layout()
    plt.show()
    fig.savefig('necessity_for_good.pdf', bbox_inches='tight')






#Sufficiency
def sufficiency(path_dict_sufficiency):
    dict_predictions_with_sum = load_obj(path_dict_sufficiency)
    all_pas = [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16]
    number_of_samples = len(dict_predictions_with_sum["sha"])
    for pas in all_pas:
        for j in range(pas, 16385, pas):
            if j==pas:
                dict_predictions_with_sum[pas]["0:"+str(pas)].append((sum([(1-m) for m in dict_predictions_with_sum[pas]["0:"+str(pas)]])*100)/number_of_samples)
            elif j==16384:
                dict_predictions_with_sum[pas][str(16384-pas)+":"+str(16384)].append((sum([(1-m) for m in dict_predictions_with_sum[pas][str(16384-pas)+":"+str(16384)]])*100)/number_of_samples)
            else:
                dict_predictions_with_sum[pas][str(j-pas)+":"+str(j)].append((sum([(1-m) for m in dict_predictions_with_sum[pas][str(j-pas)+":"+str(j)]])*100)/number_of_samples)


    split = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    count = 0

    fig, ax = plt.subplots(figsize=(10, 10), nrows=10, ncols=1)
    for k, v in dict_predictions_with_sum.items():
        if k!="sha" and k!="index":
            list_for_imshow = []
            len_of_keys = len(list(v.keys()))
            extend_with = int(1024/len_of_keys)
            percents = []
            for sub_v in list(v.values()):
                percents.append(sub_v[-1])
            for perc in percents:
                for j in range(extend_with):
                    list_for_imshow.append(perc)

            ax[count].imshow(np.array(list_for_imshow).reshape(1, 1024), cmap='gray', vmin=0, vmax=100,  aspect='auto')
            ax[count].tick_params(axis="x", colors="red")
            ax[count].set_xticks([h for h in range(0, int(16384/16), int(k/16))])
            ax[count].set_xticklabels([])
            ax[count].set_yticks([])
            ax[count].set_title("images are split into "+str(split[count])+ " parts")
            count+=1

    im=cm.ScalarMappable(cmap='gray')

    cbaxes = fig.add_axes([0.02, -0.08, 0.968, 0.02]) 
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), orientation="horizontal", ticklocation="bottom", cax = cbaxes)
    cbar.ax.get_xaxis().set_ticks([])

    cbar.ax.tick_params(axis="x", colors="red", width=2)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([])


    sufficiencies = ['The sufficiency is low', 'The sufficiency is high']
    my_labels = ['$0$% of goodware are detected','$100$% of goodware are detected']

    cbar.ax.text(0, -0.7, my_labels[0], va='center', fontsize=14)
    cbar.ax.text(0, -1.7, sufficiencies[0], va='center', fontsize=14)
    cbar.ax.text(1, -0.7, my_labels[1], ha="right", va='center', fontsize=14)
    cbar.ax.text(1, -1.7, sufficiencies[1], ha="right", va='center', fontsize=14)



    fig.tight_layout()
    plt.show()
    fig.savefig('sufficiency_for_good.pdf', bbox_inches='tight')





if __name__ == "__main__":
    args = parseargs()
    path_dict_necessity = args.pathdictnece
    path_dict_sufficiency = args.pathdictsuff
    
    
    necessity(path_dict_necessity)
    sufficiency(path_dict_sufficiency)
