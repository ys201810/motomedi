# -*- coding: utf-8 -*- 
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle

def print_cmx(y_true, y_pred):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize = (20,14))
    sn.heatmap(df_cmx, annot=True)
    plt.savefig('confusion_matrix.png')

if __name__ == '__main__':
    with open("/home/yusuke/work/motomedi/training/script/classification/pred_results/all_pred_list.pickle", "rb") as pf:
        pred_list = pickle.load(pf)

    with open("/home/yusuke/work/motomedi/training/script/classification/pred_results/all_true_list.pickle", "rb") as tf:
        true_list = pickle.load(tf)

    print_cmx(true_list, pred_list)