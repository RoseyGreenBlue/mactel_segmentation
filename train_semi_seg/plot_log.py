
import numpy as np
import os
import glob
import argparse

import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_history(log_1, log_2, log_3, fpath_out):
    # Scatter plot
    fig = plt.figure(figsize=(9,3))# width and height in inches
    fig.tight_layout()
    #fig.subplots_adjust(top=0.85)
    log_1 = np.array(log_1)
    log_2 = np.array(log_2)
    log_3 = np.array(log_3)
    epochs = len(log_1)
    x = np.arange(1, epochs+1, 1)
    #ax1 = fig.add_subplot(121)
    plt.title('Learning Curve')
    plt.xlabel('epochs')
    plt.ylabel('training_loss')
    plt.plot(x, log_1, '-b', label='train_sup1')
    plt.plot(x, log_2, '-g', label='train_sup2')
    plt.plot(x, log_3, '-r', label='train_cps')
    plt.legend(loc="upper right")
    print("save file to "+fpath_out)
    plt.savefig(fpath_out, dpi=100)
    plt.clf()

def plot_history_bs(log_1, fpath_out):
    # Scatter plot
    fig = plt.figure(figsize=(9,3))# width and height in inches
    fig.tight_layout()
    #fig.subplots_adjust(top=0.85)
    log_1 = np.array(log_1)
    epochs = len(log_1)
    x = np.arange(1, epochs+1, 1)
    #ax1 = fig.add_subplot(121)
    plt.title('Learning Curve')
    plt.xlabel('epochs')
    plt.ylabel('training_loss')
    plt.plot(x, log_1, '-b', label='train_sup1')

    plt.legend(loc="upper right")
    print("save file to "+fpath_out)
    plt.savefig(fpath_out, dpi=100)
    plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-eventfile", type=str, help="path to event file", required = True)
    args = parser.parse_args()
    event_file = glob.glob(os.path.join(args.eventfile, "events.*")) #"MVCNN_eyeact_61_efficientnetb0_view24_reg_debug_stage_1/events.out.tfevents.1656442219.415286d9c484"
    event_file = event_file[0]
    train_loss_sup1, train_loss_sup2, train_loss_cps = [], [], []
    for e in summary_iterator(event_file):
        for v in e.summary.value:
            if v.tag == 'train_loss_sup':
                train_loss_sup1.append(v.simple_value)
            if v.tag == 'train_loss_sup_r':
                train_loss_sup2.append(v.simple_value)
            if v.tag == 'train_loss_cps':
                train_loss_cps.append(v.simple_value)

    if len(train_loss_sup1) > 0:
        #plot_history(train_loss_sup1, train_loss_sup2, train_loss_cps, "%s/train_loss_%s.png"%(event_file.split("/")[0], event_file.split("/")[-1].replace(".", "_")))
        plot_history_bs(train_loss_sup1, "%s/train_loss_%s.png"%(event_file.split("/")[0], event_file.split("/")[-1].replace(".", "_")))
