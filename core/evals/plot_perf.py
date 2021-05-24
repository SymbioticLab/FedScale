from __future__ import print_function
import os  
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import re, math
from matplotlib import rcParams
import matplotlib, csv, sys
from matplotlib import rc
import pickle

# rc('font',**{'family':'serif','serif':['Times']})
# rc('text', usetex=True)

def plot_line(datas, xs, linelabels = None, label = None, y_label = "CDF", name = "ss", _type=-1):
    _fontsize = 9
    fig = plt.figure(figsize=(2, 1.6)) # 2.5 inch for 1/3 double column width
    ax = fig.add_subplot(111)

    plt.ylabel(y_label, fontsize=_fontsize)
    plt.xlabel(label, fontsize=_fontsize)

    colors = ['black', 'orange',  'blueviolet', 'slateblue', 'DeepPink', 
            '#FF7F24', 'blue', 'blue', 'blue', 'red', 'blue', 'red', 'red', 'grey', 'pink']
    linetype = ['-', '--', '-.', '-', '-' ,':']
    markertype = ['o', '|', '+', 'x']

    X_max = float('inf')

    X = [i for i in range(len(datas[0]))]

    for i, data in enumerate(datas):
        _type = max(_type, i)
        plt.plot(xs[i], data, linetype[_type%len(linetype)], color=colors[i%len(colors)], label=linelabels[i], linewidth=1.)
        X_max = min(X_max, max(xs[i]))
    
    legend_properties = {'size':_fontsize} 
    
    plt.legend(
        prop = legend_properties,
        frameon = False)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")

    plt.tight_layout()
    
    plt.tight_layout(pad=0.1, w_pad=0.01, h_pad=0.01)
    plt.yticks(fontsize=_fontsize)
    plt.xticks(fontsize=_fontsize)

    plt.xlim(0) 
    plt.ylim(0)

    plt.savefig(name)


def load_results(file):
    with open(file, 'rb') as fin:
        history = pickle.load(fin)

    return history


def movingAvg(arr, windows):

    mylist = arr
    N = windows
    cumsum, moving_aves = [0], []

    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/float(N)
            moving_aves.append(moving_ave)

    return moving_aves


def main(files):
    walltime = []
    metrics = []
    setting_labels = []
    task_type = None
    task_metrics = {'cv': 'top_5: ', 'speech': 'top_1: ', 'nlp': 'loss'}
    metrics_label = {'cv': 'Accuracy (%)', 'speech': 'Accuracy (%)', 'nlp': 'Perplexity'}
    plot_metric = None

    for file in files:
        history = load_results(file)
        if task_type is None:
            task_type = history['task']
        else:
            assert task_type == history['task'], "Please plot the same type of task (openimage, speech or nlp)"

        walltime.append([])
        metrics.append([])
        setting_labels.append(f"{history['sample_mode']}+{'Prox' if history['gradient_policy'] is None else history['gradient_policy']}")

        metric_name = task_metrics[task_type]

        for r in history['perf'].keys():
            walltime[-1].append(history['perf'][r]['clock']/3600.)
            metrics[-1].append(history['perf'][r][metric_name] if task_type != 'nlp' else history['perf'][r][metric_name] ** 2)

        metrics[-1] = movingAvg(metrics[-1], 2)
        walltime[-1] = walltime[-1][:len(metrics[-1])]
        plot_metric = metrics_label[history['task']]

    plot_line(metrics, walltime, setting_labels, 'Training Time (hours)', plot_metric, 'time_to_acc.pdf')

main(sys.argv[1:])

