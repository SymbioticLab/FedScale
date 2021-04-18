import pickle
import random, gc
import numpy as np
from numpy import *

from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
from collections import OrderedDict
import math
import sys
from kuiper import create_testing_selector


N = []
global_ys = None
global_xs = None

def plot(datas, xs, linelabels = None, label = None, y_label = "CDF", name = "ss.pdf", _type=-1):
    global global_mean, global_cnt, global_xs, global_ys

    _fontsize = 8
    fig = plt.figure(figsize=(2, 1.6))
    ax = fig.add_subplot(111)
    plt.ylabel(y_label, fontsize=_fontsize)
    plt.xlabel(label, fontsize=_fontsize)

    colors = ['grey', 'black']

    linetype = ['-.', '-', '-.', '-', '-' ,':']
    markertype = ['o', '|', '+', 'x']
    X_max = 9999999999999
    Y_max = -1
    X = [i for i in range(len(datas[0]))]
    for i, data in enumerate(datas):
        _type = max(_type, i)
        _max = []
        _min = []
        avg = []
        for j, trial in enumerate(data):

            avg.append(sorted(trial)[int(0.5*len(trial))])

            sortedTemp = sorted(trial)
            mx = sortedTemp[-1]

            _max.append(mx)
            _min.append(sortedTemp[0])

        xs = array(xs)
        global_ys = array(global_ys)

        plt.fill_betweenx(xs, x1=_max, x2=_min, alpha=0.3, color=colors[_type%len(colors)], label='Empirical Dev.')
        plt.plot(global_xs, global_ys, '-', color='blueviolet', linewidth=1., label='Kuiper')


    ax.set_yscale('log')
    legend_properties = {'size':_fontsize}
    plt.legend(
        loc = 'upper right',
        #bbox_to_anchor=(0.44, 0.61),
        prop = legend_properties,
        frameon = False)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.xticks([0, 0.25, 0.5, 0.75, 1.0],fontsize=_fontsize)
    plt.yticks([10, 100, 1000], fontsize=_fontsize)

    ax.tick_params(axis='x', which='minor', colors='black')
    #plt.xticks([10, 100, 1000], fontsize=_fontsize)
    plt.ylim(10, 3500)
    plt.xlim(0, 1.)
    plt.tight_layout(pad=0.2)


    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.savefig(name)


def get_l1_distance(norm_global, results, N):
    distance = []

    for sample_num_of_client in N:
        samples = results[sample_num_of_client]
        distance.append([])

        for sample in samples:
            distance[-1].append((np.abs(np.array(sample)/float(sample_num_of_client) - norm_global)))

    return distance

def load_results(file):
    with open(file, 'rb') as fin:
        data = pickle.load(fin)
    return data

def process_dev(data_file):
    global global_xs, global_ys

    ''' Result of fig 16(a) -- OpenImage dataset '''
    datas = load_results(data_file)

    num_of_client = datas['clients']
    N = datas['sample_n']
    global_data = datas['global_dist']

    num_of_class = datas['num_of_class']
    max_min_range = datas['max_min_range']

    # ==============Statistics of random sampling ==================== #
    distance = get_l1_distance(global_data/float(num_of_client), datas, N)
    dev_dicts = OrderedDict()

    dev_max = []
    temp_max_devs = []
    for i, dist in enumerate(distance):
        max_temp = np.max(dist)
        dev_max.append(max_temp)
        max_runs = np.max(dist, axis=1)
        dev_dicts[max_temp] = [max_runs, N[i]]

    hoeff_ns = []

    testing_selector = create_testing_selector()
    # Given dev target, output # of participants needed
    for dev in dev_max:
        n = testing_selector.select_by_deviation(dev_target=dev, range_of_capacity=max_min_range,
                                                total_num_clients=num_of_client, confidence=0.95, overcommit=1.0)
        dev_dicts[dev].append(n)
        #print(n)

    # strictly sort the dict
    ordered_dev_dicts = OrderedDict()
    dev_lists = list(dev_dicts.keys())

    for i, dev in enumerate(dev_lists):
        ordered_dev_dicts[dev] = dev_dicts[dev]


    Ns = []
    Devs = []
    Hoeff_n = []
    for dev in ordered_dev_dicts:
        Devs.append(ordered_dev_dicts[dev][0].flatten())
        Ns.append(ordered_dev_dicts[dev][1])
        Hoeff_n.append(ordered_dev_dicts[dev][2])



    max_dev = max(ordered_dev_dicts.keys())
    global_xs = list(ordered_dev_dicts.keys())/max_dev
    global_ys = array(Hoeff_n)
    norm_dev = array(Devs)/max_dev

    #print(Ns)
    return norm_dev, Ns

norm_dev, Ns = process_dev(data_file='./speech_samples_f16.pkl')
plot([norm_dev], array(Ns), linelabels = [''],  label = "Deviation Target", y_label='# of Sampled Clients', name='figure16.pdf')
