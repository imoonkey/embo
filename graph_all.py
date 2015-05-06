# -*- coding: utf-8 -*-
"""
Created on Wed May 06 14:10:33 2015

@author: Daniel
"""

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

def cum_max(arr):
    max_out = np.zeros(arr.shape)
    curr_max = arr[:,0]
    for i in range(arr.shape[1]):
        curr_max = np.maximum(curr_max, arr[:,i])
        max_out[:,i] = curr_max
    return max_out

def graph_all():
    actual_ll = np.load('actual_ll.npy')
    em_ll = np.load('em_ll.npy')
    bo_em_ll_tmp = np.load('bo-em-hmm.npy')
    bo_ll = -np.load('simple-bo-hmm.npy')
    
    # fix bo_em_ll
    bo_em_ll = np.zeros((10, 50)) - 1000000
    for i in range(10):
        for j,v in enumerate(bo_em_ll_tmp[i]):
            bo_em_ll[i,j] = -v
    
    em_ll = cum_max(em_ll)
    bo_em_ll = cum_max(bo_em_ll)
    bo_ll = cum_max(bo_ll)
    
    em_ll_ratios = em_ll - actual_ll[:, np.newaxis]
    bo_em_ll_ratios = bo_em_ll - actual_ll[:, np.newaxis]
    bo_ll_ratios = bo_ll - actual_ll[:, np.newaxis]
    
    em_ll_ratio_means = em_ll_ratios.mean(axis=0)
    bo_em_ll_ratio_means = bo_em_ll_ratios.mean(axis=0)
    bo_ll_ratio_means = bo_ll_ratios.mean(axis=0)
    
    plt.figure(dpi=300, figsize=(10,10))
    plt.style.use('ggplot')
    plt.plot(em_ll_ratio_means, label='EM', lw=5)
    plt.plot(bo_ll_ratio_means, label='BO', lw=5)
    plt.plot(bo_em_ll_ratio_means, label='EM+BO', lw=5)
    plt.ylim(ymin=-1, ymax=1.5)
    plt.legend(loc='lower right')
    plt.xlabel("Number of points/restarts")
    plt.ylabel("log likelihood ratio")
    plt.title("Comparison of MLE Approximation")
    # plt.show()
    plt.savefig('mle_plot.pdf')

    
if __name__ == '__main__':
    graph_all()