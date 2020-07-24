# -*- coding: utf-8 -*-
#pylint: skip-file
import sys
import os
import os.path
import time
from operator import itemgetter
import theano
import numpy as np
import pickle
import random
from commons import *

def load_stopwords(f_path = None):
    stop_words = {}
    f = open(f_path, "r")
    for line in f:
        line = line.strip('\n').lower()
        stop_words[line] = 1
    return stop_words

def get_batch_index(n, batch_size, is_predict = False):
    i_list = [i for i in range(n)]
    if not is_predict:
        random.shuffle(i_list)

    batch_index = []
    x = []
    for i in range(n):
        x.append(i_list[i])
        if len(x) == batch_size or i == (n-1):
            batch_index.append(x)
            x = []
    return batch_index

def get_batch_data(raw_x, consts, options, modules):
    batch_size = len(raw_x)
    hfw = modules["hfw"]
    w2i = modules["w2i"]
    i2w = modules["i2w"]
    w2w = modules["w2w"]
    w2idf = modules["w2idf"]
    u2ws = modules["u2ws"]
    i2ws = modules["i2ws"]
    len_y = consts["len_y"]
    lvt_dict_size = consts["lvt_dict_size"]

    xu = np.zeros(batch_size, dtype = "int64")
    xi = np.zeros(batch_size, dtype = "int64")
    yr = np.zeros(batch_size, dtype = theano.config.floatX)
    yr_vec = np.zeros((batch_size, DIM_RATING) , dtype = theano.config.floatX)
    y_review_mark = np.zeros((batch_size, lvt_dict_size) , dtype = theano.config.floatX)
    y_review_tf = np.zeros((batch_size, lvt_dict_size) , dtype = theano.config.floatX)
    y_summary_index = np.zeros((len_y, batch_size, 1) , dtype = "int64")
    y_summary_mark = np.zeros((len_y, batch_size) , dtype = theano.config.floatX)
    y_summary_lvt = np.zeros((len_y, batch_size, 1) , dtype = "int64")

    dic_review_words = {}
    w2i_review_words = {}
    lst_review_words = []
    lvt_user_item = {w2i[W_EOS]}
    lvt_i2i = {}
    for i in range(batch_size):
        uid, iid, r, summary, review, _, _ = raw_x[i] 
        xu[i] = uid
        xi[i] = iid
        yr[i] = r
        yr_vec[i, int(r)] = 1.
        lvt_user_item |= u2ws[uid] | i2ws[iid]

        if len(summary) > len_y:
            summary = summary[0:len_y-1] + [W_EOS]

        for wi in range(len(summary)):
            w = summary[wi]
            w = w2w[w]
            if w in dic_review_words:
                dic_review_words[w] += 1
            else:
                w2i_review_words[w] = len(dic_review_words)
                dic_review_words[w] = 1
                lst_review_words.append(w2i[w])
            y_summary_index[wi, i, 0] = w2i[w]
            y_summary_lvt[wi, i, 0] = w2i_review_words[w]
            y_summary_mark[wi, i] = 1

        for w in review:
            w = w2w[w]
            if w in dic_review_words:
                dic_review_words[w] += 1
            else:
                w2i_review_words[w] = len(dic_review_words)
                dic_review_words[w] = 1
                lst_review_words.append(w2i[w])
            y_review_mark[i, w2i_review_words[w]] = 1
            y_review_tf[i, w2i_review_words[w]] += 1#w2idf[w]

    y_review_tf /= 10.

    if len(dic_review_words) < lvt_dict_size:
        for rd_hfw in hfw:
            if rd_hfw not in dic_review_words:
                w2i_review_words[rd_hfw] = len(dic_review_words)
                dic_review_words[rd_hfw] = 0
                lst_review_words.append(w2i[rd_hfw])
            if len(dic_review_words) == lvt_dict_size:
                break
    else:
        print ("!!!!!!!!!!!!")

    for i in range(len(lst_review_words)):
        lvt_i2i[i] = lst_review_words[i]
    assert len(dic_review_words) == lvt_dict_size
    assert len(lst_review_words) == lvt_dict_size

    if options["is_predicting"]:
        lvt_i2i = {}
        lst_review_words = list(lvt_user_item)
        for i in range(len(lst_review_words)):
            lvt_i2i[i] = lst_review_words[i]
        
    lst_review_words = np.asarray(lst_review_words)

    return xu, xi, yr, yr_vec, lst_review_words, y_review_tf, y_review_mark, y_summary_index, y_summary_mark, y_summary_lvt, lvt_i2i

