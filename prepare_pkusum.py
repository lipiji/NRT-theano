# -*- coding: utf-8 -*-
import sys
import os 
import cPickle as pickle
from nrec import *
from utils_pg import *
import data as datar
import math
import time
import argparse
from commons import *
import copy
from os import makedirs
from os.path import exists

SUMM_PATH = "/misc/projdata12/info_fil/pjli/workspace/recsys_summ/result/"

def init_modules(data_type):
    use_gpu(0)
    
    options = {}
   
    options["is_predicting"] = True
    print "is_predicting: ", options["is_predicting"], ", data = " + data_type

    options["is_unicode"] = True
    
    print "loading..."
    all_data = pickle.load(open("./data/data_"+ data_type + ".pkl", "rb"))
    [x_raw_train, x_raw_test, old_dic, dic, hfw, i2w, w2i, w2w, i2user, user2i, i2item, item2i, user2w, item2w] = all_data

    consts = {}
    consts["num_user"] = len(user2i)
    consts["num_item"] = len(item2i)
    consts["dict_size"] = len(dic)
    consts["hfw_dict_size"] = len(hfw)
    consts["lvt_dict_size"] = LVT_DICT_SIZE
    consts["len_y"] = LEN_Y
    if data_type == "yelp":
        consts["len_y"] = LEN_Y_YELP
        consts["lvt_dict_size"] = LVT_DICT_SIZE_YELP

    consts["dim_x"] = DIM_X
    consts["dim_w"] = DIM_W
    consts["dim_h"] = DIM_H
    consts["dim_r"] = DIM_RATING
    consts["max_len_predict"] = consts["len_y"]
    consts["min_len_predict"] = 1

    consts["batch_size"] = 200
    
    consts["test_batch_size"] = 1
    consts["test_samples"] = 2000


    consts["beam_size"] = 10

    consts["lr"] = 0.01

    consts["print_size"] = 400

    print "#train = ", len(x_raw_train), ", #test = ", len(x_raw_test)
    print "#user = ", consts["num_user"], ", #item = ", consts["num_item"]
    print "#dic = ", consts["dict_size"], ", #hfw = ", consts["hfw_dict_size"]

    modules = {}
    modules["all_data"] = all_data
    modules["hfw"] = hfw
    modules["w2i"] = w2i
    modules["i2w"] = i2w
    modules["w2w"] = w2w
    modules["u2ws"] = user2w
    modules["i2ws"] = item2w
    modules["eos_emb"] = modules["w2i"][W_EOS]
    modules["unk_emb"] = modules["w2i"][W_UNK]
    
    modules["optimizer"] = "adadelta"
    
    modules["stopwords"] = datar.load_stopwords("/misc/projdata12/info_fil/pjli/data/stopwords/stopwords_en.txt")

    return modules, consts, options

def predict(data_type, modules, consts, options):
    predict_summary(data_type, modules, consts, options)

def predict_summary(data_type,  modules, consts, options):
    print "start predicting summary ..."
    SUMM_PATH_IDX = SUMM_PATH + data_type + "/sum_idx/"
    SUMM_PATH_WORD = SUMM_PATH + data_type + "/sum_word/"
    MODEL_PATH_IDX = SUMM_PATH + data_type + "/model_idx/"
    MODEL_PATH_WORD = SUMM_PATH + data_type + "/model_word/"
    RATING_PATH = SUMM_PATH + data_type + "/rating/"
    
    if not exists(SUMM_PATH_IDX):
        makedirs(SUMM_PATH_IDX)
    if not exists(SUMM_PATH_WORD):
        makedirs(SUMM_PATH_WORD)
    if not exists(MODEL_PATH_IDX):
        makedirs(MODEL_PATH_IDX)
    if not exists(MODEL_PATH_WORD):
        makedirs(MODEL_PATH_WORD)
    if not exists(RATING_PATH):
        makedirs(RATING_PATH)

    result_paths = [SUMM_PATH_IDX, SUMM_PATH_WORD, MODEL_PATH_IDX, MODEL_PATH_WORD, RATING_PATH]

    f = open(SUMM_PATH + data_type + "/" + data_type + "_summ_test_ui.txt", "wb")
    [x_raw_train, x_raw_test, old_dic, dic, hfw, i2w, w2i, w2w, i2user, user2i, i2item, item2i, user2w, item2w] = modules["all_data"]
    for i in xrange(len(x_raw_test)):
        [user_id, item_id, rating, terms, review_words, unix_time, raw_time] = x_raw_test[i]
        f.write(i2user[user_id] + "," + i2item[item_id] + "," + str(rating) + "," + str(unix_time)  + "\n")
    f.close()

    

def run(data_type, existing_model_name = None):
    modules, consts, options = init_modules(data_type)
    predict(data_type,  modules, consts, options)
        
if __name__ == "__main__":
    existing_model_name = None
    run("yelp", existing_model_name)
    run("amazon", existing_model_name)
    run("books", existing_model_name)
    run("electronics", existing_model_name)
