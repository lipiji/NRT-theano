# -*- coding: utf-8 -*-
#pylint: skip-file
import sys
import os
import numpy as np
import theano
import theano.tensor as T
import gzip
import pickle
import json
import string
import argparse
import time
import datetime
from commons import *
import nltk
import operator
from nltk.tokenize import sent_tokenize

stop_words_f = "./data/stopwords_en.txt"

def load_stop_words():
    stop_words = {}
    with open(stop_words_f) as f:
        for line in f:
            line = line.strip("\n")
            stop_words[line] = 1
    return stop_words

def load_amazon():

    LFW_T = 3 #3 for music, 20 for others

    amazon_all = "./data/amazon_review/aggressive_dedup.json"
    amazon_mini = "./data/amazon_review/mini.json"
    amazon_core5 = "./data/amazon_review/kcore_5.json"
    amazon_movie = "./data/amazon_review/item_cats/reviews_Movies_and_TV_5.json"
    amazon_music = './data/Musical_Instruments_5.json'
    amazon_data = amazon_music

    size_corpus = 10000 #1697533. #41135700. #82836502.    
    stop_words = load_stop_words()

    dic = {}
    i2w = {}
    w2i = {}
    i2user = {}
    user2i = {}
    i2item = {}
    item2i = {}
    user2w = {}
    item2w = {}
    x_raw = []
    timestamps = []
    w2df = {}
 
    translator = str.maketrans('', '', string.punctuation) 
    with open(amazon_data) as f:
        i = 0.
        for line in f:
            try:
                line = line.strip("\n")
                a = json.loads(line)
                user_id = a["reviewerID"]
                item_id = a["asin"]
                review = a["reviewText"].lower()
                rating = a["overall"]
                summary = a["summary"].lower()
                unix_time = a["unixReviewTime"]
                raw_time = a["reviewTime"]

                if user_id not in user2i:
                    user2i[user_id] = len(i2user)
                    i2user[user2i[user_id]] = user_id
                if item_id not in item2i:
                    item2i[item_id] = len(i2item)
                    i2item[item2i[item_id]] = item_id
                
                if len(summary.split()) < MIN_SUM_LEN:
                    rev_sents = sent_tokenize(review)
                    for sent in rev_sents:
                        summary = sent
                        if len(summary.split()) >= MIN_SUM_LEN:
                            break
                
                if len(summary.split()) < MIN_SUM_LEN:
                    continue

                review = review.translate(translator)
                terms = review.split()
                review_words = []
                for t in terms:
                    if t in stop_words:
                        continue
                    review_words.append(t)
                    if t in dic:
                        dic[t] += 1
                    else:
                        dic[t] = 1
                    
                    if t in w2df:
                        w2df[t].add(len(timestamps))
                    else:
                        w2df[t] = set([item_id])

                summary = summary.translate(translator) 
                terms = summary.split() 
                for t in terms:
                    if t in dic:
                        dic[t] += 1
                    else:
                        dic[t] = 1

                    if t in w2df:
                        w2df[t].add(len(timestamps))
                    else:
                        w2df[t] = set([item_id])

                x_raw.append([user2i[user_id], item2i[item_id], rating, terms, review_words, unix_time, raw_time])
                timestamps.append(unix_time)
            except KeyError:
                print ("ops: " + line)
            i += 1.
            print ('\r{0}'.format(i / size_corpus) + " / 1 ",)
    
    timestamps.sort()
    spliter1 = timestamps[int(len(timestamps) * 0.8)]
    spliter2 = timestamps[int(len(timestamps) * 0.9)]
    x_raw_train = []
    x_raw_test = []
    x_raw_dev = []
    train_uid = {}
    train_iid = {}
    for i in range(len(x_raw)):
        if x_raw[i][5] < spliter1:
            x_raw_train.append(x_raw[i])
            train_uid[x_raw[i][0]] = 1
            train_iid[x_raw[i][1]] = 1
        elif x_raw[i][5] < spliter2:
            # test samples must be in train set
            if (x_raw[i][0] in train_uid) and (x_raw[i][1] in train_iid):
                x_raw_dev.append(x_raw[i])
        else:
            if (x_raw[i][0] in train_uid) and (x_raw[i][1] in train_iid):
                x_raw_test.append(x_raw[i])
    

    new_dic = {}
    w2i = {}
    iw2 = {}
    w2w = {}
    hfw = []
    new_w2df = {}

    if W_EOS not in dic:
        dic[W_EOS] = len(dic)
        w2df[W_EOS] = set([1])
    if W_UNK not in dic:
        dic[W_UNK] = len(dic)
        w2df[W_UNK] = set([1])

    for w in dic:
        if dic[w] >= LFW_T:
            w2i[w] = len(new_dic)
            i2w[w2i[w]] = w
            new_dic[w] = dic[w]
            w2w[w] = w
            new_w2df[w] = np.log10(len(timestamps) / len(w2df[w])) 
        else:
            w2w[w] = W_UNK

    new_w2df[W_EOS] = 1e-15
    new_w2df[W_UNK] = 1e-15

    sorted_x = sorted(new_dic.items(), key=operator.itemgetter(1), reverse=True)
    for w in sorted_x:
        hfw.append(w[0])

    for data_item in x_raw_train:
        [uid, iid, _, summ, review_words, _, _] = data_item
        for w in (summ + review_words):
            if uid in user2w:
                user2w[uid].add(w2i[w2w[w]])
            else:
                user2w[uid] = {w2i[w2w[w]]}
            if iid in item2w:
                item2w[iid].add(w2i[w2w[w]])
            else:
                item2w[iid] = {w2i[w2w[w]]}

    all_data = [x_raw_train, x_raw_dev, x_raw_test, dic, new_dic, hfw, i2w, w2i, w2w, new_w2df, i2user, user2i, i2item, item2i, user2w, item2w]

    o = open('./data/data_amazon.pkl', 'wb')
    pickle.dump(all_data, o, protocol = pickle.HIGHEST_PROTOCOL)
    o.close()
    
    print (len(dic), len(new_dic), len(i2w), len(w2i), len(w2w))
    print ("#dic=", len(dic), "#new_dic=", len(new_dic), "#user=", len(i2user), "#item=", len(i2item))
    print ("#x_raw=", len(x_raw), "#x_train=", len(x_raw_train),"#x_dev=", len(x_raw_dev),  "#x_test=", len(x_raw_test))
    x_raw = []

    ###################################################
    # movie
    #dic= 2719934 #user= 123960 #item= 50052
    #x_raw= 1697533 #x_train= 1357893 #x_test= 163529
    ###################################################
    # kcore-5
    #dic= 17511889 #user= 3035045 #item= 1569973
    #x_raw= 41135696 #x_train= 32906330 #x_test= 4928037
    ###################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="which dataset will be processed")
    args = parser.parse_args()

    if args.data == "amazon":
        load_amazon()
    elif args.data == "yelp":
        load_yelp()
    elif args.data == "yelp5":
        load_yelp5()
    else:
        print ("error: data")
