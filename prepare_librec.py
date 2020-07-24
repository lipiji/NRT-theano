# -*- coding: utf-8 -*-
import sys
import os 
import cPickle as pickle
import time
import argparse

data_type = "amazon"

print "loading..."
all_data = pickle.load(open("./data/data_"+ data_type + ".pkl", "rb"))
[x_raw_train, x_raw_test, old_dic, dic, hfw, i2w, w2i, w2w, i2user, user2i, i2item, item2i, user2w, item2w] = all_data

with open('./data/movies/test.txt', 'wb') as f:
    for i in xrange(len(x_raw_test)):
        uid, iid, r, _, _, time, _ = x_raw_test[i]
        f.write(str(uid) + "\t" + str(iid) + "\t" + str(r) + "\t" + str(int(time)) + "\n")
print len(x_raw_test)


with open('./data/movies/train.txt', 'wb') as f:
    for i in xrange(len(x_raw_train)):
        uid, iid, r, _, _, time, _ = x_raw_train[i]
        f.write(str(uid) + "\t" + str(iid) + "\t" + str(r) + "\t" + str(int(time)) + "\n")
print len(x_raw_train)

