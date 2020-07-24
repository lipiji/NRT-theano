# -*- coding: utf-8 -*-
import os 
cudaid = 0
os.environ["THEANO_FLAGS"] = "device=cuda" + str(cudaid)

import sys
import  pickle
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

SUMM_PATH = "./result/"

def init_modules(data_type):
    options = {}
   
    options["is_predicting"] = False

    print ("is_predicting: ", options["is_predicting"], ", data = " + data_type)

    options["is_unicode"] = True
    
    print ("loading...")
    all_data = pickle.load(open("./data/data_"+ data_type + ".pkl", "rb"))
    [x_raw_train, x_raw_dev, x_raw_test, old_dic, dic, hfw, i2w, w2i, w2w, w2idf, i2user, user2i, i2item, item2i, user2w, item2w] = all_data

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


    consts["beam_size"] = 5

    consts["lr"] = 1.

    consts["print_size"] = 2#400 2 for demo

    print ("#train = ", len(x_raw_train), ", #dev = ", len(x_raw_dev), ", #test = ", len(x_raw_test))
    print ("#user = ", consts["num_user"], ", #item = ", consts["num_item"])
    print ("#dic = ", consts["dict_size"], ", #hfw = ", consts["hfw_dict_size"])

    modules = {}
    modules["all_data"] = all_data
    modules["hfw"] = hfw
    modules["w2i"] = w2i
    modules["i2w"] = i2w
    modules["w2w"] = w2w
    modules["u2ws"] = user2w
    modules["i2ws"] = item2w
    modules["w2idf"] = w2idf

    modules["eos_emb"] = modules["w2i"][W_EOS]
    modules["unk_emb"] = modules["w2i"][W_UNK]
    
    modules["optimizer"] = "adadelta"
    
    modules["stopwords"] = datar.load_stopwords("./data/stopwords_en.txt")

    return modules, consts, options

def beam_decode(result_paths, fname, batch_raw, model, modules, consts, options):
    
    print (fname)

    [SUMM_PATH_IDX, SUMM_PATH_WORD, MODEL_PATH_IDX, MODEL_PATH_WORD, RATING_PATH] = result_paths
    beam_size = consts["beam_size"]
    num_live = 1
    num_dead = 0
    samples = []
    sample_scores = np.zeros(beam_size, dtype = theano.config.floatX)

    last_traces = [[]]
    last_scores = np.zeros(1, dtype = theano.config.floatX)
    last_states = []

    XU, XI, Y_r, Y_r_vec, \
            Dict_lvt, Y_rev_tf, Y_rev_mark, \
            Y_sum_idx, Y_sum_mark, Y_sum_lvt, lvt_i2i = datar.get_batch_data(batch_raw, consts, options, modules)
    

    rat_pred, dec_state, emb_r, xu_emb, xi_emb = model.encode(XU, XI, consts["test_batch_size"])
    next_y = -np.ones((1, num_live, 1), dtype="int64")
    
    write_rating("".join((RATING_PATH, "rating.", fname)), str(rat_pred[0]) + "\t" + str(Y_r[0]))

    for step in range(consts["max_len_predict"]):
        tile_emb_r = np.tile(emb_r, (num_live, 1))
        tile_xu_emb = np.tile(xu_emb, (num_live, 1))
        tile_xi_emb = np.tile(xi_emb, (num_live, 1))

        y_pred, dec_state = model.decode_once(next_y, dec_state, tile_emb_r, tile_xu_emb, tile_xi_emb, num_live, Dict_lvt)
        
        dict_size = y_pred.shape[-1]
        cand_scores = last_scores + np.log(y_pred) # 分数最大越好
        cand_scores = cand_scores.flatten()
        idx_top_joint_scores = np.argsort(cand_scores)[-(beam_size - num_dead):]
        idx_last_traces = idx_top_joint_scores // dict_size
        idx_word_now = idx_top_joint_scores % dict_size
        top_joint_scores = cand_scores[idx_top_joint_scores]

        traces_now = []
        scores_now = np.zeros((beam_size - num_dead), dtype = theano.config.floatX)
        states_now = []
        for i, [j, k] in enumerate(zip(idx_last_traces, idx_word_now)):
            traces_now.append(last_traces[j] + [lvt_i2i[k]])
            scores_now[i] = copy.copy(top_joint_scores[i])
            states_now.append(copy.copy(dec_state[j, :]))

        num_live = 0
        last_traces = []
        last_scores = []
        last_states = []

        for i in range(len(traces_now)):
            if traces_now[i][-1] == modules["eos_emb"]:
                samples.append([str(e) for e in traces_now[i][:-1]])
                sample_scores[num_dead] = scores_now[i]
                num_dead += 1
            else:
                last_traces.append(traces_now[i])
                last_scores.append(scores_now[i])
                last_states.append(states_now[i])
                num_live += 1
        if num_live == 0 or num_dead >= beam_size:
            break

        last_scores = np.array(last_scores).reshape((num_live, 1))
        next_y = np.array([e[-1] for e in last_traces], dtype = "int64").reshape((1, num_live, 1))
        dec_state = np.array(last_states, dtype = theano.config.floatX).reshape((num_live, dec_state.shape[-1]))
        assert num_live + num_dead == beam_size

    if num_live > 0:
        for i in range(num_live):
            samples.append([str(e) for e in last_traces[i]])
            sample_scores[num_dead] = last_scores[i]
            num_dead += 1
    
    
    #weight by length
    for i in range(len(sample_scores)):
        sent_len = float(len(samples[i]))
        #sample_scores[i] = sample_scores[i] * math.exp(-sent_len / 8)
        # Google's Neural Machine Translation System
        lpy = math.pow((consts["min_len_predict"] + sent_len), 0.6) / math.pow((consts["min_len_predict"] + 1), 0.6)
        sample_scores[i] = sample_scores[i]  / lpy

    idx_sorted_scores = np.argsort(sample_scores) # 低分到高分
    ly = int(Y_sum_mark[:, 0].sum())
    y_true = Y_sum_idx[0 : ly, 0, 0].tolist()
    y_true = [str(i) for i in y_true[:-1]] # delete <eos>

    sorted_samples = []
    sorted_scores = []
    filter_idx = []
    for e in idx_sorted_scores:
        if len(samples[e]) >= consts["min_len_predict"] and (str(modules["unk_emb"]) not in set(samples[e])):
            filter_idx.append(e)
    if len(filter_idx) == 0:
        filter_idx = idx_sorted_scores
    for e in filter_idx:
        sorted_samples.append(samples[e])
        sorted_scores.append(sample_scores[e])

    num_samples = len(sorted_samples)
    if len(sorted_samples) == 1:
        sorted_samples = sorted_samples[0]
        num_samples = 1
    
    try:
        if beam_size == 1:
            write_summ("".join((SUMM_PATH_IDX, "summ.", fname)), sorted_samples, 1)
        else:
            write_summ("".join((SUMM_PATH_IDX, "summ.", fname)), sorted_samples[-1], 1)
	    
        write_summ("".join((MODEL_PATH_IDX, "model.", fname)), y_true, 1)

        write_summ("".join((SUMM_PATH_WORD, "summ.", fname)), sorted_samples, num_samples, modules["i2w"], sorted_scores)
	    
        write_summ("".join((MODEL_PATH_WORD, "model.", fname)), y_true, 1, modules["i2w"])
    except Exception as e:
        print ("error", fname)
        raise


def predict(data_type, model, modules, consts, options):
    predict_rmse(data_type, model, modules, consts, options)
    predict_summary(data_type, model, modules, consts, options)


def predict_summary(data_type, model, modules, consts, options):
    print ("start predicting summary ...")
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
    
    rebuild_dir(SUMM_PATH_IDX)
    rebuild_dir(SUMM_PATH_WORD)
    rebuild_dir(MODEL_PATH_IDX)
    rebuild_dir(MODEL_PATH_WORD)
    rebuild_dir(RATING_PATH)

    result_paths = [SUMM_PATH_IDX, SUMM_PATH_WORD, MODEL_PATH_IDX, MODEL_PATH_WORD, RATING_PATH]


    [x_raw_train, x_raw_dev, x_raw_test, old_dic, dic, hfw, i2w, w2i, w2w, w2idf, i2user, user2i, i2item, item2i, user2w, item2w] = modules["all_data"]
    batch_list = datar.get_batch_index(len(x_raw_dev), consts["test_batch_size"], options["is_predicting"])
    error_i = 0
    for batch_index in batch_list:
        local_batch_size = len(batch_index)
        batch_raw = [x_raw_dev[bxi] for bxi in batch_index]
        beam_decode(result_paths, str(batch_index[0]), batch_raw, model, modules, consts, options)
        error_i += 1
        if error_i == consts["test_samples"]:
            break

def predict_rmse(data_type, model, modules, consts, options):
    print ("start predicting rmse...")

    [x_raw_train, x_raw_dev, x_raw_test, old_dic, dic, hfw, i2w, w2i, w2w, w2idf, i2user, user2i, i2item, item2i, user2w, item2w] = modules["all_data"]
    batch_list = datar.get_batch_index(len(x_raw_dev), consts["batch_size"])
   
    ae = 0
    se = 0
    se_i = 0
    
    for batch_index in batch_list:
        local_batch_size = len(batch_index)
        batch_raw = [x_raw_dev[bxi] for bxi in batch_index]
        XU, XI, Y_r, Y_r_vec, \
            Dict_lvt, Y_rev_tf, Y_rev_mark, \
            Y_sum_idx, Y_sum_mark, Y_sum_lvt, lvt_i2i = datar.get_batch_data(batch_raw, consts, options, modules)

        rat_pred, dec_state, emb_r, xu_emb, xi_emb = model.encode(XU, XI, len(batch_raw))

        for i in range(len(batch_raw)):
            se += (rat_pred[i, 0] -  Y_r[i]) * (rat_pred[i, 0] -  Y_r[i])
            ae += abs(rat_pred[i, 0] -  Y_r[i])
        se_i += len(batch_raw)
        
        #print "MAE now = ", ae / se_i,
        #print "RMSE now = ", np.sqrt(se / se_i), 
        #print "#X = ", se_i

    print ("MAE = ", ae / se_i,)
    print ("RMSE = ", np.sqrt(se / se_i), )
    print ("#X = ", se_i)

def run(data_type, existing_model_name = None):
    modules, consts, options = init_modules(data_type)
    
    if options["is_predicting"]:
        need_load_model = True
        training_model = False
    else:
        need_load_model = False
        training_model = True

    print ("compiling...")
    model = NeuralRecsys(modules, consts, options)
    
    if need_load_model:
        if not existing_model_name:
            existing_model_name = "m_amazon.0.20"
        print ("loading model...", existing_model_name)
        load_model("./model/" + existing_model_name, model)

    if training_model:
        print ("training......")
        [x_raw_train, x_raw_dev, x_raw_test, old_dic, dic, hfw, i2w, w2i, w2w, w2idf, i2user, user2i, i2item, item2i, user2w, item2w] = modules["all_data"]
        for epoch in range(0, 50):
            
            if epoch > 2:
                consts["lr"] = 0.1

            start = time.time()
            batch_list = datar.get_batch_index(len(x_raw_train), consts["batch_size"])
            error = 0
            error_i = 0
            error_rmse = 0
            e_nll = 0
            e_nllu = 0
            e_nlli = 0
            e_cce = 0
            e_mae = 0
            e_l2n = 0
            for batch_index in batch_list:
                local_batch_size = len(batch_index)
                batch_raw = [x_raw_train[bxi] for bxi in batch_index]
                load_time = time.time()
                XU, XI, Y_r, Y_r_vec, \
                        Dict_lvt, Y_rev_tf, Y_rev_mark, \
                        Y_sum_idx, Y_sum_mark, Y_sum_lvt, lvt_i2i = datar.get_batch_data(batch_raw, consts, options, modules)
                cost, mse, nll, nllu, nlli, cce, mae, l2n, rat_pred, sum_pred = model.train(XU, XI, Y_r, Y_r_vec, \
                                                   Dict_lvt, Y_rev_tf, Y_rev_mark, \
                                                   Y_sum_idx, Y_sum_mark, Y_sum_lvt, \
                                                   local_batch_size, consts["lr"])
                #print cost, mse, nll, cce, l2n
                error += cost
                error_rmse += math.sqrt(mse)
                e_nll += nll
                e_nllu += nllu
                e_nlli += nlli
                e_cce += cce
                e_mae += mae
                e_l2n += l2n
                error_i += 1
                if error_i % consts["print_size"] == 0:
                    print ("epoch=", epoch, ", Error now = ", error / error_i, ", ",)
                    print ("RMSE now = ", error_rmse / error_i, ", ",)
                    print ("MAE now = ", e_mae / error_i, ", ",)
                    print ("NLL now = ", e_nll / error_i, ", ",)
                    print ("NLLu now = ", e_nllu / error_i, ", ",)
                    print ("NLLi now = ", e_nlli / error_i, ", ",)
                    print ("CCE now = ", e_cce / error_i, ", ",)
                    print ("L2 now = ", e_l2n / error_i, ", ",)
                    print (error_i, "/", len(batch_list), "time=", time.time()-start)
                    save_model("./model/m_" + data_type + "." + str(epoch) + "." + str(int(error_i / consts["print_size"])), model)
                    print_sent_dec(sum_pred, Y_sum_idx, Y_sum_mark, modules, consts, options, lvt_i2i)
            
            save_model("./model/m_" + data_type + "." + str(epoch), model)
            #print_sent_dec(sum_pred, Y_sum_idx, Y_sum_mark, modules, consts, options, lvt_i2i)
            print ("epoch=", epoch, ", Error = ", error / len(batch_list),)
            print (", RMSE = ", error_rmse / len(batch_list), ", time=", time.time()-start)
    else:
        predict(data_type, model, modules, consts, options)
        
if __name__ == "__main__":
    np.set_printoptions(threshold = np.inf)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="amazon", help="which dataset", )
    parser.add_argument("-m", "--model", default=None, help="existing_model_name")
    args = parser.parse_args()

    data_type = "amazon"
    existing_model_name = None

    if args.data and args.data == "yelp":
        data_type = "yelp"
    if args.data and args.data == "books":
        data_type = "books"
    if args.data and args.data == "elec":
        data_type = "electronics"
    if args.data and args.data == "yelp5":
        data_type = "yelp5"
    if args.model:
        existing_model_name = "m_" + data_type + "." + str(args.model)

    print (data_type, existing_model_name)
    run(data_type, existing_model_name)
