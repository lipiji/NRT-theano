# -*- coding: utf-8 -*-
#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import sys

from utils_pg import *
from updates import *


class NeuralRecsys(object):
    def __init__(self, modules, consts, options):
        self.is_predicting = options["is_predicting"]
        self.optimizer = modules["optimizer"]

        self.xu = T.vector("xu", dtype = "int64")
        self.xi = T.vector("xi", dtype = "int64")
        self.y_rating = T.vector("y_rating")
        self.lvt_dict = T.lvector("lvt_dict")
        self.y_rev_mark = T.matrix("y_rev_mark")
        self.y_rev_tf = T.matrix("y_rev_tf")
        self.y_rat_vec = T.matrix("y_rat_vec")
        self.y_sum_idx = T.tensor3("y_sum_idx", dtype = "int64")
        self.y_sum_mark = T.matrix("y_sum_mark")
        if not self.is_predicting:
            self.y_sum_lvt = T.tensor3("y_sum_lvt", dtype = "int64")

        self.lr = T.scalar("lr")
        self.batch_size = T.iscalar("batch_size")

        self.num_user = consts["num_user"]
        self.num_item = consts["num_item"]
        self.dict_size = consts["hfw_dict_size"]
        self.lvt_dict_size = consts["lvt_dict_size"] 
        self.dim_x = consts["dim_x"]
        self.dim_w = consts["dim_w"]
        self.dim_h = consts["dim_h"]
        self.dim_r = consts["dim_r"]
        self.len_y = consts["len_y"]

        self.params = []
        self.sub_params = []
        
        self.define_layers(modules, consts, options)
        if not self.is_predicting:
            self.define_train_funcs(modules, consts, options)

    def define_layers(self, modules, consts, options):
        self.emb_u = init_weights((self.num_user, self.dim_x), "emb_u", sample = "normal")
        self.emb_i = init_weights((self.num_item, self.dim_x), "emb_i", sample = "normal")
    
        self.params += [self.emb_u, self.emb_i]

        xu_flat = self.xu.flatten()
        xi_flat = self.xi.flatten()
        xu_emb = self.emb_u[xu_flat, :]
        xu_emb = T.reshape(xu_emb, (self.batch_size, self.dim_x))
        xi_emb = self.emb_i[xi_flat, :]
        xi_emb = T.reshape(xi_emb, (self.batch_size, self.dim_x))

        ## user-doc
        self.W_ud_uh = init_weights((self.dim_x, self.dim_h), "W_ud_uh")
        self.b_ud_uh = init_bias(self.dim_h, "b_ud_uh")
        self.W_ud_hd = init_weights((self.dim_h, self.dict_size), "W_ud_hd")
        self.b_ud_hd = init_bias(self.dict_size, "b_ud_hd")
        self.sub_W_ud_hd = self.W_ud_hd[:, self.lvt_dict]
        self.sub_b_ud_hd = self.b_ud_hd[self.lvt_dict]

        self.params += [self.W_ud_uh, self.b_ud_uh]
        self.sub_params = [(self.W_ud_hd, self.sub_W_ud_hd, (self.dim_h, self.lvt_dict_size)),
                           (self.b_ud_hd, self.sub_b_ud_hd, (self.lvt_dict_size,))]
       
        h_ru = T.nnet.sigmoid(T.dot(xu_emb, self.W_ud_uh) + self.b_ud_uh)
        
        self.W_ud_uh2 = init_weights((self.dim_h, self.dim_h), "W_ud_uh2")
        self.b_ud_uh2 = init_bias(self.dim_h, "b_ud_uh2")
        self.params += [self.W_ud_uh2, self.b_ud_uh2]
        h_ru2 = T.nnet.sigmoid(T.dot(h_ru, self.W_ud_uh2) + self.b_ud_uh2) ###
       
        self.W_ud_uh3 = init_weights((self.dim_h, self.dim_h), "W_ud_uh3")
        self.b_ud_uh3 = init_bias(self.dim_h, "b_ud_uh3")
        self.params += [self.W_ud_uh3, self.b_ud_uh3]
        h_ru3 = T.nnet.sigmoid(T.dot(h_ru2, self.W_ud_uh3) + self.b_ud_uh3) ##############

        ## item-doc
        self.W_id_ih = init_weights((self.dim_x, self.dim_h), "W_id_ih")
        self.b_id_ih = init_bias(self.dim_h, "b_id_ih")
        self.W_id_hd = init_weights((self.dim_h, self.dict_size), "W_id_hd")
        self.b_id_hd = init_bias(self.dict_size, "b_id_hd")
        self.sub_W_id_hd = self.W_id_hd[:, self.lvt_dict]
        self.sub_b_id_hd = self.b_id_hd[self.lvt_dict]
        self.params += [self.W_id_ih, self.b_id_ih]
        self.sub_params += [(self.W_id_hd, self.sub_W_id_hd, (self.dim_h, self.lvt_dict_size)),
                           (self.b_id_hd, self.sub_b_id_hd, (self.lvt_dict_size,))]

        h_ri = T.nnet.sigmoid(T.dot(xi_emb, self.W_id_ih) + self.b_id_ih)
        
        self.W_id_ih2 = init_weights((self.dim_h, self.dim_h), "W_id_ih2")
        self.b_id_ih2 = init_bias(self.dim_h, "b_id_ih2")
        self.params += [self.W_id_ih2, self.b_id_ih2]
        h_ri2 = T.nnet.sigmoid(T.dot(h_ri, self.W_id_ih2) + self.b_id_ih2) ###
        
        self.W_id_ih3 = init_weights((self.dim_h, self.dim_h), "W_id_ih3")
        self.b_id_ih3 = init_bias(self.dim_h, "b_id_ih3")
        self.params += [self.W_id_ih3, self.b_id_ih3]
        h_ri3 = T.nnet.sigmoid(T.dot(h_ri2, self.W_id_ih3) + self.b_id_ih3) #####

        ## user-item-doc
        self.W_uid_uh = init_weights((self.dim_x, self.dim_h), "W_uid_uh")
        self.W_uid_ih = init_weights((self.dim_x, self.dim_h), "W_uid_ih")
        self.b_uid_uih = init_bias(self.dim_h, "b_uid_uih")
        self.W_uid_hd = init_weights((self.dim_h, self.dict_size), "W_uid_hd")
        self.b_uid_hd = init_bias(self.dict_size, "b_uid_hd")
        self.sub_W_uid_hd = self.W_uid_hd[:, self.lvt_dict]
        self.sub_b_uid_hd = self.b_uid_hd[self.lvt_dict]
        self.params += [self.W_uid_uh, self.W_uid_ih, self.b_uid_uih]
        self.sub_params += [(self.W_uid_hd, self.sub_W_uid_hd, (self.dim_h, self.lvt_dict_size)),
                           (self.b_uid_hd, self.sub_b_uid_hd, (self.lvt_dict_size,))]

        h_rui = T.nnet.sigmoid(T.dot(xu_emb, self.W_uid_uh) + T.dot(xi_emb, self.W_uid_ih) + self.b_uid_uih)
        
        self.W_uid_uih2 = init_weights((self.dim_h, self.dim_h), "W_uid_uih2")
        self.b_uid_uih2 = init_bias(self.dim_h, "b_uid_uih2")
        self.params += [self.W_uid_uih2, self.b_uid_uih2]
        h_rui2 = T.nnet.sigmoid(T.dot(h_rui, self.W_uid_uih2) + self.b_uid_uih2) ###
        self.W_uid_uih3 = init_weights((self.dim_h, self.dim_h), "W_uid_uih3")
        self.b_uid_uih3 = init_bias(self.dim_h, "b_uid_uih3")
        self.params += [self.W_uid_uih3, self.b_uid_uih3]
        h_rui3 = T.nnet.sigmoid(T.dot(h_rui2, self.W_uid_uih3) + self.b_uid_uih3) #####

        # predict 1: rating
        self.W_r_uh = init_weights((self.dim_x, self.dim_h), "W_r_uh")
        self.W_r_ih = init_weights((self.dim_x, self.dim_h), "W_r_ih")
        self.W_r_uih = init_weights((self.dim_x, self.dim_h), "W_r_uih")
        self.b_r_uih = init_bias(self.dim_h, "b_r_uih")
        self.W_r_hr = init_weights((self.dim_h, 1), "W_r_hr")
        self.W_r_ur = init_weights((self.dim_x, 1), "W_r_ur")
        self.W_r_ir = init_weights((self.dim_x, 1), "W_r_ir")
        self.b_r_hr = init_bias(1, "b_r_hr")
        self.params += [self.W_r_uh, self.W_r_ih, self.W_r_uih, self.b_r_uih, self.W_r_hr, self.b_r_hr, self.W_r_ur, self.W_r_ir]

        h_r = T.nnet.sigmoid(T.dot(xu_emb, self.W_r_uh) + T.dot(xi_emb, self.W_r_ih) + T.dot(xu_emb * xi_emb,  self.W_r_uih) +  self.b_r_uih)

        self.W_r_hh2 = init_weights((self.dim_h, self.dim_h), "W_r_hh2")
        self.b_r_hh2 = init_bias(self.dim_h, "b_r_hh2")
        self.params += [self.W_r_hh2, self.b_r_hh2]
        h_r2 = T.nnet.sigmoid(T.dot(h_r, self.W_r_hh2) + self.b_r_hh2) ###
        self.W_r_hh3 = init_weights((self.dim_h, self.dim_h), "W_r_hh3")
        self.b_r_hh3 = init_bias(self.dim_h, "b_r_hh3")
        self.params += [self.W_r_hh3, self.b_r_hh3]
        h_r3 = T.nnet.sigmoid(T.dot(h_r2, self.W_r_hh3) + self.b_r_hh3) #####

        ## merge them
        self.review_i = T.nnet.softmax(T.dot(h_ri3, self.sub_W_id_hd) + self.sub_b_id_hd)
        self.review_u = T.nnet.softmax(T.dot(h_ru3, self.sub_W_ud_hd) + self.sub_b_ud_hd)
        self.review = T.nnet.softmax(T.dot(h_rui3, self.sub_W_uid_hd) + self.sub_b_uid_hd)
        
        self.W_r_uhh = init_weights((self.dim_h, self.dim_h), "W_r_uhh")
        self.W_r_ihh = init_weights((self.dim_h, self.dim_h), "W_r_ihh")
        self.W_r_uihh = init_weights((self.dim_h, self.dim_h), "W_r_uihh")
        self.W_r_rhh = init_weights((self.dim_h, self.dim_h), "W_r_rhh")
        self.b_r_uirh = init_bias(self.dim_h, "b_r_uirh")
        self.params += [self.W_r_uhh, self.W_r_ihh, self.W_r_uihh, self.W_r_rhh, self.b_r_uirh]

        h_r4 = T.nnet.sigmoid(T.dot(h_ru3, self.W_r_uhh) + T.dot(h_ri3, self.W_r_ihh) + T.dot(h_r3, self.W_r_rhh) + T.dot(h_rui3, self.W_r_uihh) + self.b_r_uirh)
        self.r_pred = T.dot(h_r4, self.W_r_hr) + T.dot(xu_emb, self.W_r_ur) + T.dot(xi_emb, self.W_r_ir) + self.b_r_hr
    
        # predict 3: summary
        self.emb_w = init_weights((self.dict_size, self.dim_w), "emb_w", sample = "normal")
        
        if self.is_predicting:
            emb_r = T.zeros((self.batch_size, self.dim_r))
            r_idx = T.round(self.r_pred)
            r_idx = T.cast(r_idx, "int64")
            r_idx = T.clip(r_idx, 0, 5) 

            emb_r = T.inc_subtensor(emb_r[:, r_idx], T.cast(1.0, dtype = theano.config.floatX))
        else:
            emb_r = self.y_rat_vec
        
        self.W_s_uh = init_weights((self.dim_x, self.dim_h), "W_s_uh")
        self.W_s_ih = init_weights((self.dim_x, self.dim_h), "W_s_ih")
        self.b_s_uih = init_bias(self.dim_h, "b_s_uih")
        self.W_s_rh = init_weights((self.dim_r, self.dim_h), "W_s_rh")
        self.W_s_uch = init_weights((self.dim_h, self.dim_h), "W_s_uch")
        self.W_s_ich = init_weights((self.dim_h, self.dim_h), "W_s_ich")
        self.W_s_uich = init_weights((self.dim_h, self.dim_h), "W_s_uich")
        
        self.params += [self.emb_w, self.W_s_uh, self.W_s_ih, self.b_s_uih, self.W_s_rh, self.W_s_uch, self.W_s_ich, self.W_s_uich]
        
        ## initialize hidden state
        h_s = T.tanh(T.dot(xu_emb, self.W_s_uh) + T.dot(xi_emb, self.W_s_ih) + T.dot(emb_r, self.W_s_rh) \
            + T.dot(h_ru3, self.W_s_uch) + T.dot(h_ri3, self.W_s_ich) + T.dot(h_rui3, self.W_s_uich) + self.b_s_uih)
        
        
        if self.is_predicting:
            inputs = [self.xu, self.xi, self.batch_size]
            self.encode = theano.function(inputs = inputs,
                    outputs = [self.r_pred, h_s, emb_r, xu_emb, xi_emb],
                    on_unused_input = 'ignore')


        y_flat = self.y_sum_idx.flatten()
        if self.is_predicting:
            y_emb = ifelse(T.lt(T.sum(y_flat), 0),
                    T.zeros((self.batch_size, self.dim_w)), self.emb_w[y_flat, :])
            y_emb = T.reshape(y_emb, (self.batch_size, self.dim_w))
        else:
            y_emb = self.emb_w[y_flat, :]
            y_emb = T.reshape(y_emb, (self.len_y, self.batch_size, self.dim_w))
            y_shifted = T.zeros_like(y_emb)
            y_shifted = T.set_subtensor(y_shifted[1:, :, :], y_emb[:-1, :, :])
            y_emb = y_shifted
        
        # gru decoder
        self.W_yh = init_weights((self.dim_w, self.dim_h), "W_yh", num_concatenate = 2, axis_concatenate = 1)
        self.b_yh = init_bias(self.dim_h, "b_yh", num_concatenate = 2)
        self.W_yhx = init_weights((self.dim_w, self.dim_h), "W_yhx")
        self.b_yhx = init_bias(self.dim_h, "b_yhx")
        self.W_hru = init_weights((self.dim_h, self.dim_h), "W_hru", "ortho", num_concatenate = 2, axis_concatenate = 1)
        self.W_hh = init_weights((self.dim_h, self.dim_h), "W_hh", "ortho")

        self.params += [self.W_yh, self.b_yh, self.W_yhx, self.b_yhx, self.W_hru, self.W_hh]

        def _slice(x, n):
            if x.ndim == 3:
                return x[:, :, n * self.dim_h : (n + 1) * self.dim_h]
            return x[:, n * self.dim_h : (n + 1) * self.dim_h]

        dec_in_x = T.dot(y_emb, self.W_yh) + self.b_yh
        dec_in_xx = T.dot(y_emb, self.W_yhx) + self.b_yhx

        def _active(x, xx, y_mask, pre_h, W_hru, W_hh):
            tmp1 = T.nnet.sigmoid(T.dot(pre_h, W_hru) + x)
            r1 = _slice(tmp1, 0)
            u1 = _slice(tmp1, 1)
            h1 = T.tanh(T.dot(pre_h * r1, W_hh) + xx)
            h1 = u1 * pre_h + (1.0 - u1) * h1
            h1 = y_mask[:, None] * h1 #+ (1.0 - y_mask) * pre_h
            return h1

        if self.is_predicting:
            self.y_sum_mark = T.ones((self.batch_size, 1))
        sequences = [dec_in_x, dec_in_xx, self.y_sum_mark]
        non_sequences = [self.W_hru, self.W_hh]

        if self.is_predicting:
            self.dec_next_state = T.matrix("dec_next_state")
            print ("use one-step decoder")
            hs = _active(*(sequences + [self.dec_next_state] + non_sequences))
        else:
            hs, _ = theano.scan(_active, 
                    sequences = sequences,
                    outputs_info = [h_s], 
                    non_sequences = non_sequences,
                    allow_gc = False, strict = True)
        
        # output layer
        self.W_hy = init_weights((self.dim_h, self.dim_h), "W_hy")
        self.W_yy = init_weights((self.dim_w, self.dim_h), "W_yy")
        self.W_uy = init_weights((self.dim_x, self.dim_h), "W_uy")
        self.W_iy = init_weights((self.dim_x, self.dim_h), "W_iy")
        self.W_ry = init_weights((self.dim_r, self.dim_h), "W_ry")
        self.b_hy = init_bias(self.dim_h, "b_hy")
        
        self.W_logit = init_weights((self.dim_h, self.dict_size), "W_logit")
        self.b_logit = init_bias(self.dict_size, "b_logit")

        self.sub_W_logit = self.W_logit[:, self.lvt_dict]
        self.sub_b_logit = self.b_logit[self.lvt_dict]
            
        self.params += [self.W_hy, self.W_yy, self.W_uy, self.W_iy, self.W_ry, self.b_hy]
        self.sub_params += [(self.W_logit, self.sub_W_logit, (self.dim_h, self.lvt_dict_size)),
                           (self.b_logit, self.sub_b_logit, (self.lvt_dict_size,))]

        logit = T.tanh(T.dot(hs, self.W_hy) + T.dot(y_emb, self.W_yy) \
                     + T.dot(xu_emb, self.W_uy) + T.dot(xi_emb, self.W_iy) \
                     + T.dot(emb_r, self.W_ry) \
                     + self.b_hy)
        logit = T.dot(logit, self.sub_W_logit) + self.sub_b_logit
        
        old_shape = logit.shape
        if self.is_predicting:
            self.y_sum_pred = T.nnet.softmax(logit)
        else:
            self.y_sum_pred = T.nnet.softmax(logit.reshape((old_shape[0] * old_shape[1], old_shape[2]))).reshape(old_shape)

        if self.is_predicting:
            emb_r_enc = T.matrix("emb_r_enc")
            xu_emb_enc = T.matrix("xu_emb_enc")
            xi_emb_enc = T.matrix("xi_emb_enc")
            
            inputs = [self.y_sum_idx, self.dec_next_state, emb_r, xu_emb, xi_emb, self.batch_size, self.lvt_dict]
            self.decode_once = theano.function(inputs = inputs,
                    outputs = [self.y_sum_pred, hs],
                    on_unused_input = 'ignore')
    
    def l2_norm(self):
        return T.sum(self.emb_u**2) + T.sum(self.emb_i**2)

    def cost_mse(self, pred, label):
        cost = T.mean((pred - label) ** 2)
        return cost

    def cost_mae(self, pred, label):
        return T.mean(T.abs_(pred - label))

    def cost_nll(self, pred, label, mark):
        cost = -T.log(pred) * label * mark
        cost = T.mean(T.sum(cost, axis = 1))
        return cost

    def cost_kld(self, pred, label):
        cost = pred * T.log(pred / T.nnet.softmax(label))
        cost = T.mean(T.sum(cost, axis = 1))
        return cost

    def categorical_crossentropy(self, modules):
        y_flat = self.y_sum_lvt.flatten()
        y_flat_idx = T.arange(y_flat.shape[0]) * self.lvt_dict_size + y_flat
        cost = -T.log(self.y_sum_pred.flatten()[y_flat_idx])
        cost = cost.reshape(self.y_sum_idx.shape)
        y_sum_mark = T.reshape(self.y_sum_mark, self.y_sum_idx.shape)
        cost = T.sum(cost * y_sum_mark, axis = 0)
        cost = T.mean(cost.reshape((self.batch_size, )))
        return cost 

    def define_train_funcs(self, modules, consts, options):
        mse = self.cost_mse(self.r_pred, self.y_rating.reshape((self.batch_size, 1)))
        nll = self.cost_nll(self.review, self.y_rev_tf, self.y_rev_mark)
        nllu = self.cost_nll(self.review_u, self.y_rev_tf, self.y_rev_mark)
        nlli = self.cost_nll(self.review_i, self.y_rev_tf, self.y_rev_mark)
        cce = self.categorical_crossentropy(modules)

        mae = self.cost_mae(self.r_pred, self.y_rating.reshape((self.batch_size, 1)))
        l2n = self.l2_norm()

        cost = mse + nll + nllu + nlli  + cce

        gparams = []
        for param in self.params:
            gparams.append(T.clip(T.grad(cost, param), -10, 10))
        sub_gparams = []
        for param in self.sub_params:
            sub_gparams.append(T.clip(T.grad(cost, param[1]), -10, 10))

        optimizer = eval(self.optimizer)
        updates = optimizer(self.params, gparams, self.sub_params, sub_gparams, self.lr)

        inputs = [self.xu, self.xi, self.y_rating, self.y_rat_vec, \
                  self.lvt_dict, self.y_rev_mark, self.y_rev_tf, \
                  self.y_sum_idx, self.y_sum_mark, self.y_sum_lvt,\
                  self.batch_size]
        
        self.train = theano.function(
                inputs = inputs + [self.lr],
                outputs = [cost, mse, nll, nllu, nlli, cce, mae, l2n, self.r_pred, self.y_sum_pred],
                updates = updates,
                on_unused_input = 'ignore')

        
        
