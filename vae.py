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
        self.y_rev_i = T.matrix("y_rev_i")
        self.is_train = T.iscalar('is_train') 
        self.lr = T.scalar("lr")
        self.batch_size = T.iscalar("batch_size")

        self.num_user = consts["num_user"]
        self.num_item = consts["num_item"]
        self.dict_size = consts["hfw_dict_size"]
        self.dim_x = consts["dim_x"]
        self.dim_w = consts["dim_w"]
        self.dim_h = consts["dim_h"]
        self.dim_r = consts["dim_r"]
        self.len_y = consts["len_y"]
        self.dim_z = 100

        self.params = []
        self.sub_params = []
        
        self.define_layers(modules, consts, options)
        if not self.is_predicting:
            self.define_train_funcs(modules, consts, options)

    def define_layers(self, modules, consts, options):
        p = 0.2
        rng = np.random.RandomState(1234)
        srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
        
        self.emb_u = init_weights((self.num_user, self.dim_x), "emb_u", sample = "normal")
        self.emb_i = init_weights((self.num_item, self.dim_z), "emb_i", sample = "normal")
        self.params += [self.emb_u, self.emb_i]

        xu_flat = self.xu.flatten()
        xu_emb = self.emb_u[xu_flat, :]
        xu_emb = T.reshape(xu_emb, (self.batch_size, self.dim_x))


        #VAE for item embedding
        self.W_xh = init_weights((self.dict_size, self.dim_h), "W_xh")
        self.b_xh = init_bias(self.dim_h, "b_xh")

        self.W_hu = init_weights((self.dim_h, self.dim_z), "W_hu")
        self.b_hu = init_bias(self.dim_z, "b_hu")
        self.W_hsigma = init_weights((self.dim_h, self.dim_z), "W_hsigma")
        self.b_hsigma = init_bias(self.dim_z, "b_hsigma")

        self.W_zh = init_weights((self.dim_z, self.dim_h), "W_zh")
        self.b_zh = init_bias(self.dim_h, "b_zh")
 
        self.params += [self.W_xh, self.b_xh, self.W_hu, self.b_hu, self.W_hsigma, self.b_hsigma, \
                        self.W_zh, self.b_zh]

        self.W_hy = init_weights((self.dim_h, self.dict_size), "W_hy")
        self.b_hy = init_bias(self.dict_size, "b_hy")
        self.params += [self.W_hy, self.b_hy]


        if  self.is_predicting:
            # encoder
            h_enc = T.nnet.relu(T.dot(self.y_rev_i, self.W_xh) + self.b_xh)
        
            self.mu = T.dot(h_enc, self.W_hu) + self.b_hu
            log_var = T.dot(h_enc, self.W_hsigma) + self.b_hsigma
            self.var = T.exp(log_var)
            self.sigma = T.sqrt(self.var)

            eps = srng.normal(self.mu.shape)
            self.z = self.mu + self.sigma * 0

            # decoder
            h_dec = T.nnet.relu(T.dot(self.z, self.W_zh) + self.b_zh)
            self.review_i = T.nnet.softmax(T.dot(h_dec, self.W_hy) + self.b_hy)

            xi_emb = self.z

        else:
            xi_flat = self.xi.flatten()
            xi_emb = self.emb_i[xi_flat, :]
            xi_emb = T.reshape(xi_emb, (self.batch_size, self.dim_z))

            h_dec = T.nnet.relu(T.dot(xi_emb, self.W_zh) + self.b_zh)
            self.review_i = T.nnet.softmax(T.dot(h_dec, self.W_hy) + self.b_hy)

        # predict 1: rating
        self.W_r_uh = init_weights((self.dim_x, self.dim_h), "W_r_uh")
        self.W_r_ih = init_weights((self.dim_z, self.dim_h), "W_r_ih")
        self.b_r_uih = init_bias(self.dim_h, "b_r_uih")
        
        self.W_r_hr = init_weights((self.dim_h, 1), "W_r_hr")
        self.b_r_hr = init_bias(1, "b_r_hr")
        
        self.params += [self.W_r_uh, self.W_r_ih,  self.b_r_uih, self.W_r_hr, self.b_r_hr]

        ## layer1
        h_r = T.nnet.relu(T.dot(xu_emb, self.W_r_uh) + T.dot(xi_emb, self.W_r_ih) + self.b_r_uih)

        '''
        ## layer2
        self.W_r_hh2 = init_weights((self.dim_h, self.dim_h), "W_r_hh2")
        self.b_r_hh2 = init_bias(self.dim_h, "b_r_hh2")
        self.params += [self.W_r_hh2, self.b_r_hh2]

        h_r2 = T.nnet.relu(T.dot(h_r, self.W_r_hh2) + self.b_r_hh2)
        drop_mask = srng.binomial(n = 1, p = 1-p, size = h_r2.shape, dtype = theano.config.floatX)
        h_r2 = T.switch(T.eq(self.is_train, 1), h_r2 * drop_mask, h_r2 * (1 - p))
        
        ## layer 3
        self.W_r_hh3 = init_weights((self.dim_h, self.dim_h), "W_r_hh3")
        self.b_r_hh3 = init_bias(self.dim_h, "b_r_hh3")
        self.params += [self.W_r_hh3, self.b_r_hh3]

        h_r3 = T.nnet.relu(T.dot(h_r2, self.W_r_hh3) + self.b_r_hh3)
        drop_mask = srng.binomial(n = 1, p = 1-p, size = h_r3.shape, dtype = theano.config.floatX)
        h_r3 = T.switch(T.eq(self.is_train, 1), h_r3 * drop_mask, h_r3 * (1 - p))
        
        '''
        self.r_pred = T.dot(h_r, self.W_r_hr) + self.b_r_hr
    
        if self.is_predicting:
            inputs = [self.xu, self.xi, self.y_rev_i, self.batch_size]
            self.encode = theano.function(inputs = inputs,
                    outputs = [self.r_pred, self.review_i],
                    givens = {self.is_train : np.cast['int32'](0)},
                    on_unused_input = 'ignore')
    
    def multivariate_bernoulli(self, y_pred, y_true):
        return T.sum(y_true * T.log(y_pred) + (1 - y_true) * T.log(1 - y_pred), axis=1)

    def kld(self, mu, var):
        return 0.5 * T.sum(1 + T.log(var) - mu**2 - var, axis=1)

    def l2_norm(self):
        return T.sum(self.emb_u**2) + T.sum(self.emb_i**2)

    def cost_mse(self, pred, label):
        cost = T.mean((pred - label) ** 2)
        return cost

    def cost_mae(self, pred, label):
        return T.mean(T.abs_(pred - label))

    def cost_nll(self, pred, label):
        cost = -T.log(pred) * label
        cost = T.mean(T.sum(cost, axis = 1))
        return cost

    def define_train_funcs(self, modules, consts, options):
        mse = self.cost_mse(self.r_pred, self.y_rating.reshape((self.batch_size, 1)))
        kld = -T.mean(self.kld(self.mu, self.var))
        nll = self.cost_nll(self.review_i, self.y_rev_i)
        
        cost = mse + kld + nll

        grad_ignore_set = set(["emb_i"])
        gparams = []
        fd_params = []
        for param in self.params:
            if param.name not in grad_ignore_set:
                gparams.append(T.clip(T.grad(cost, param), -10, 10))
                fd_params.append(param)
        sub_gparams = []
        for param in self.sub_params:
            sub_gparams.append(T.clip(T.grad(cost, param[1]), -10, 10))

        optimizer = eval(self.optimizer)
        updates = optimizer(fd_params, gparams, self.sub_params, sub_gparams, self.lr)
        ##### write memory
        updates.append((self.emb_i, T.set_subtensor(self.emb_i[self.xi,:], self.z)))

        inputs = [self.xu, self.xi, self.y_rating, self.y_rev_i, self.batch_size]
        
        self.train = theano.function(
                inputs = inputs + [self.lr],
                outputs = [cost, mse, kld, nll, self.r_pred],
                updates = updates,
                givens = {self.is_train : np.cast['int32'](1)},
                on_unused_input = 'ignore')

