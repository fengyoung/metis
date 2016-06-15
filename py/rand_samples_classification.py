#! /usr/bin/python
# encoding=utf8

#
# Generate samples randomly for classification
#
# Output: 
# (1) training data set
# (2) testing data set

import sys
import os
import numpy as np


################################################################
# Random Functions 

def randNormal(mu, sigma, bot = -1.0, top = 1.0):
    val = sigma * np.random.randn() + mu
    if val < bot:
        return bot
    elif val > top:
        return top
    else:
        return val


def randUni(bot = -1.0, top = 1.0):
    return np.random.random() * (top - bot) + bot


def randBinomial(n, p): 
    c = 0    
    for k in range(n): 
        if randUni(0.0, 1.0) < p:
            c += 1
    return c


def randInt(bot, top): 
    return np.random.randint(bot, top)


def randLabels(y_cnt):
    labels = [0] * y_cnt    
    off = np.random.randint(0, y_cnt)     
    labels[off] = 1
    return labels

# 
################################################################


def samples_linear(dims, train_cnt, test_cnt, out_prefix): 
    '''
    '''
    dim_info_s = []    # [[pos_mu, neg_mu, sigma], ...] 
    for i in range(dims): 
        pos_mu = randUni(-1.5, 1.5)
        neg_mu = 0.0 - pos_mu    
        sigma = abs(pos_mu - neg_mu) * 2.0
        dim_info_s.append([pos_mu, neg_mu, sigma])

    def gen_x(y0, dim_info_s): 
        x = []
        for diminfo in dim_info_s:
            if y0 == 0:    
                v = randNormal(diminfo[1], diminfo[2], -3.0, 3.0)        
            else:
                v = randNormal(diminfo[0], diminfo[2], -3.0, 3.0)        
            x.append(v)
        return x

    fp_train = open(out_prefix + '_linear.train', 'w')
    for k in range(train_cnt): 
        y = randLabels(2)
        x = gen_x(y[0], dim_info_s)    
        fp_train.write(reduce(lambda x, y: str(x) + ',' + str(y), y) + ';' + reduce(lambda x, y: str(x) + ',' + str(y), x) + '\n')
    fp_train.close()    
    
    fp_test = open(out_prefix + '_linear.test', 'w')
    for k in range(test_cnt): 
        y = randLabels(2)
        x = gen_x(y[0], dim_info_s)    
        fp_test.write(reduce(lambda x, y: str(x) + ',' + str(y), y) + ';' + reduce(lambda x, y: str(x) + ',' + str(y), x) + '\n')
    fp_test.close()    

    
def samples_nonlinear(dims, train_cnt, test_cnt, out_prefix): 
    '''
    '''
    dim_info_s = []    # [[pos_mu, neg_mu, sigma, s], ...] 
    for i in range(dims): 
        pos_mu = randUni(1.0, 10.0)
        neg_mu = 11.0 - pos_mu    
        sigma = abs(pos_mu - neg_mu) * 2.0
        s = randInt(0, 7) + 1
        dim_info_s.append([pos_mu, neg_mu, sigma, s])

    def gen_x(y0, dim_info_s): 
        x = []
        for diminfo in dim_info_s:
            if y0 == 0:    
                vv = randNormal(diminfo[1], diminfo[2], 1.0, 10.0)
            else:
                vv = randNormal(diminfo[0], diminfo[2], 1.0, 10.0)
            v = abs(vv) ** (1.0 / diminfo[3])
            if int(diminfo[3]) % 2 == 0:
                if randBinomial(1, 0.5) == 1: 
                    v *= -1.0
#            v = 0.0    
#            for s in range(1, diminfo[3]+1):
#                v += (vv ** s) / s
#                v += abs(vv) ** (1.0 / s) 
            x.append(v)
        return x

    fp_train = open(out_prefix + '_nonlinear.train', 'w')
    for k in range(train_cnt): 
        y = randLabels(2)
        x = gen_x(y[0], dim_info_s)    
        fp_train.write(reduce(lambda x, y: str(x) + ',' + str(y), y) + ';' + reduce(lambda x, y: str(x) + ',' + str(y), x) + '\n')
    fp_train.close()    
    
    fp_test = open(out_prefix + '_nonlinear.test', 'w')
    for k in range(test_cnt): 
        y = randLabels(2)
        x = gen_x(y[0], dim_info_s)    
        fp_test.write(reduce(lambda x, y: str(x) + ',' + str(y), y) + ';' + reduce(lambda x, y: str(x) + ',' + str(y), x) + '\n')
    fp_test.close()    


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print "Usage: %s <\"linear\"/\"nonlinear\"> <dims> <train_cnt> <test_cnt> <out_prefix>" % sys.argv[0]
        exit(-1)

    dims = int(sys.argv[2])
    train_cnt = int(sys.argv[3])
    test_cnt = int(sys.argv[4])
    out_prefix = sys.argv[5]

    if(sys.argv[1] == "linear"):
        samples_linear(dims, train_cnt, test_cnt, out_prefix)
    elif(sys.argv[1] == "nonlinear"):
        samples_nonlinear(dims, train_cnt, test_cnt, out_prefix)
    else:
        print "unsupported type!"
        exit(-2)    
    
    exit(0)


