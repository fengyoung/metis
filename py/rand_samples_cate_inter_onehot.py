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

    
    
def gen_samples(dims, train_cnt, test_cnt, out_prefix): 
    if dims % 2 == 0:
        d = dims / 2
    else:    
        d = (dims + 1) / 2

    inter_info_s = []
    for i in range(d): 
        inter_info_s.append([randInt(0, 2), randUni(0.3, 0.9)])        # [[inter_related, confusion], ...]

    def gen_x(y0, inter_info_s):
        x = []
        if y0 == 0:  
            for inter_info in inter_info_s:
                r = randUni(0.0, 1.0)
                if inter_info[0] == 0: # negative
                    if r > (1.0 + inter_info[1]) / 2.0:        
                        xx = 1    
                    else:
                        xx = randInt(0, 2)
                else:    # positive
                    if r > (1.0 + inter_info[1]) / 2.0:        
                        xx = 0
                    else:
                        xx = randInt(0, 2)
                if xx == 0:
                    x1 = randInt(0, 2)
                    if x1 == 0:
                        x2 = randInt(0, 2)
                    else:
                        x2 = 0
                else:
                    x1 = randInt(0, 2)
                    if x1 == 0:
                        x2 = randInt(0, 2)
                    else:
                        x2 = 1
                x.append(x1)    
                x.append(x2)    
        else:    
            for inter_info in inter_info_s:
                r = randUni(0.0, 1.0)
                if inter_info[0] == 0: # negative
                    if r > (1.0 + inter_info[1]) / 2.0:        
                        xx = 0
                    else:
                        xx = randInt(0, 2)
                else:    # positive
                    if r > (1.0 + inter_info[1]) / 2.0:        
                        xx = 1
                    else:
                        xx = randInt(0, 2)
                if xx == 0:
                    x1 = randInt(0, 2)
                    if x1 == 0:
                        x2 = randInt(0, 2)
                    else:
                        x2 = 0
                else:
                    x1 = randInt(0, 2)
                    if x1 == 0:
                        x2 = randInt(0, 2)
                    else:
                        x2 = 1
                x.append(x1)    
                x.append(x2)    
        return x    

    fp_train = open(out_prefix + '_onehot.train', 'w')
    for k in range(train_cnt): 
        y = randLabels(2)
        x = gen_x(y[0], inter_info_s)    
        fp_train.write(reduce(lambda x, y: str(x) + ',' + str(y), y) + ';' + reduce(lambda x, y: str(x) + ',' + str(y), x) + '\n')
    fp_train.close()  
    
    fp_test = open(out_prefix + '_onehot.test', 'w')
    for k in range(test_cnt): 
        y = randLabels(2)
        x = gen_x(y[0], inter_info_s)    
        fp_test.write(reduce(lambda x, y: str(x) + ',' + str(y), y) + ';' + reduce(lambda x, y: str(x) + ',' + str(y), x) + '\n')
    fp_test.close()    



if __name__ == "__main__":
    if len(sys.argv) != 5:
        print "Usage: %s <dims> <train_cnt> <test_cnt> <out_prefix>" % sys.argv[0]
        exit(-1)

    dims = int(sys.argv[1])
    train_cnt = int(sys.argv[2])
    test_cnt = int(sys.argv[3])
    out_prefix = sys.argv[4]

    gen_samples(dims, train_cnt, test_cnt, out_prefix)
    
    exit(0)
