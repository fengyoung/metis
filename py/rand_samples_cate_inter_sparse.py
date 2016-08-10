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


def gen_x(y0, dim_infos): 
	x = [randBinomial(1, 0.5)]
	for dim_info in dim_infos:
		r = randUni(0.0, 1.0)
		if r <= dim_info[1]: 
			x.append(0)
		else:
			if y0 == 0:	# negative sample
				if dim_info[0] == 0:	# negative interaction
					if x[len(x)-1] == 1:
						x.append(1)
					else:	
						x.append(randBinomial(1,0.5))
				else:					# positive interaction 
					if x[len(x)-1] == 1:
						x.append(0)
					else:	
						x.append(randBinomial(1,0.5))
			else:	# positive sample
				if dim_info[0] == 0:	# negative interaction
					if x[len(x)-1] == 1:
						x.append(0)
					else:	
						x.append(randBinomial(1,0.5))
				else:					# positive interaction 
					if x[len(x)-1] == 1:
						x.append(1)
					else:	
						x.append(randBinomial(1,0.5))
	return x	
		
		
def gen_samples(dims, train_cnt, test_cnt, out_prefix): 
	dim_infos = []
	for i in range(dims-1): 
		dim_infos.append([randBinomial(1,0.5), randUni(0.5, 0.8)])	# [[interaction with before relation to y, missing degree] ... ]
		
	fp_train = open(out_prefix + '_sparse.train', 'w')
	for k in range(train_cnt): 
		y = randLabels(2)
		x = gen_x(y[0], dim_infos)	
		fp_train.write(reduce(lambda x, y: str(x) + ',' + str(y), y) + ';' + reduce(lambda x, y: str(x) + ',' + str(y), x) + '\n')
	fp_train.close()  
	
	fp_test = open(out_prefix + '_sparse.test', 'w')
	for k in range(test_cnt): 
		y = randLabels(2)
		x = gen_x(y[0], dim_infos)	
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



